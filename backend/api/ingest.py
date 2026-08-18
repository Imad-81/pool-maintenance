"""
External ingestion endpoint for automated telemetric or API feeds with optional Bearer token auth.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from prisma import Prisma

from backend.store.client import get_db
from backend.store import repo
from backend.deps import get_prediction_service, get_weather_lookup_provider
from backend.api.upload import _auto_detect_mapping, _parse_date_flexible, _safe_float
from backend.settings import settings

router = APIRouter(prefix="/api/ingest", tags=["ingest"])
log = logging.getLogger("backend.api.ingest")



def _check_token(authorization: Optional[str] = Header(None)) -> None:
    """Validate Bearer token if `ingestion_token` is configured."""
    token = settings.ingestion_token
    if token is None:
        return
    if not authorization:
        raise HTTPException(401, "Authorization header required for ingestion API.")
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != token:
        raise HTTPException(403, "Invalid ingestion token.")


class IngestJSONRequest(BaseModel):
    mapping: Optional[Dict[str, str]] = Field(
        None, description="Optional column mapping overrides (e.g. {'pool_id': 'id', ...})"
    )
    rows: List[Dict[str, Any]] = Field(..., description="List of raw reading objects")


@router.post("/readings")
async def ingest_readings(
    payload: IngestJSONRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    client: Prisma = Depends(get_db),
):
    """Accept a JSON payload of readings with optional or auto-detected column mappings."""
    _check_token(authorization)

    rows_raw = payload.rows
    if not rows_raw:
        raise HTTPException(400, "Payload rows list cannot be empty.")

    # Determine mapping
    mapping = payload.mapping or {}
    if not mapping:
        sample_keys = list(rows_raw[0].keys())
        mapping = _auto_detect_mapping(sample_keys)

    if "pool_id" not in mapping or "reading_date" not in mapping:
        raise HTTPException(400, "Could not resolve 'pool_id' and 'reading_date' columns.")

    entries, skipped = [], []
    for i, raw in enumerate(rows_raw):
        entry = {}
        pid = str(raw.get(mapping["pool_id"], "")).strip()
        if not pid or pid.lower() == "nan":
            skipped.append({"row": i + 1, "reason": "Missing Pool ID"})
            continue
        entry["pool_id"] = pid

        rd = _parse_date_flexible(raw.get(mapping["reading_date"]))
        if not rd:
            skipped.append({"row": i + 1, "reason": "Invalid reading date"})
            continue
        entry["reading_date"] = rd

        for f2 in ("ph", "free_chlorine", "turbidity", "hypochlorite_dosing_pct", "hypochlorite_dosing_hours", "water_temperature"):
            entry[f2] = _safe_float(raw.get(mapping[f2])) if f2 in mapping else None

        if "pool_volume_m3" in mapping:
            entry["pool_volume_m3"] = _safe_float(raw.get(mapping["pool_volume_m3"]))
        if "community_name" in mapping:
            cn = str(raw.get(mapping["community_name"], "")).strip()
            entry["community_name"] = cn if cn.lower() != "nan" else ""

        if all(entry.get(k) is None for k in ("ph", "free_chlorine", "turbidity")):
            skipped.append({"row": i + 1, "reason": "No valid chemical measurements"})
            continue

        entries.append(entry)

    if not entries:
        raise HTTPException(400, "No valid reading rows parsed from JSON payload.")

    # Upsert pools & readings
    unique_pools = {r["pool_id"]: r.get("community_name", "") for r in entries}
    for pid, cn in unique_pools.items():
        await repo.upsert_pool({"pool_id": pid, "community_name": cn}, client=client)

    n = await repo.upsert_readings_batch(entries, source="ingest_api", client=client)

    await repo.add_ingest_log(
        source="ingest_api",
        filename=None,
        pool_count=len(unique_pools),
        row_count=len(entries),
        skipped_count=len(skipped),
        detail_json=json.dumps({"skipped": skipped[:50]}),
        client=client,
    )

    # Refresh predictions for affected pools
    try:
        svc = get_prediction_service(request)
        wx_lookup = await get_weather_lookup_provider(request)
        as_of = datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
        await repo.compute_and_store_daily_predictions(as_of, svc, wx_lookup, client=client, pool_ids=list(unique_pools.keys()))
    except Exception as e:
        log.warning("Post-JSON-ingest daily prediction refresh: %s", e)

    return {
        "success": True,
        "loaded_rows": n,
        "pool_count": len(unique_pools),
        "skipped_count": len(skipped),
    }


@router.post("/readings/file")
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    client: Prisma = Depends(get_db),
):
    """Accept an Excel or CSV file via REST API and automatically parse and import it."""
    _check_token(authorization)
    fname = (file.filename or "").lower()

    content = await file.read()
    try:
        if fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        elif fname.endswith(".csv"):
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    text = content.decode(enc)
                    break
                except Exception:
                    pass
            else:
                text = content.decode("utf-8", errors="replace")
            first = text.split("\n")[0]
            sep = "\t" if "\t" in first else (";" if ";" in first else ",")
            df = pd.read_csv(io.StringIO(text), sep=sep)
        else:
            raise HTTPException(400, "Unsupported file format. Please upload .csv or .xlsx.")

        cols = list(df.columns.astype(str))
        mapping = _auto_detect_mapping(cols)
        if "pool_id" not in mapping or "reading_date" not in mapping:
            raise HTTPException(400, f"Could not auto-detect pool_id and date from columns: {cols}")

        entries, skipped = [], []
        for idx, raw in df.iterrows():
            entry = {}
            pid = str(raw.get(mapping["pool_id"], "")).strip()
            if not pid or pid.lower() == "nan":
                skipped.append({"row": int(idx) + 2, "reason": "Missing Pool ID"})
                continue
            entry["pool_id"] = pid

            rd = _parse_date_flexible(raw.get(mapping["reading_date"]))
            if not rd:
                skipped.append({"row": int(idx) + 2, "reason": "Invalid reading date"})
                continue
            entry["reading_date"] = rd

            for param in ("ph", "free_chlorine", "turbidity", "hypochlorite_dosing_pct", "hypochlorite_dosing_hours", "water_temperature"):
                if param in mapping:
                    entry[param] = _safe_float(raw.get(mapping[param]))

            if "pool_volume_m3" in mapping:
                entry["pool_volume_m3"] = _safe_float(raw.get(mapping["pool_volume_m3"]))
            if "community_name" in mapping:
                cn = str(raw.get(mapping["community_name"], "")).strip()
                entry["community_name"] = cn if cn.lower() != "nan" else ""

            if all(entry.get(k) is None for k in ("ph", "free_chlorine", "turbidity")):
                skipped.append({"row": int(idx) + 2, "reason": "No valid chemical measurements"})
                continue

            entries.append(entry)

        if not entries:
            raise HTTPException(400, "No valid reading rows could be parsed from the file.")

        unique_pools = {r["pool_id"]: r.get("community_name", "") for r in entries}
        for pid, cn in unique_pools.items():
            await repo.upsert_pool({"pool_id": pid, "community_name": cn}, client=client)

        n = await repo.upsert_readings_batch(entries, source="ingest_file", client=client)

        await repo.add_ingest_log(
            source="ingest_file",
            filename=file.filename,
            pool_count=len(unique_pools),
            row_count=len(entries),
            skipped_count=len(skipped),
            detail_json=json.dumps({"skipped": skipped[:50]}),
            client=client,
        )

        # Refresh predictions for affected pools
        try:
            svc = get_prediction_service(request)
            wx_lookup = await get_weather_lookup_provider(request)
            as_of = datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
            await repo.compute_and_store_daily_predictions(as_of, svc, wx_lookup, client=client, pool_ids=list(unique_pools.keys()))
        except Exception as e:
            log.warning("Post-file-ingest daily prediction refresh: %s", e)

        return {
            "success": True,
            "filename": file.filename,
            "loaded_rows": n,
            "pool_count": len(unique_pools),
            "skipped_count": len(skipped),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Ingest file error: %s", e)
        raise HTTPException(400, f"Could not process ingestion file: {e}")