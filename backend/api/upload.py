"""
Stateless file upload, column auto-mapping, and manual reading endpoints with typed Pydantic schemas.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import uuid
import tempfile
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from prisma import Prisma

from backend.store.client import get_db
from backend.store import repo
from backend.settings import settings

router = APIRouter(tags=["upload"])
manual_router = APIRouter(prefix="/api/readings", tags=["readings"])
log = logging.getLogger("backend.api.upload")

COLUMN_PATTERNS = {
    "pool_id":        ["pool_id", "pool id", "poolid", "pool", "id", "piscina", "nombre"],
    "reading_date":   ["reading_date", "date", "fecha", "datetime", "timestamp", "fecha_lectura"],
    "ph":             ["ph", "p.h."],
    "free_chlorine":  ["free_chlorine", "chlorine", "cl", "cloro", "free_cl", "cloro_libre", "free chlorine"],
    "turbidity":      ["turbidity", "turb", "ntu", "turbidez"],
    "pool_volume_m3": ["pool_volume_m3", "volume", "vol", "volumen", "m3", "pool_volume"],
    "community_name": ["community_name", "community", "comunidad", "urbanizacion", "urbanización", "location"],
    "hypochlorite_dosing_pct": ["hypochlorite_dosing_pct", "dosing_pct", "cl_pct"],
    "hypochlorite_dosing_hours": ["hypochlorite_dosing_hours", "dosing_hours", "cl_hours"],
    "water_temperature": ["water_temperature", "temperature", "temp", "temp_agua"],
}

# Temporary directory for multi-worker safe pending uploads (cleaned after ingestion)
UPLOAD_TEMP_DIR = Path(tempfile.gettempdir()) / "pool_uploads"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_detect_mapping(columns: list[str]) -> dict:
    cols_lower = {c: c.lower().strip() for c in columns}
    mapping, used = {}, set()
    for internal, patterns in COLUMN_PATTERNS.items():
        best_col, best_score = None, 0.0
        for col, col_low in cols_lower.items():
            if col in used:
                continue
            for pat in patterns:
                if pat in col_low or col_low in pat:
                    score = 0.9 + len(pat) / 100
                    if score > best_score:
                        best_score, best_col = score, col
                ratio = SequenceMatcher(None, pat, col_low).ratio()
                if ratio > 0.7 and ratio > best_score:
                    best_score, best_col = ratio, col
        if best_col and best_score > 0.5:
            mapping[internal] = best_col
            used.add(best_col)
    return mapping


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if pd.notna(val) else None
    s = str(val).strip().replace(",", ".")
    if not s or s.lower() == "nan":
        return None
    try:
        f = float(s)
        return None if pd.isna(f) else f
    except Exception:
        return None


def _parse_date_flexible(val) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    if isinstance(val, datetime):
        return val
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    formats = (
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        if pd.notna(dt):
            return dt.to_pydatetime()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------

class UploadPreviewResponse(BaseModel):
    upload_id: str
    filename: str
    total_rows: int
    columns: List[str]
    suggested_mapping: Dict[str, str]
    preview: List[Dict[str, Any]]


class MapColumnsRequest(BaseModel):
    upload_id: str
    mapping: Dict[str, str]


class ManualReadingRequest(BaseModel):
    pool_id: str = Field(..., description="Pool identifier (e.g. 'Cabo Verde (19)')")
    reading_date: str = Field(..., description="Date or datetime of measurement")
    ph: Optional[float] = Field(None, ge=0.0, le=14.0, description="pH level (0-14)")
    free_chlorine: Optional[float] = Field(None, ge=0.0, le=20.0, description="Free Chlorine (mg/L)")
    turbidity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Turbidity (NTU)")
    pool_volume_m3: Optional[float] = Field(None, gt=0.0, description="Pool volume in m³")
    community_name: Optional[str] = None
    water_temperature: Optional[float] = None
    hypochlorite_dosing_pct: Optional[float] = None
    hypochlorite_dosing_hours: Optional[float] = None


# ---------------------------------------------------------------------------
# Upload Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/upload", response_model=UploadPreviewResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV or Excel file for preview and automated column detection."""
    fname = (file.filename or "").lower()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds maximum allowed size of {settings.max_upload_size_mb} MB")

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

            first_line = text.split("\n")[0]
            sep = "\t" if "\t" in first_line else (";" if ";" in first_line else ",")
            df = pd.read_csv(io.StringIO(text), sep=sep)
        else:
            raise HTTPException(400, f"Unsupported file type. Please upload a .csv or .xlsx file.")

        if df.empty or len(df.columns) < 2:
            raise HTTPException(400, "Uploaded file appears empty or contains too few columns.")

        # Persist to disk with unique token for stateless multi-worker access
        upload_id = str(uuid.uuid4())
        cache_path = UPLOAD_TEMP_DIR / f"{upload_id}.json"
        df.to_json(cache_path, orient="split")

        cols = list(df.columns.astype(str))
        suggested = _auto_detect_mapping(cols)
        preview_rows = df.head(5).fillna("").astype(str).to_dict(orient="records")

        return UploadPreviewResponse(
            upload_id=upload_id,
            filename=file.filename or "unknown",
            total_rows=len(df),
            columns=cols,
            suggested_mapping=suggested,
            preview=preview_rows,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("File parse error: %s", e)
        raise HTTPException(400, f"Could not parse file: {e}")


@router.post("/api/map-columns")
async def map_columns(
    payload: MapColumnsRequest,
    client: Prisma = Depends(get_db),
):
    """Confirm column mappings and import parsed readings into PostgreSQL."""
    cache_path = UPLOAD_TEMP_DIR / f"{payload.upload_id}.json"
    if not cache_path.exists():
        raise HTTPException(400, "Upload session expired or not found. Please upload the file again.")

    try:
        df = pd.read_json(cache_path, orient="split")
    except Exception as e:
        raise HTTPException(500, f"Failed to load cached upload: {e}")

    mapping = payload.mapping
    if "pool_id" not in mapping or "reading_date" not in mapping:
        raise HTTPException(400, "Pool ID and Reading Date columns are strictly required.")
    if not any(k in mapping for k in ("ph", "free_chlorine", "turbidity")):
        raise HTTPException(400, "At least one water measurement column (pH, Chlorine, Turbidity) must be mapped.")

    rows, skipped = [], []
    for idx, raw in df.iterrows():
        entry = {}
        pid = str(raw.get(mapping["pool_id"], "")).strip()
        if not pid or pid == "nan":
            skipped.append({"row": int(idx) + 2, "reason": "Missing Pool ID"})
            continue
        entry["pool_id"] = pid

        rd = _parse_date_flexible(raw.get(mapping["reading_date"]))
        if not rd:
            skipped.append({"row": int(idx) + 2, "reason": "Invalid or unparseable reading date"})
            continue
        entry["reading_date"] = rd

        for param in ("ph", "free_chlorine", "turbidity", "hypochlorite_dosing_pct", "hypochlorite_dosing_hours", "water_temperature"):
            if param in mapping:
                entry[param] = _safe_float(raw.get(mapping[param]))
            else:
                entry[param] = None

        if "pool_volume_m3" in mapping:
            entry["pool_volume_m3"] = _safe_float(raw.get(mapping["pool_volume_m3"]))
        if "community_name" in mapping:
            cn = str(raw.get(mapping["community_name"], "")).strip()
            entry["community_name"] = cn if cn != "nan" else ""

        if all(entry.get(k) is None for k in ("ph", "free_chlorine", "turbidity")):
            skipped.append({"row": int(idx) + 2, "reason": "No valid chemical measurements present"})
            continue

        rows.append(entry)

    if not rows:
        raise HTTPException(400, "No valid reading rows could be parsed from the file.")

    # 1. Upsert pools if pool metadata present
    unique_pools = {r["pool_id"]: r.get("community_name", "") for r in rows}
    for pid, cn in unique_pools.items():
        await repo.upsert_pool({"pool_id": pid, "community_name": cn}, client=client)

    # 2. Upsert readings batch
    n = await repo.upsert_readings_batch(rows, source="upload", client=client)

    # 3. Log ingest audit
    await repo.add_ingest_log(
        source="upload",
        filename=f"upload_{payload.upload_id[:8]}",
        pool_count=len(unique_pools),
        row_count=len(rows),
        skipped_count=len(skipped),
        detail_json=json.dumps({"skipped": skipped[:50]}),
        client=client,
    )

    # Cleanup temp file
    try:
        cache_path.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "success": True,
        "loaded_rows": n,
        "pool_count": len(unique_pools),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
    }


# ---------------------------------------------------------------------------
# Manual Reading Endpoint
# ---------------------------------------------------------------------------

@manual_router.post("")
async def add_manual_reading(
    payload: ManualReadingRequest,
    client: Prisma = Depends(get_db),
):
    """Accept single technician field measurement."""
    rd = _parse_date_flexible(payload.reading_date)
    if not rd:
        raise HTTPException(400, f"Invalid date format '{payload.reading_date}'. Use ISO 8601 or YYYY-MM-DD.")

    if payload.ph is None and payload.free_chlorine is None and payload.turbidity is None:
        raise HTTPException(400, "At least one chemical parameter (pH, Free Chlorine, Turbidity) is required.")

    # Ensure Pool exists
    pool_id = payload.pool_id.strip()
    pool_record = {
        "pool_id": pool_id,
        "community_name": payload.community_name or "",
    }
    if payload.pool_volume_m3:
        pool_record["pool_volume_m3"] = payload.pool_volume_m3
    await repo.upsert_pool(pool_record, client=client)

    reading_entry = {
        "pool_id": pool_id,
        "reading_date": rd,
        "ph": payload.ph,
        "free_chlorine": payload.free_chlorine,
        "turbidity": payload.turbidity,
        "community_name": payload.community_name or "",
        "water_temperature": payload.water_temperature,
        "hypochlorite_dosing_pct": payload.hypochlorite_dosing_pct,
        "hypochlorite_dosing_hours": payload.hypochlorite_dosing_hours,
    }
    n = await repo.upsert_readings_batch([reading_entry], source="manual", client=client)

    # Ingest log
    await repo.add_ingest_log(
        source="manual",
        filename=None,
        pool_count=1,
        row_count=1,
        skipped_count=0,
        client=client,
    )

    return {"success": True, "pool_id": pool_id, "rows": n}