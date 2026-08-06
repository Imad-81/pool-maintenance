"""External ingestion endpoint — same column-mapping pipeline as upload, but
via REST with optional token auth."""

import csv
import io
import json
import logging
import traceback
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Header, Request, UploadFile, File
from sqlmodel import Session

from backend.store.schema import get_session
from backend.store import repo
from backend.api.upload import _auto_detect_mapping, _safe_float, _parse_date_flexible
from backend.settings import settings

router = APIRouter(prefix="/api/ingest", tags=["ingest"])
log = logging.getLogger(__name__)


def _check_token(authorization: Optional[str] = Header(None)):
    """Optional token guard — skipped if ingestion_token is not set."""
    token = settings.ingestion_token
    if token is None:
        return
    if not authorization:
        raise HTTPException(401, "Authorization header required for ingestion")
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != token:
        raise HTTPException(403, "Invalid ingestion token")


@router.post("/readings")
def ingest_readings(
    payload: dict,
    session: Session = Depends(get_session),
    authorization: Optional[str] = Header(None),
):
    """Accept a JSON payload of readings with explicit column mapping.

    Body:
        {"mapping": {"pool_id": "colA", ...}, "rows": [{...}, ...]}

    If `ingestion_token` is set in env, an `Authorization: Bearer <token>`
    header is required.
    """
    _check_token(authorization)
    mapping = payload.get("mapping", {})
    rows_raw = payload.get("rows", [])
    if "pool_id" not in mapping or "reading_date" not in mapping:
        raise HTTPException(400, "pool_id and reading_date columns must be mapped")
    if not any(k in mapping for k in ("ph", "free_chlorine", "turbidity")):
        raise HTTPException(400, "At least one measurement column required")

    entries, skipped = [], []
    for i, raw in enumerate(rows_raw):
        entry = {}
        pid = str(raw.get(mapping["pool_id"], "")).strip()
        if not pid or pid == "nan":
            skipped.append({"row": i + 1, "reason": "Missing pool ID"}); continue
        entry["pool_id"] = pid.lower()
        rd = _parse_date_flexible(raw.get(mapping["reading_date"]))
        if not rd:
            skipped.append({"row": i + 1, "reason": "Invalid date"}); continue
        entry["reading_date"] = rd
        for f2 in ("ph", "free_chlorine", "turbidity"):
            entry[f2] = _safe_float(raw.get(mapping[f2])) if f2 in mapping else None
        if "pool_volume_m3" in mapping:
            entry["pool_volume_m3"] = _safe_float(raw.get(mapping["pool_volume_m3"]))
        if "community_name" in mapping:
            cn = str(raw.get(mapping["community_name"], "")).strip()
            entry["community_name"] = cn if cn != "nan" else ""
        if all(entry.get(k) is None for k in ("ph", "free_chlorine", "turbidity")):
            skipped.append({"row": i + 1, "reason": "No valid measurements"}); continue
        entries.append(entry)

    if not entries:
        raise HTTPException(400, "No valid rows in payload")
    n = repo.upsert_readings_batch(session, entries, source="ingest")
    repo.add_ingest_log(session, source="ingest_api",
                        pool_count=len({r["pool_id"] for r in entries}),
                        row_count=len(entries), skipped_count=len(skipped),
                        detail_json=json.dumps({"skipped": skipped[:50]}))
    session.commit()
    return {"success": True, "loaded_rows": n, "skipped_count": len(skipped)}


@router.post("/readings/file")
def ingest_file(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    authorization: Optional[str] = Header(None),
):
    """Accept an Excel/CSV file via the ingestion endpoint. Same auto-detect
    column mapping as the upload flow."""
    _check_token(authorization)
    return {"status": "TODO — file ingestion endpoint"}  # deferred for now