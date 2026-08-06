"""Upload and manual-reading endpoints."""

import csv
import io
import json
import logging
import traceback
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlmodel import Session

from backend.store.schema import get_session
from backend.store import repo

router = APIRouter(tags=["upload"])
manual_router = APIRouter(prefix="/api/readings", tags=["readings"])
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column auto-detection (ported from prototype_ui/app.py)
# ---------------------------------------------------------------------------
COLUMN_PATTERNS = {
    "pool_id":        ["pool_id", "pool id", "poolid", "pool", "id", "piscina", "nombre"],
    "reading_date":   ["reading_date", "date", "fecha", "datetime", "timestamp", "fecha_lectura"],
    "ph":             ["ph", "p.h."],
    "free_chlorine":  ["free_chlorine", "chlorine", "cl", "cloro", "free_cl", "cloro_libre", "free chlorine"],
    "turbidity":      ["turbidity", "turb", "ntu", "turbidez"],
    "pool_volume_m3": ["pool_volume_m3", "volume", "vol", "volumen", "m3", "pool_volume"],
    "community_name": ["community_name", "community", "comunidad", "urbanizacion", "urbanización", "location"],
}
_pending_upload: dict = {}


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
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_date_flexible(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    s = str(val).strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except Exception:
            pass
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        if pd.notna(dt):
            return dt.isoformat()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/api/upload")
def upload_file(file: UploadFile = File(...)):
    fname = (file.filename or "").lower()
    try:
        if fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file, engine="openpyxl")
        elif fname.endswith(".csv"):
            raw = file.file.read()
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    text = raw.decode(enc)
                    break
                except Exception:
                    pass
            else:
                text = raw.decode("utf-8", errors="replace")
            first = text.split("\n")[0]
            sep = "\t" if "\t" in first else (";" if ";" in first else ",")
            df = pd.read_csv(io.StringIO(text), sep=sep)
        else:
            raise HTTPException(400, f"Unsupported type: {os.path.splitext(fname)[1]}. Use .csv or .xlsx")
        if df.empty or len(df.columns) < 2:
            raise HTTPException(400, "File appears empty or has too few columns.")
        _pending_upload["df"] = df
        _pending_upload["filename"] = file.filename
        _pending_upload["columns"] = list(df.columns.astype(str))
        _pending_upload["suggested_mapping"] = _auto_detect_mapping(_pending_upload["columns"])
        return {
            "columns": _pending_upload["columns"],
            "suggested_mapping": _pending_upload["suggested_mapping"],
            "filename": file.filename,
            "total_rows": len(df),
            "preview": df.head(5).fillna("").astype(str).to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Could not parse file: {e}")


@router.post("/api/map-columns")
def map_columns(payload: dict, session: Session = Depends(get_session)):
    mapping = payload.get("mapping", {})
    df = _pending_upload.get("df")
    if df is None:
        raise HTTPException(400, "No file uploaded yet.")
    if "pool_id" not in mapping or "reading_date" not in mapping:
        raise HTTPException(400, "Pool ID and Date columns are required.")
    if not any(k in mapping for k in ("ph", "free_chlorine", "turbidity")):
        raise HTTPException(400, "At least one measurement column must be mapped.")

    rows, skipped = [], []
    for idx, raw in df.iterrows():
        entry = {}
        pid = str(raw.get(mapping["pool_id"], "")).strip()
        if not pid or pid == "nan":
            skipped.append({"row": int(idx) + 2, "reason": "Missing pool ID"})
            continue
        entry["pool_id"] = pid.lower()
        rd = _parse_date_flexible(raw.get(mapping["reading_date"]))
        if not rd:
            skipped.append({"row": int(idx) + 2, "reason": "Invalid date"})
            continue
        entry["reading_date"] = rd
        for f2 in ("ph", "free_chlorine", "turbidity"):
            entry[f2] = _safe_float(raw.get(mapping[f2])) if f2 in mapping else None
        if "pool_volume_m3" in mapping:
            entry["pool_volume_m3"] = _safe_float(raw.get(mapping["pool_volume_m3"]))
        if "community_name" in mapping:
            cn = str(raw.get(mapping["community_name"], "")).strip()
            entry["community_name"] = cn if cn != "nan" else ""
        if all(entry.get(k) is None for k in ("ph", "free_chlorine", "turbidity")):
            skipped.append({"row": int(idx) + 2, "reason": "No valid measurements"})
            continue
        rows.append(entry)

    if not rows:
        raise HTTPException(400, "No valid rows parsed.")
    n = repo.upsert_readings_batch(session, rows, source="upload")
    session.commit()
    repo.add_ingest_log(session, source="upload", filename=_pending_upload.get("filename", ""),
                        pool_count=len({r["pool_id"] for r in rows}),
                        row_count=len(rows), skipped_count=len(skipped))
    session.commit()
    _pending_upload.pop("df", None)
    return {"success": True, "loaded_rows": n, "skipped_count": len(skipped), "skipped": skipped[:50]}


# ---------------------------------------------------------------------------
# Manual reading
# ---------------------------------------------------------------------------

@manual_router.post("")
def add_manual_reading(payload: dict, session: Session = Depends(get_session)):
    pool_id = str(payload.get("pool_id", "")).strip()
    if not pool_id:
        raise HTTPException(400, "Pool ID is required.")
    rd = _parse_date_flexible(payload.get("reading_date", ""))
    if not rd:
        raise HTTPException(400, "Valid reading date is required.")
    ph = _safe_float(payload.get("ph"))
    cl = _safe_float(payload.get("free_chlorine"))
    turb = _safe_float(payload.get("turbidity"))
    vol = _safe_float(payload.get("pool_volume_m3"))
    errs = []
    if ph is not None and (ph < 0 or ph > 14):
        errs.append("pH must be 0–14.")
    if cl is not None and cl < 0:
        errs.append("Chlorine cannot be negative.")
    if turb is not None and turb < 0:
        errs.append("Turbidity cannot be negative.")
    if vol is not None and vol <= 0:
        errs.append("Volume must be positive.")
    if ph is None and cl is None and turb is None:
        errs.append("At least one measurement required.")
    if errs:
        raise HTTPException(400, " ".join(errs))

    entry = {
        "pool_id": pool_id.lower(),
        "reading_date": rd,
        "ph": ph, "free_chlorine": cl, "turbidity": turb,
        "community_name": str(payload.get("community_name", "")).strip(),
        "pool_volume_m3": vol,
    }
    n = repo.upsert_readings_batch(session, [entry], source="manual")
    session.commit()
    return {"success": True, "pool_id": pool_id.lower(), "rows": n}


import os  # noqa (used in upload_file, kept top-level)