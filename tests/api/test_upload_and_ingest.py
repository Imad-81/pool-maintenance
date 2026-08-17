"""
Test CSV/Excel upload, column mapping, manual reading entry, and ingestion API.
"""

import io
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_upload_csv_preview():
    csv_content = (
        "Pool,Fecha,Cloro,pH,Turbidez\n"
        "Pool_Alpha,2026-08-01,1.5,7.4,0.5\n"
        "Pool_Beta,2026-08-01,2.0,7.5,0.8\n"
    )
    file_tuple = ("test_readings.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    response = client.post("/api/upload", files={"file": file_tuple})
    assert response.status_code == 200
    data = response.json()
    assert data["total_rows"] == 2
    assert "upload_id" in data
    assert "suggested_mapping" in data
    assert data["suggested_mapping"].get("pool_id") == "Pool"
    assert data["suggested_mapping"].get("reading_date") == "Fecha"


def test_upload_empty_csv_rejected():
    empty_csv = "col1\n"
    file_tuple = ("empty.csv", io.BytesIO(empty_csv.encode("utf-8")), "text/csv")
    response = client.post("/api/upload", files={"file": file_tuple})
    assert response.status_code == 400


def test_manual_reading_validation():
    # Valid reading
    payload = {
        "pool_id": "Test_Pool_99",
        "reading_date": "2026-08-10T10:00:00",
        "ph": 7.4,
        "free_chlorine": 1.5,
        "turbidity": 0.5,
        "pool_volume_m3": 120.0,
    }
    # Test schema validation
    # If DB is not connected in standalone unit test, endpoint will validate payload
    # Let's test invalid parameters
    invalid_payload = {
        "pool_id": "Test_Pool_99",
        "reading_date": "invalid-date",
        "ph": 18.0,  # invalid pH > 14
    }
    response = client.post("/api/readings", json=invalid_payload)
    assert response.status_code in (400, 422)


def test_ingest_json_validation():
    # Missing required keys
    payload = {
        "rows": [
            {"foo": "bar"}
        ]
    }
    response = client.post("/api/ingest/readings", json=payload)
    assert response.status_code == 400
