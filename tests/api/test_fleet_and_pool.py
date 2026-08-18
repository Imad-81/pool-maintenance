"""
Test fleet overview and pool detail API endpoints.
"""

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from backend.main import app

client = TestClient(app)


def test_fleet_endpoint_success():
    response = client.get("/api/fleet")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)


def test_fleet_summary_endpoint():
    response = client.get("/api/fleet/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "counts" in data
    assert "compliance_rate" in data
    assert "as_of_date" in data


def test_trigger_fleet_inference():
    response = client.post("/api/fleet/run-inference")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predictions_generated" in data
    assert "as_of_date" in data




def test_fleet_query_params_validation():
    # Invalid date format
    response = client.get("/api/fleet?date=not-a-date")
    assert response.status_code == 400

    # Invalid pagination
    response = client.get("/api/fleet?page=-1")
    assert response.status_code == 422


def test_fleet_pool_ids():
    response = client.get("/api/fleet/pool-ids")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_fleet_dates():
    response = client.get("/api/fleet/dates")
    assert response.status_code == 200
    data = response.json()
    assert "min" in data
    assert "max" in data


def test_pool_detail_success():
    response = client.get("/api/pool/Cabo%20Verde%20(19)")
    assert response.status_code == 200
    data = response.json()
    assert data["pool_id"] == "Cabo Verde (19)"
    assert "forecast" in data
    assert "history" in data
    assert "recommended_visit" in data
    rec = data["recommended_visit"]
    assert rec is not None
    assert "date" in rec
    assert "day_offset_from_today" in rec
    assert "predicted_cl" in rec
    assert "predicted_ph" in rec
    assert "predicted_turb" in rec
    assert "urgency" in rec
    assert "reason" in rec


def test_pool_detail_not_found(override_database_dependency):
    override_database_dependency.reading.find_first = AsyncMock(return_value=None)
    response = client.get("/api/pool/non_existent_pool_xyz_123")
    assert response.status_code == 404


def test_optimise_success():
    response = client.get("/api/optimise/Cabo%20Verde%20(19)")
    assert response.status_code == 200
    data = response.json()
    assert "recommended_dosing" in data
    assert "feasible_configurations" in data


def test_optimise_not_found(override_database_dependency):
    override_database_dependency.reading.find_first = AsyncMock(return_value=None)
    response = client.get("/api/optimise/non_existent_pool_xyz_123")
    assert response.status_code == 404
