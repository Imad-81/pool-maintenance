"""
Test admin API endpoints and token authorization.
"""

from fastapi.testclient import TestClient
from backend.main import app
from backend.settings import settings

client = TestClient(app)


def test_admin_weather_status():
    response = client.get("/api/admin/weather-status")
    assert response.status_code == 200
    data = response.json()
    assert "latest_weather_date" in data


def test_admin_runs_list():
    response = client.get("/api/admin/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_admin_ingest_log():
    response = client.get("/api/admin/ingest-log")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_admin_token_auth():
    settings.admin_token = "test-secret-123"
    try:
        # Unauthorized without header
        res_unauth = client.get("/api/admin/runs")
        assert res_unauth.status_code == 401

        # Forbidden with wrong token
        res_wrong = client.get("/api/admin/runs", headers={"Authorization": "Bearer wrong-token"})
        assert res_wrong.status_code == 403

        # Authorized with correct token
        res_ok = client.get("/api/admin/runs", headers={"Authorization": "Bearer test-secret-123"})
        assert res_ok.status_code == 200
    finally:
        settings.admin_token = None
