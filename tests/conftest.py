"""
Pytest configuration and FastAPI test client fixtures with mock Prisma layer for hermetic unit testing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient

from backend.main import app
from backend.store.client import get_db
from prisma.models import Pool, Reading, WeatherDaily, ModelRun, IngestLog, DailyPrediction


class MockPrisma:
    def __init__(self):
        sample_pool = Pool.model_construct(
            pool_id="Cabo Verde (19)",
            community_name="Cabo Verde",
            pool_volume_m3=150.0,
            pool_surface_m2=100.0,
            filter_diameter=600.0,
            filter_count=1.0,
            motor_count=1.0,
            pool_heated=0,
            pool_community=1,
            pool_outdoor=1,
            pool_private=0,
            pool_public=0,
            pool_skimmer=1,
            pool_overflow=0,
            pool_oval=0,
            pool_round=0,
            pool_rectangular_0714=1,
            pool_rectangular_07=0,
            vegetation_contamination=0,
            deck_grass=0.0,
            deck_mixed=0.0,
            deck_paved=1.0,
        )

        sample_reading = Reading.model_construct(
            id=1,
            pool_id="Cabo Verde (19)",
            reading_date=datetime(2026, 8, 1, 10, 0),
            ph=7.4,
            free_chlorine=1.5,
            turbidity=0.5,
            hypochlorite_dosing_pct=50.0,
            hypochlorite_dosing_hours=4.0,
            water_temperature=26.0,
            source="master",
            created_at=datetime(2026, 8, 1, 10, 0),
        )

        sample_weather = WeatherDaily.model_construct(
            date=datetime(2026, 8, 1),
            w_temp_max=32.0,
            w_temp_mean=26.0,
            w_uv_max=8.0,
            w_uv_clear_sky_max=8.5,
            w_solar_radiation=25.0,
            w_sunshine_hours=12.0,
            w_precipitation_mm=0.0,
            w_wind_max_kmh=15.0,
            w_et0=5.0,
            w_weather_code=0,
            fetched_at=datetime(2026, 8, 1),
        )

        sample_modelrun = ModelRun.model_construct(
            run_id="v6-setpoint-v2",
            artifact_dir="models/v6-setpoint-v2",
            created_at=datetime(2026, 8, 1),
            is_active=1,
            metrics_json='{"chlorine_next": {"mae": 0.197}, "ph_next": {"mae": 0.033}, "turbidity_next": {"mae": 0.042}}',
            promoted_at=datetime(2026, 8, 1),
            promote_reason="Baseline active model",
        )

        sample_ingestlog = IngestLog.model_construct(
            id=1,
            source="upload",
            filename="test.csv",
            pool_count=5,
            row_count=20,
            skipped_count=0,
            created_at=datetime(2026, 8, 1),
            detail_json=None,
        )

        sample_prediction = DailyPrediction.model_construct(
            id=1,
            pool_id="Cabo Verde (19)",
            as_of_date=datetime(2026, 8, 1),
            last_reading_date=datetime(2026, 8, 1),
            ph=7.4,
            free_chlorine=1.5,
            turbidity=0.5,
            urgency="Routine",
            urgency_order=5,
            breach_proba=0.0,
            predicted_cl_today=1.45,
            predicted_ph_today=7.42,
            predicted_turb_today=0.52,
            predicted_cl_tmrw=1.35,
            predicted_ph_tmrw=7.45,
            predicted_turb_tmrw=0.55,
            today_forecast_json='{"predicted_cl": 1.45, "predicted_ph": 7.42, "predicted_turb": 0.52, "urgency": "Routine", "status": "OK"}',
            tomorrow_forecast_json='{"predicted_cl": 1.35, "predicted_ph": 7.45, "predicted_turb": 0.55, "urgency": "Routine", "status": "OK"}',
            pool=sample_pool,
        )

        # Pool mock
        self.pool = MagicMock()
        self.pool.find_many = AsyncMock(return_value=[sample_pool])
        self.pool.find_unique = AsyncMock(return_value=sample_pool)
        self.pool.upsert = AsyncMock(return_value=sample_pool)
        self.pool.count = AsyncMock(return_value=1)

        # Reading mock
        self.reading = MagicMock()
        self.reading.find_many = AsyncMock(return_value=[sample_reading])
        self.reading.find_first = AsyncMock(return_value=sample_reading)
        self.reading.upsert = AsyncMock(return_value=sample_reading)
        self.reading.count = AsyncMock(return_value=10)

        # Weather mock
        self.weatherdaily = MagicMock()
        self.weatherdaily.find_many = AsyncMock(return_value=[sample_weather])
        self.weatherdaily.find_first = AsyncMock(return_value=sample_weather)
        self.weatherdaily.find_unique = AsyncMock(return_value=sample_weather)
        self.weatherdaily.upsert = AsyncMock(return_value=sample_weather)

        # Model run mock
        self.modelrun = MagicMock()
        self.modelrun.find_many = AsyncMock(return_value=[sample_modelrun])
        self.modelrun.find_first = AsyncMock(return_value=sample_modelrun)
        self.modelrun.create = AsyncMock(return_value=sample_modelrun)
        self.modelrun.update = AsyncMock(return_value=sample_modelrun)
        self.modelrun.update_many = AsyncMock(return_value=MagicMock())

        # Ingest log mock
        self.ingestlog = MagicMock()
        self.ingestlog.find_many = AsyncMock(return_value=[sample_ingestlog])
        self.ingestlog.create = AsyncMock(return_value=sample_ingestlog)

        # DailyPrediction mock
        self.dailyprediction = MagicMock()
        self.dailyprediction.find_many = AsyncMock(return_value=[sample_prediction])
        self.dailyprediction.find_unique = AsyncMock(return_value=sample_prediction)
        self.dailyprediction.upsert = AsyncMock(return_value=sample_prediction)
        self.dailyprediction.count = AsyncMock(return_value=1)
        self.dailyprediction.delete_many = AsyncMock(return_value=MagicMock())

        # Transactions
        self.tx = MagicMock()
        self.tx.return_value.__aenter__ = AsyncMock(return_value=self)
        self.tx.return_value.__aexit__ = AsyncMock(return_value=None)

        self.query_raw = AsyncMock(return_value=[{"day": datetime(2026, 8, 1), "count": 5}])



@pytest.fixture(autouse=True)
def override_database_dependency():
    """Automatically mock database client for FastAPI tests."""
    mock_db = MockPrisma()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.clear()
