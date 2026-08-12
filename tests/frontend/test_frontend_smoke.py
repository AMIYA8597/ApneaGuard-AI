import os
import pytest
from fastapi.testclient import TestClient

# Ensure testing environment variable is set
os.environ["TESTING"] = "1"

from api.main import app
from api.db import Base, engine
from api.models import Recording, ModelVersion, Prediction, SeverityScore

# Ensure tables are created for tests
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_frontend_smoke():
    """
    Smoke test to confirm the frontend dashboard is properly mounted and serving 
    the essential UI components without crashing, including the mandated compliance banner.
    """
    # 1. Test that the frontend is mounted and serves index.html
    res = client.get("/")
    assert res.status_code == 200
    assert "ApneaGuard AI Dashboard" in res.text
    assert "recording-select" in res.text
    
    # 2. Test compliance banner exists
    assert "Not a diagnostic device" in res.text
    assert "PhysioNet" in res.text
    
    # 3. Test that the API endpoints used by the dashboard are accessible
    res_recordings = client.get("/recordings/")
    assert res_recordings.status_code == 200
    assert isinstance(res_recordings.json(), list)
