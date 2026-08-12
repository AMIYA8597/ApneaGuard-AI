import os
import pytest
from fastapi.testclient import TestClient

# Ensure testing environment variable is set BEFORE importing main
os.environ["TESTING"] = "1"

from api.main import app
from api.db import Base, engine, get_db
from api import models

# Recreate tables just in case
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_create_recording_success():
    res = client.post("/recordings/", json={"id": "test_rec"})
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "test_rec"
    assert data["status"] == "ingested"
def test_create_recording_malformed():
    # Missing required 'id' field
    res = client.post("/recordings/", json={"wrong_field": "test_rec"})
    assert res.status_code == 422 # Unprocessable Entity from Pydantic

def test_create_recording_success(client, db_session):
    payload = {"id": "test_01"}
    response = client.post("/recordings/", json=payload)
    assert response.status_code == 200

def test_create_recording_duplicate():
    # Duplicate
    res = client.post("/recordings/", json={"id": "test_rec"})
    assert res.status_code == 400

def test_get_recording_not_found():
    res = client.get("/recordings/nonexistent")
    assert res.status_code == 404

def test_predict_recording_invalid_model():
    res = client.post("/recordings/test_rec/predict?model_version=invalid_model")
    assert res.status_code == 422
    assert "Invalid model version" in res.text
    
def test_predict_recording_missing_features():
    # Attempting to predict on a recording that doesn't have features in the parquet table 
    # (test_rec is not in our real parquet file)
    # The ml_pipeline raises ValueError which our router catches and returns 400
    res = client.post("/recordings/test_rec/predict?model_version=cnn")
    assert res.status_code == 400
    assert "not found" in res.text.lower()

def test_severity_without_predictions():
    # create new recording without running predict
    client.post("/recordings/", json={"id": "test_sev_rec"})
    res = client.get("/recordings/test_sev_rec/severity")
    assert res.status_code == 400
    assert "No predictions found" in res.text

def test_explanation_not_found():
    res = client.get("/predictions/9999/explanation")
    assert res.status_code == 404
