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

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_create_recording_success(client):
    res = client.post("/recordings/", json={"id": "test_rec"})
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "test_rec"
    assert data["status"] == "ingested"

def test_create_recording_malformed(client):
    # Missing required 'id' field
    res = client.post("/recordings/", json={"wrong_field": "test_rec"})
    assert res.status_code == 422 # Unprocessable Entity from Pydantic

def test_create_recording_duplicate(client):
    # Duplicate
    res = client.post("/recordings/", json={"id": "test_rec"})
    assert res.status_code == 400

def test_get_recording_not_found(client):
    res = client.get("/recordings/nonexistent")
    assert res.status_code == 404

def test_predict_recording_invalid_model(client):
    res = client.post("/recordings/test_rec/predict?model_version=invalid_model")
    assert res.status_code == 422
    assert "Invalid model version" in res.text
    
def test_predict_recording_determinism(client):
    # Attempting to predict on a recording 'test_rec'. 
    # Since we need a real recording for WFDB to load in run_predictions, 
    # we will mock wfdb.rdrecord if test_rec doesn't exist, OR 
    # we expect the 400 error about missing raw data.
    res1 = client.post("/recordings/test_rec/predict?model_version=cnn")
    if res1.status_code == 400 and "raw data" in res1.text.lower():
        pytest.skip("Raw data not available for integration test.")
        
    assert res1.status_code == 200
    preds1 = res1.json()
    
    res2 = client.post("/recordings/test_rec/predict?model_version=cnn")
    assert res2.status_code == 200
    preds2 = res2.json()
    
    assert preds1 == preds2, "Predictions should be deterministic!"

def test_severity_without_predictions(client):
    # create new recording without running predict
    client.post("/recordings/", json={"id": "test_sev_rec"})
    res = client.get("/recordings/test_sev_rec/severity")
    assert res.status_code == 400
    assert "No predictions found" in res.text

def test_explanation_generation_and_not_found(client):
    res = client.get("/predictions/9999/explanation")
    assert res.status_code == 404
    
    # Generate a real prediction first
    res_pred = client.post("/recordings/test_rec/predict?model_version=cnn")
    if res_pred.status_code == 400 and "raw data" in res_pred.text.lower():
        pytest.skip("Raw data not available for integration test.")
    assert res_pred.status_code == 200
    preds = res_pred.json()
    assert len(preds) > 0
    
    pred_id = preds[0]["id"]
    
    # Request explanation
    res_exp = client.get(f"/predictions/{pred_id}/explanation")
    assert res_exp.status_code == 200
    
    exp = res_exp.json()
    plot_path = exp["plot_path"]
    
    assert os.path.exists(plot_path), f"Plot file was not created: {plot_path}"
    
    # Ensure it's not an empty or tiny text file (mock image data is 15 bytes)
    # A real matplotlib png should be at least 5-10KB
    file_size = os.path.getsize(plot_path)
    assert file_size > 1000, f"Plot file is suspiciously small ({file_size} bytes). Not a real PNG."
