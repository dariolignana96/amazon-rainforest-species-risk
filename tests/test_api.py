"""
Test suite per Amazon Rainforest Species Risk API.
Tutti i test girano in locale - nessun servizio cloud, nessun costo.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


# Fixtures

@pytest.fixture
def mock_models():
    """Mocka modello e preprocessor per evitare dipendenza dai .pkl."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([3])
    mock_model.predict_proba.return_value = np.array([[0.02, 0.05, 0.08, 0.85]])

    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 9))

    mock_metadata = {
        "n_features": 9,
        "feature_names": [
            "population_size", "habitat_fragmentation", "climate_vulnerability",
            "illegal_hunting_pressure", "conservation_efforts_index",
            "habitat_Canopy", "habitat_Floor", "breeding_program_exists", "legal_protection"
        ],
        "iucn_categories": ["Least Concern", "Vulnerable", "Endangered", "Critically Endangered"]
    }
    return mock_model, mock_preprocessor, mock_metadata


@pytest.fixture
def client(mock_models):
    """Client FastAPI con modelli mockati."""
    mock_model, mock_preprocessor, mock_metadata = mock_models

    with patch("api.main.MODEL", mock_model), \
         patch("api.main.PREPROCESSOR", mock_preprocessor), \
         patch("api.main.METADATA", mock_metadata), \
         patch("api.main.load_models"):
        from api.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def jaguar_payload():
    """Payload di test: Giaguaro amazzonico (alto rischio)."""
    return {
        "population_size": 173,
        "habitat_fragmentation": 0.85,
        "climate_vulnerability": 0.72,
        "illegal_hunting_pressure": 0.78,
        "conservation_efforts_index": 0.45,
        "habitat": "Canopy",
        "breeding_program_exists": 1,
        "legal_protection": 1
    }


@pytest.fixture
def capybara_payload():
    """Payload di test: Capibara (basso rischio)."""
    return {
        "population_size": 15000,
        "habitat_fragmentation": 0.55,
        "climate_vulnerability": 0.30,
        "illegal_hunting_pressure": 0.20,
        "conservation_efforts_index": 0.70,
        "habitat": "Floor",
        "breeding_program_exists": 0,
        "legal_protection": 1
    }


# Test /health

class TestHealth:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_version_present(self, client):
        data = client.get("/health").json()
        assert "version" in data


# Test /info

class TestInfo:

    def test_info_returns_200(self, client):
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_has_feature_names(self, client):
        data = client.get("/info").json()
        assert "feature_names" in data
        assert len(data["feature_names"]) > 0

    def test_info_has_iucn_categories(self, client):
        data = client.get("/info").json()
        assert "iucn_categories" in data
        assert len(data["iucn_categories"]) == 4

    def test_info_has_n_features(self, client):
        data = client.get("/info").json()
        assert data["n_features"] == 9


# Test /predict

class TestPredict:

    def test_predict_returns_200(self, client, jaguar_payload):
        response = client.post("/predict", json=jaguar_payload)
        assert response.status_code == 200

    def test_predict_response_structure(self, client, jaguar_payload):
        data = client.post("/predict", json=jaguar_payload).json()
        assert "risk_category" in data
        assert "risk_code" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_risk_code_range(self, client, jaguar_payload):
        data = client.post("/predict", json=jaguar_payload).json()
        assert 0 <= data["risk_code"] <= 3

    def test_predict_confidence_range(self, client, jaguar_payload):
        data = client.post("/predict", json=jaguar_payload).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_risk_category_valid(self, client, jaguar_payload):
        data = client.post("/predict", json=jaguar_payload).json()
        valid = ["Least Concern", "Vulnerable", "Endangered", "Critically Endangered"]
        assert data["risk_category"] in valid

    def test_predict_probabilities_sum_to_one(self, client, jaguar_payload):
        data = client.post("/predict", json=jaguar_payload).json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_missing_field_returns_422(self, client):
        response = client.post("/predict", json={"population_size": 173})
        assert response.status_code == 422

    def test_predict_invalid_habitat_fragmentation(self, client, jaguar_payload):
        jaguar_payload["habitat_fragmentation"] = 99.0
        response = client.post("/predict", json=jaguar_payload)
        assert response.status_code == 422

    def test_predict_negative_population_returns_422(self, client, jaguar_payload):
        jaguar_payload["population_size"] = -1
        response = client.post("/predict", json=jaguar_payload)
        assert response.status_code == 422


# Test /bulk-predict

class TestBulkPredict:

    def test_bulk_returns_200(self, client, jaguar_payload, capybara_payload):
        response = client.post("/bulk-predict", json=[jaguar_payload, capybara_payload])
        assert response.status_code == 200

    def test_bulk_has_index_field(self, client, jaguar_payload):
        data = client.post("/bulk-predict", json=[jaguar_payload]).json()
        assert "index" in data["predictions"][0]
