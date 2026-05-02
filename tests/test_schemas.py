"""
Test validazione Pydantic schemas.
"""

import pytest
from pydantic import ValidationError
from api.schemas import SpeciesFeatures, HealthResponse


class TestSpeciesFeatures:

    def test_valid_payload(self):
        s = SpeciesFeatures(
            population_size=173,
            habitat_fragmentation=0.85,
            climate_vulnerability=0.72,
            illegal_hunting_pressure=0.78,
            conservation_efforts_index=0.45,
            habitat="Canopy",
            breeding_program_exists=1,
            legal_protection=1
        )
        assert s.population_size == 173

    def test_population_must_be_positive(self):
        with pytest.raises(ValidationError):
            SpeciesFeatures(
                population_size=-1,
                habitat_fragmentation=0.5,
                climate_vulnerability=0.5,
                illegal_hunting_pressure=0.5,
                conservation_efforts_index=0.5,
                habitat="Canopy",
                breeding_program_exists=0,
                legal_protection=0
            )

    def test_fragmentation_above_1_invalid(self):
        with pytest.raises(ValidationError):
            SpeciesFeatures(
                population_size=100,
                habitat_fragmentation=1.5,
                climate_vulnerability=0.5,
                illegal_hunting_pressure=0.5,
                conservation_efforts_index=0.5,
                habitat="Canopy",
                breeding_program_exists=0,
                legal_protection=0
            )

    def test_breeding_program_must_be_binary(self):
        with pytest.raises(ValidationError):
            SpeciesFeatures(
                population_size=100,
                habitat_fragmentation=0.5,
                climate_vulnerability=0.5,
                illegal_hunting_pressure=0.5,
                conservation_efforts_index=0.5,
                habitat="Canopy",
                breeding_program_exists=5,
                legal_protection=0
            )


class TestHealthResponse:

    def test_valid_health(self):
        h = HealthResponse(status="ok", model_loaded=True, version="1.0.0")
        assert h.status == "ok"
        assert h.model_loaded is True
