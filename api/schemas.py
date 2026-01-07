"""
Pydantic schemas per validazione request/response API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SpeciesFeatures(BaseModel):
    """Features di una specie per predizione rischio."""
    
    population_size: float = Field(..., gt=0, description="Popolazione stimata")
    habitat_fragmentation: float = Field(..., ge=0, le=1, description="Frammentazione habitat (0-1)")
    climate_vulnerability: float = Field(..., ge=0, le=1, description="Vulnerabilita climatica (0-1)")
    illegal_hunting_pressure: float = Field(..., ge=0, le=1, description="Pressione caccia illegale (0-1)")
    conservation_efforts_index: float = Field(..., ge=0, le=1, description="Indice sforzi conservazione (0-1)")
    habitat: str = Field(..., description="Habitat (Canopy, Floor, River)")
    breeding_program_exists: int = Field(..., ge=0, le=1, description="Programma riproduzione (0/1)")
    legal_protection: int = Field(..., ge=0, le=1, description="Protezione legale (0/1)")


class PredictionResponse(BaseModel):
    """Risposta predizione rischio."""
    
    risk_category: str = Field(..., description="Categoria IUCN: Least Concern, Vulnerable, Endangered, Critically Endangered")
    risk_code: int = Field(..., description="Codice rischio (0-3)")
    confidence: float = Field(..., ge=0, le=1, description="Confidenza predizione (0-1)")
    probabilities: dict = Field(..., description="Probabilita per ogni classe")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    version: str


class InfoResponse(BaseModel):
    """Info modelli."""
    
    n_features: int
    feature_names: List[str]
    models_available: List[str]
    iucn_categories: List[str]
