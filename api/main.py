"""
FastAPI app per Amazon Species Risk Predictor.
"""

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    InfoResponse,
    PredictionResponse,
    SpeciesFeatures,
)

# ===== INIT =====
app = FastAPI(
    title="Amazon Rainforest Species Risk Predictor",
    description="ML API per predire rischio estinzione specie amazzoniche",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== GLOBAL STATE =====
MODEL = None
PREPROCESSOR = None
METADATA = None


def load_models():
    """Carica modelli e preprocessor al startup."""
    global MODEL, PREPROCESSOR, METADATA

    try:
        model_path = Path("models/xgboost_v1.pkl")
        preprocessor_path = Path("models/preprocessor.pkl")
        metadata_path = Path("models/metadata.json")

        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {model_path}")

        MODEL = joblib.load(model_path)
        PREPROCESSOR = joblib.load(preprocessor_path)

        with open(metadata_path, "r") as f:
            METADATA = json.load(f)

        print(" Modelli caricati con successo")
    except Exception as e:
        print(f" Errore caricamento modelli: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Eseguito al startup dell'app."""
    load_models()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if MODEL is not None else "error",
        model_loaded=MODEL is not None,
        version="1.0.0",
    )


@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Restituisci info sui modelli."""
    if METADATA is None:
        raise HTTPException(status_code=503, detail="Modelli non caricati")

    return InfoResponse(
        n_features=METADATA["n_features"],
        feature_names=METADATA["feature_names"],
        models_available=["xgboost", "random_forest", "logistic_regression"],
        iucn_categories=METADATA["iucn_categories"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(features: SpeciesFeatures):
    """Predici rischio estinzione per una specie."""
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Modelli non caricati")

    try:
        # Crea DataFrame con una riga
        import pandas as pd

        data = {
            "population_size": [features.population_size],
            "habitat_fragmentation": [features.habitat_fragmentation],
            "climate_vulnerability": [features.climate_vulnerability],
            "illegal_hunting_pressure": [features.illegal_hunting_pressure],
            "conservation_efforts_index": [features.conservation_efforts_index],
            "habitat": [features.habitat],
            "breeding_program_exists": [features.breeding_program_exists],
            "legal_protection": [features.legal_protection],
        }
        df = pd.DataFrame(data)

        # Preprocessing
        X = PREPROCESSOR.transform(df)

        # Predici
        prediction = MODEL.predict(X)[0]
        probabilities = MODEL.predict_proba(X)[0]

        # Map codice a categoria
        category_map = {
            0: "Least Concern",
            1: "Vulnerable",
            2: "Endangered",
            3: "Critically Endangered",
        }

        risk_category = category_map.get(int(prediction), "Unknown")
        confidence = float(np.max(probabilities))

        return PredictionResponse(
            risk_category=risk_category,
            risk_code=int(prediction),
            confidence=confidence,
            probabilities={
                f"class_{i}": float(prob) for i, prob in enumerate(probabilities)
            },
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore predizione: {str(e)}")


@app.post("/bulk-predict")
async def bulk_predict(features_list: List[SpeciesFeatures]):
    """Predici rischio per multiple specie."""
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Modelli non caricati")

    try:
        import pandas as pd

        # Crea DataFrame
        data = {
            "population_size": [f.population_size for f in features_list],
            "habitat_fragmentation": [f.habitat_fragmentation for f in features_list],
            "climate_vulnerability": [f.climate_vulnerability for f in features_list],
            "illegal_hunting_pressure": [f.illegal_hunting_pressure for f in features_list],
            "conservation_efforts_index": [f.conservation_efforts_index for f in features_list],
            "habitat": [f.habitat for f in features_list],
            "breeding_program_exists": [f.breeding_program_exists for f in features_list],
            "legal_protection": [f.legal_protection for f in features_list],
        }
        df = pd.DataFrame(data)

        X = PREPROCESSOR.transform(df)
        predictions = MODEL.predict(X)
        probabilities = MODEL.predict_proba(X)

        category_map = {
            0: "Least Concern",
            1: "Vulnerable",
            2: "Endangered",
            3: "Critically Endangered",
        }

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "index": i,
                "risk_category": category_map.get(int(pred), "Unknown"),
                "risk_code": int(pred),
                "confidence": float(np.max(probabilities[i])),
            })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore predizione batch: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
