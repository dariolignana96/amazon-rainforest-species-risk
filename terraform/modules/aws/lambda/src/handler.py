"""
Lambda Handler - Rainforest Species Risk Prediction
====================================================
Questo file è il punto di ingresso della Lambda.
Carica il modello ML da S3 (cached dopo il primo invocation warm) 
e restituisce la predizione del rischio di estinzione.

Cold start vs Warm start:
  - Cold start: Lambda scarica il modello da S3 (~2-3 secondi extra)
  - Warm start: il modello è già in memoria (/tmp), risposta <100ms
"""

import json
import os
import io
import logging
import boto3
import joblib
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Variabili globali: persistono tra invocazioni warm (ottimizzazione)
model = None
preprocessor = None
s3_client = boto3.client("s3")

RISK_CATEGORIES = {
    0: "Least Concern",
    1: "Vulnerable",
    2: "Endangered",
    3: "Critically Endangered"
}


def load_model_from_s3():
    """Carica il modello ML da S3 e lo mette in cache su /tmp."""
    global model, preprocessor

    bucket = os.environ["S3_BUCKET"]
    model_key = os.environ["MODEL_KEY"]
    preprocessor_key = os.environ["PREPROCESSOR_KEY"]

    logger.info(f"Loading model from s3://{bucket}/{model_key}")

    # Scarica modello
    model_obj = s3_client.get_object(Bucket=bucket, Key=model_key)
    model = joblib.load(io.BytesIO(model_obj["Body"].read()))

    # Scarica preprocessor
    prep_obj = s3_client.get_object(Bucket=bucket, Key=preprocessor_key)
    preprocessor = joblib.load(io.BytesIO(prep_obj["Body"].read()))

    logger.info("Model loaded successfully")


def lambda_handler(event, context):
    """
    Entry point Lambda.
    
    Input (API Gateway event body):
    {
        "population_size": 173,
        "habitat_fragmentation": 0.85,
        "climate_vulnerability": 0.72,
        "illegal_hunting_pressure": 0.78,
        "conservation_efforts_index": 0.45,
        "habitat": "Canopy",
        "breeding_program_exists": 1,
        "legal_protection": 1
    }
    """
    global model, preprocessor

    # Carica il modello solo la prima volta (warm start optimization)
    if model is None:
        load_model_from_s3()

    try:
        body = json.loads(event.get("body", "{}"))

        # Prepara features nell'ordine atteso dal preprocessor
        features = {
            "population_size":            body["population_size"],
            "habitat_fragmentation":      body["habitat_fragmentation"],
            "climate_vulnerability":      body["climate_vulnerability"],
            "illegal_hunting_pressure":   body["illegal_hunting_pressure"],
            "conservation_efforts_index": body["conservation_efforts_index"],
            "habitat":                    body["habitat"],
            "breeding_program_exists":    body["breeding_program_exists"],
            "legal_protection":           body["legal_protection"],
        }

        # Preprocessing + Predict
        X = preprocessor.transform([list(features.values())])
        probabilities = model.predict_proba(X)[0]
        risk_code = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        response_body = {
            "risk_category": RISK_CATEGORIES[risk_code],
            "risk_code":     risk_code,
            "confidence":    round(confidence, 4),
            "probabilities": {
                f"class_{i}": round(float(p), 4)
                for i, p in enumerate(probabilities)
            }
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(response_body)
        }

    except KeyError as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Campo mancante: {str(e)}"})
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Errore interno"})
        }
