#  Amazon Rainforest Species Risk Predictor

ML/MLOps project for predicting extinction risk of Amazon rainforest species using synthetic data, FastAPI, Docker, Kubernetes, and Terraform.

[![GitHub](https://img.shields.io/badge/GitHub-dariolignana96-blue)](https://github.com/dariolignana96/amazon-rainforest-species-risk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  Project Overview

This project demonstrates a **complete ML/MLOps workflow** for species conservation:

- ** Synthetic Data Generation**: 1000 realistic Amazon species with 10+ ecological features
- ** Ensemble ML Models**: XGBoost, Random Forest, Logistic Regression
- ** FastAPI REST API**: Production-ready endpoints for risk prediction
- ** Docker & Orchestration**: Containerized deployment ready for Kubernetes
- ** Infrastructure as Code**: Terraform for AWS deployment (design only)
- ** Full Documentation**: Architecture, API, deployment guides

---

##  Features

### Data Generation
- 100% **original synthetic dataset** (no scraping, no copyright issues)
- 1000 species with realistic ecological features
- IUCN risk categories: Least Concern  Critically Endangered
- Deterministic generation (reproducible with seed=42)

### ML Pipeline
- **Preprocessing**: StandardScaler, OneHotEncoder, train/test split (80/20)
- **Models**:
  - XGBoost (main predictor, 4-class classification)
  - Random Forest (ensemble diversity)
  - Logistic Regression (baseline, interpretability)
- **Metrics**: Accuracy, F1-score (weighted), feature importance analysis
- **Model Registry**: Joblib serialization for inference

### FastAPI Endpoints
- \GET /health\ - Health check
- \GET /info\ - Model info and available features
- \POST /predict\ - Single species risk prediction
- \POST /bulk-predict\ - Batch prediction for multiple species
- Auto-generated Swagger UI at \/docs\

---

##  Quick Start

### Prerequisites
- Python 3.11+
- Git

### Installation & Training

\\\ash
# Clone repository
git clone https://github.com/dariolignana96/amazon-rainforest-species-risk.git
cd amazon-rainforest-species-risk

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset
python data/synthetic_generator.py

# Train ML models
python ml/train.py
\\\

### Run API Locally

\\\ash
# Start API server
uvicorn api.main:app --reload

# Access Swagger UI at http://127.0.0.1:8000/docs
\\\

---

##  API Usage Examples

### Health Check
\\\ash
curl http://localhost:8000/health
\\\

### Single Prediction
\\\ash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "population_size": 500,
    "habitat_fragmentation": 0.7,
    "climate_vulnerability": 0.5,
    "illegal_hunting_pressure": 0.3,
    "conservation_efforts_index": 0.2,
    "habitat": "Canopy",
    "breeding_program_exists": 0,
    "legal_protection": 1
  }'
\\\

### Bulk Prediction
\\\ash
curl -X POST http://localhost:8000/bulk-predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "population_size": 500,
      "habitat_fragmentation": 0.7,
      "climate_vulnerability": 0.5,
      "illegal_hunting_pressure": 0.3,
      "conservation_efforts_index": 0.2,
      "habitat": "Canopy",
      "breeding_program_exists": 0,
      "legal_protection": 1
    }
  ]'
\\\

---

##  Project Structure

\\\
amazon-rainforest-species-risk/
 data/
    synthetic_generator.py      # Data generation (100% original)
    raw/
       amazon_species.csv      # Generated dataset (1000 species)
    processed/

 ml/
    preprocessing.py            # Data preprocessing pipeline
    models.py                   # Model definitions
    train.py                    # Training script

 api/
    main.py                     # FastAPI app
    schemas.py                  # Pydantic models
    routers/

 models/
    xgboost_v1.pkl              # Trained XGBoost model
    preprocessor.pkl            # Sklearn preprocessor
    metadata.json               # Model metadata

 k8s/                            # Kubernetes manifests
 infra/                          # Terraform IaC
 tests/                          # Pytest unit tests
 notebooks/                      # Jupyter notebooks

 Dockerfile                      # Multi-stage Docker build
 docker-compose.yml              # Local dev setup
 requirements.txt                # Python dependencies
 .gitignore                      # Git exclusions
 LICENSE                         # MIT License
 README.md                       # This file
\\\

---

##  ML Models

### XGBoost (Primary)
- **Architecture**: Gradient boosting classifier, 100 estimators
- **Hyperparameters**: max_depth=6, learning_rate=0.1
- **Input**: 9 features (5 numeric + 4 categorical encoded)
- **Output**: 4-class IUCN category
- **Performance**: ~85% accuracy on test set

### Random Forest (Ensemble)
- **Architecture**: 100 estimators, max_depth=10
- **Performance**: ~82% accuracy on test set

### Logistic Regression (Baseline)
- **Performance**: ~75% accuracy on test set

---

##  Docker Deployment

### Build Image
\\\ash
docker build -t rainforest-api:latest .
\\\

### Run with Docker Compose
\\\ash
docker-compose up
\\\

---

##  Dataset Details

### Amazon Species Registry (Synthetic)
- **Size**: 1000 records
- **Features**: 10 ecological variables + 1 target
- **Target**: IUCN category (4 classes)
- **Generation**: 100% original, deterministic, reproducible

### Features
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| population_size | numeric | 10-200,000 | Estimated population |
| habitat_fragmentation | numeric | 0-1 | Habitat fragmentation index |
| climate_vulnerability | numeric | 0-1 | Climate change vulnerability |
| illegal_hunting_pressure | numeric | 0-1 | Hunting/poaching pressure |
| conservation_efforts_index | numeric | 0-1 | Conservation effort index |
| habitat | categorical | 3 values | Primary habitat type |
| breeding_program_exists | binary | 0/1 | Active breeding program |
| legal_protection | binary | 0/1 | Legal protection status |

---

##  Roadmap

### Phase 1  (Current)
- Synthetic data generator
- ML training pipeline (XGBoost, RF, LR)
- FastAPI REST API
- Docker containerization

### Phase 2 (Next)
- Frontend (HTML/CSS/JS)
- Batch prediction UI
- Model monitoring dashboard

### Phase 3
- Image classification (CNN)
- Temporal forecasting
- Advanced monitoring

---

##  License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**All code is 100% original, open-source, and free from copyright restrictions.**

---

##  Author

**Dario Lignana**
- GitHub: [@dariolignana96](https://github.com/dariolignana96)
- Email: lignana.dario@gmail.com

---
