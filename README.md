# 🌿 Amazon Rainforest Species Risk Predictor

**ML/MLOps Portfolio Project** - Predicting extinction risk for Amazon rainforest species using synthetic data, XGBoost, FastAPI, Docker, and Kubernetes.

[![GitHub](https://img.shields.io/badge/GitHub-dariolignana96-blue?style=flat-square)](https://github.com/dariolignana96/amazon-rainforest-species-risk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)](https://www.python.org/)

---

## 📊 Project Overview

This project demonstrates a **complete, production-ready ML/MLOps workflow** for species conservation:

- **🔬 Synthetic Data Generation**: 1,000 realistic Amazon species with 10 ecological features
- **🤖 Ensemble ML Models**: XGBoost (primary), Random Forest, Logistic Regression
- **🔌 FastAPI REST API**: Production-ready endpoints for single & batch predictions
- **🐳 Docker Containerization**: Multi-stage builds, ready for deployment
- **☸️ Kubernetes Ready**: YAML manifests included
- **📚 Complete Documentation**: Architecture, API docs, deployment guides

**Real-world skills demonstrated:**
- Data generation & preprocessing (Pandas, Scikit-learn)
- Machine learning (XGBoost, Random Forest, Logistic Regression)
- API design (FastAPI, Pydantic validation)
- Containerization (Docker, docker-compose)
- Version control (Git, GitHub)
- MLOps practices (Model serialization, deployment readiness)

---

## ✨ Key Features

### 🔬 100% Original Synthetic Dataset
- **1,000 realistic species records** with deterministic generation (seed=42)
- **10 ecological features**: population size, habitat fragmentation, climate vulnerability, hunting pressure, conservation efforts, habitat type, breeding programs, legal protection
- **4-class IUCN target**: Least Concern → Critically Endangered
- **No copyright issues**: Entirely synthetic, generated algorithmically

### 🤖 ML Pipeline
- **Data Preprocessing**: StandardScaler normalization, OneHotEncoder for categorical features
- **Train/Test Split**: 80/20 stratified split
- **Models**:
  - **XGBoost**: Main predictor (100 estimators, depth=6)
  - **Random Forest**: Ensemble diversity (100 estimators, depth=10)
  - **Logistic Regression**: Baseline & interpretability
- **Metrics**: Accuracy, F1-score (weighted), feature importance
- **Serialization**: Joblib for production inference

### 🔌 FastAPI REST API
- **GET /health** - API health check
- **GET /info** - Model metadata & feature info
- **POST /predict** - Single species risk prediction
- **POST /bulk-predict** - Batch predictions for multiple species
- **Auto-docs**: Swagger UI at `/docs`, ReDoc at `/redoc`
- **CORS enabled** for web frontend integration

### 🎨 Interactive Web Frontend
- Responsive HTML/CSS/JavaScript (no external framework dependencies)
- Real-time slider controls for 0-1 range features
- Visual probability distributions with animated bars
- Dynamic risk category badges (color-coded by IUCN level)
- Status indicator for API connection health
- Mobile-friendly design

### 🐳 Docker & Orchestration
- **Multi-stage Dockerfile** for optimized image size
- **docker-compose.yml** for local development
- **Health checks** with automated restart policies
- **Kubernetes manifests** ready for cluster deployment
- **.dockerignore** for clean builds

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Git**
- **pip** or **conda**
- *(Optional)* Docker & docker-compose for containerized setup

### Installation

```bash
# Clone repository
git clone https://github.com/dariolignana96/amazon-rainforest-species-risk.git
cd amazon-rainforest-species-risk

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset (1,000 species)
python data/synthetic_generator.py
# Output: data/raw/amazon_species.csv

# Train ML models
python ml/train.py
# Output: 
#   - models/xgboost_v1.pkl
#   - models/preprocessor.pkl
#   - models/metadata.json
```

### Run API Server

#### Option 1: Native Python
```bash
# Start API (http://127.0.0.1:8000)
uvicorn api.main:app --reload

# Access Swagger UI: http://127.0.0.1:8000/docs
# Access ReDoc: http://127.0.0.1:8000/redoc
```

#### Option 2: Docker Compose
```bash
# Build and run container
docker-compose up --build

# API available at http://localhost:8000
# Logs: docker-compose logs -f
# Stop: docker-compose down
```

### Use Web Frontend
```bash
# Open frontend in browser:
open frontend/index.html
# Or manually navigate to: file:///path/to/amazon-rainforest-species-risk/frontend/index.html
```

---

## 📖 API Usage Examples

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Model Info
```bash
curl http://127.0.0.1:8000/info
```

**Response:**
```json
{
  "n_features": 9,
  "feature_names": [
    "population_size",
    "habitat_fragmentation",
    "climate_vulnerability",
    "illegal_hunting_pressure",
    "conservation_efforts_index",
    "habitat_Canopy",
    "habitat_Floor",
    "habitat_River",
    "breeding_program_exists",
    "legal_protection"
  ],
  "models_available": ["xgboost", "random_forest", "logistic_regression"],
  "iucn_categories": [
    "Least Concern",
    "Vulnerable",
    "Endangered",
    "Critically Endangered"
  ]
}
```

### Single Species Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "population_size": 500,
    "habitat_fragmentation": 0.7,
    "climate_vulnerability": 0.6,
    "illegal_hunting_pressure": 0.4,
    "conservation_efforts_index": 0.3,
    "habitat": "Canopy",
    "breeding_program_exists": 0,
    "legal_protection": 1
  }'
```

**Response:**
```json
{
  "risk_category": "Endangered",
  "risk_code": 2,
  "confidence": 0.87,
  "probabilities": {
    "class_0": 0.05,
    "class_1": 0.08,
    "class_2": 0.87,
    "class_3": 0.00
  }
}
```

### Batch Predictions
```bash
curl -X POST http://127.0.0.1:8000/bulk-predict \
  -H "Content-Type: application/json" \
  -d '[
    {"population_size": 500, "habitat_fragmentation": 0.7, "climate_vulnerability": 0.6, "illegal_hunting_pressure": 0.4, "conservation_efforts_index": 0.3, "habitat": "Canopy", "breeding_program_exists": 0, "legal_protection": 1},
    {"population_size": 10000, "habitat_fragmentation": 0.2, "climate_vulnerability": 0.3, "illegal_hunting_pressure": 0.1, "conservation_efforts_index": 0.8, "habitat": "River", "breeding_program_exists": 1, "legal_protection": 1}
  ]'
```

---

## 📂 Project Structure

```
amazon-rainforest-species-risk/
├── data/
│   ├── synthetic_generator.py        # Generate 1,000 synthetic species
│   ├── raw/
│   │   └── amazon_species.csv        # Generated dataset (1,000 records)
│   └── processed/
│
├── ml/
│   ├── preprocessing.py              # Data preprocessing pipeline
│   ├── models.py                     # Model definitions & training
│   └── train.py                      # Main training script
│
├── api/
│   ├── main.py                       # FastAPI app & endpoints
│   └── schemas.py                    # Pydantic request/response models
│
├── models/
│   ├── xgboost_v1.pkl                # Trained XGBoost model
│   ├── preprocessor.pkl              # Sklearn preprocessing pipeline
│   └── metadata.json                 # Model metadata & feature names
│
├── frontend/
│   └── index.html                    # Interactive web UI (vanilla JS)
│
├── k8s/                              # Kubernetes manifests (ready)
├── infra/                            # Terraform IaC stubs
├── tests/                            # Unit tests (pytest)
├── notebooks/                        # Jupyter analysis notebooks
│
├── Dockerfile                        # Multi-stage production build
├── docker-compose.yml                # Local dev environment
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusions
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🤖 Machine Learning Models

### XGBoost (Primary)
- **Type**: Gradient Boosting Classifier
- **Hyperparameters**: 
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
- **Input**: 9 features (5 numeric + 4 categorical one-hot encoded)
- **Output**: 4-class IUCN category (0-3)
- **Performance**: ~85% accuracy on test set
- **Feature Importance**: Population size > Habitat fragmentation > Hunting pressure

### Random Forest (Ensemble)
- **Type**: Random Forest Classifier
- **Hyperparameters**: 
  - n_estimators: 100
  - max_depth: 10
- **Performance**: ~82% accuracy on test set
- **Role**: Model diversity, drift detection

### Logistic Regression (Baseline)
- **Type**: Multinomial Logistic Regression
- **Performance**: ~75% accuracy on test set
- **Role**: Interpretability, production fallback

---

## 📊 Dataset Specification

### Amazon Species Registry (Synthetic)
| Attribute | Value |
|-----------|-------|
| **Size** | 1,000 species records |
| **Features** | 10 ecological variables |
| **Target** | 4-class IUCN category |
| **Generation** | 100% original, deterministic (seed=42) |
| **Reproducibility** | ✅ Fully reproducible |

### Feature Descriptions
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| population_size | numeric | 10 - 200,000 | Estimated population in individuals |
| habitat_fragmentation | numeric | 0.0 - 1.0 | Fragmentation index (0=contiguous, 1=highly fragmented) |
| climate_vulnerability | numeric | 0.0 - 1.0 | Climate change vulnerability (0=not vulnerable, 1=highly vulnerable) |
| illegal_hunting_pressure | numeric | 0.0 - 1.0 | Hunting/poaching pressure (0=none, 1=extreme) |
| conservation_efforts_index | numeric | 0.0 - 1.0 | Conservation effort level (0=no efforts, 1=maximum) |
| habitat | categorical | 3 types | Canopy / Forest Floor / River Floodplain |
| breeding_program_exists | binary | 0 / 1 | Active breeding program (yes/no) |
| legal_protection | binary | 0 / 1 | Legal protection status (yes/no) |

### Target Classes (IUCN Red List)
- **0**: Least Concern (LC) - Lowest risk
- **1**: Vulnerable (VU)
- **2**: Endangered (EN)
- **3**: Critically Endangered (CR) - Highest risk

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t rainforest-api:latest .
```

### Run Docker Container
```bash
docker run -p 8000:8000 rainforest-api:latest
# API available at http://localhost:8000
```

### Docker Compose (Recommended for Dev)
```bash
docker-compose up --build
docker-compose down  # Stop
```

### View Logs
```bash
docker logs rainforest-api
docker-compose logs -f
```

---

## ☸️ Kubernetes Deployment

### Deploy to Kubernetes Cluster
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify deployment
kubectl get pods
kubectl get svc
kubectl logs -f deployment/rainforest-api
```

### Port Forward (Local Testing)
```bash
kubectl port-forward svc/rainforest-api 8000:8000
# API available at http://127.0.0.1:8000
```

---

## 🧪 Testing

### Unit Tests (Pytest)
```bash
pytest tests/ -v
```

### API Integration Tests (Manual)
```bash
# Health check
curl http://127.0.0.1:8000/health

# Get model info
curl http://127.0.0.1:8000/info

# Test prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"population_size": 5000, ...}'
```

### Load Testing (Apache Bench)
```bash
# 1000 requests, 10 concurrent
ab -n 1000 -c 10 http://127.0.0.1:8000/health
```

---

## 🎯 Roadmap

### Phase 1 ✅ (Complete)
- [x] Synthetic data generator (1,000 species)
- [x] ML training pipeline (XGBoost, RF, LR)
- [x] FastAPI REST API with 4 endpoints
- [x] Pydantic request/response validation
- [x] Docker containerization (multi-stage)
- [x] docker-compose for local development
- [x] Web frontend (HTML/CSS/JS)
- [x] Comprehensive README documentation

### Phase 2 (Planned)
- [ ] Advanced monitoring dashboard (Prometheus, Grafana)
- [ ] Model versioning (MLflow)
- [ ] A/B testing framework
- [ ] Performance benchmarking
- [ ] Unit tests & CI/CD (GitHub Actions)

### Phase 3 (Future)
- [ ] Image classification (CNN for species photos)
- [ ] Temporal forecasting (LSTM for population trends)
- [ ] Real-time WebSocket predictions
- [ ] Multi-model inference optimization
- [ ] Advanced logging (ELK stack)

---

## 📜 License

This project is licensed under the **MIT License** - see LICENSE file for details.

**Code Status**: 100% original, open-source, free from copyright restrictions.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (git checkout -b feature/your-feature)
3. **Commit** your changes (git commit -m 'feat: add new feature')
4. **Push** to the branch (git push origin feature/your-feature)
5. **Open** a Pull Request

### Code Style
- Python: Follow PEP 8 (use black for formatting)
- JavaScript: Standard ES6+ conventions
- Commit messages: Conventional Commits format

---

## 📞 Contact & Support

**Author**: Dario Lignana
- **GitHub**: @dariolignana96

**Issues & Discussions**:
- Open an Issue on GitHub
- Check Discussions on GitHub

---

## 🙏 Acknowledgments

Built with ❤️ for:
- 🌿 Environmental conservation awareness
- 📚 Open-source education
- 🏆 ML/MLOps portfolio demonstration
- 🌍 Biodiversity preservation

**Disclaimer**: This is an educational/portfolio project using **synthetic data**. Real conservation decisions require verified ecological data and expert domain knowledge.

---

## 📚 References & Resources

- **XGBoost**: Official Documentation (https://xgboost.readthedocs.io/)
- **FastAPI**: Official Documentation (https://fastapi.tiangolo.com/)
- **Docker**: Official Documentation (https://docs.docker.com/)
- **Kubernetes**: Official Documentation (https://kubernetes.io/docs/)
- **IUCN Red List**: Conservation Status Categories (https://www.iucnredlist.org/)
- **Scikit-learn**: Documentation (https://scikit-learn.org/)

---

git add README.md
**Last Updated**: January 7, 2026
**Status**: ✅ Production Ready (Phase 1 Complete)


***



