# 🌿 Amazon Rainforest Wildlife Risk Assessment

**Interactive ML/MLOps Portfolio Project** - Predicting extinction risk for Amazon rainforest species using synthetic data, XGBoost ensemble models, FastAPI, Docker, and modern web frontend.

[![GitHub](https://img.shields.io/badge/GitHub-dariolignana96-blue?style=flat-square)](https://github.com/dariolignana96)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)](https://fastapi.tiangolo.com/)

---

## 📊 Project Overview

A **complete, production-ready full-stack application** demonstrating ML/MLOps best practices:

**Backend:**
- 🔬 **Synthetic Data Generation**: 1,000+ realistic Amazon species with 10 ecological features
- 🤖 **Ensemble ML Models**: XGBoost (primary), Random Forest, Logistic Regression with voting ensemble
- 🔌 **FastAPI REST API**: Production-ready endpoints for single & batch predictions
- 🐳 **Docker Containerization**: Multi-stage builds, ready for deployment
- ☸️ **Kubernetes Ready**: YAML manifests included

**Frontend:**
- 🎨 **Modern Interactive UI**: HTML5/CSS3/Vanilla JavaScript (no external dependencies)
- 🌿 **Amazon Rainforest Theme**: Green glassmorphism design with dark mode
- 🦁 **10 Pre-defined Species**: Auto-fill parameters with real ecological data
  - 🐆 Amazon Jaguar
  - 🐍 Green Anaconda
  - 🦜 Scarlet Macaw
  - 🐬 Pink River Dolphin
  - 🦥 Three-Toed Sloth
  - 🐸 Poison Dart Frog
  - 🐵 Red Howler Monkey
  - 🐟 Arapaima Fish
  - 🦅 Harpy Eagle
  - 🐭 Capybara
- 📊 **Real-time Predictions**: Animated probability distribution charts
- 🔌 **Live API Status**: Connection health indicator with auto-reconnect

**Real-world skills demonstrated:**
- Full-stack development (Frontend + Backend integration)
- Data generation & preprocessing (Pandas, Scikit-learn)
- Machine learning (XGBoost, Random Forest, Logistic Regression)
- API design (FastAPI, Pydantic validation, CORS)
- Frontend development (Responsive HTML/CSS, Vanilla JS, DOM manipulation)
- Containerization (Docker, docker-compose)
- Version control (Git, GitHub workflow)
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
  - **XGBoost**: Main predictor (100 estimators, depth=6, accuracy: ~85%)
  - **Random Forest**: Ensemble diversity (100 estimators, depth=10, accuracy: ~82%)
  - **Logistic Regression**: Baseline & interpretability (accuracy: ~75%)
- **Ensemble Strategy**: Voting classifier averaging all 3 models
- **Metrics**: Accuracy, F1-score (weighted), feature importance
- **Serialization**: Joblib for production inference

### 🔌 FastAPI REST API
- **GET /health** - API health check with model status
- **GET /info** - Model metadata & feature descriptions
- **POST /predict** - Single species risk prediction
- **POST /bulk-predict** - Batch predictions for multiple species
- **CORS enabled** - Cross-origin requests from web frontend
- **Auto-docs**: Swagger UI at `/docs`, ReDoc at `/redoc`
- **Error handling**: Comprehensive validation & error messages

### 🎨 Interactive Web Frontend
- **Responsive Design**: Works on desktop, tablet, mobile
- **Species Dropdown**: 10 pre-defined Amazon species with auto-fill
- **Interactive Sliders**: 
  - Population Size (10 - 50,000)
  - Habitat Fragmentation (0 - 1)
  - Climate Vulnerability (0 - 1)
  - Hunting Pressure (0 - 1)
  - Conservation Efforts (0 - 1)
- **Real-time Updates**: Slider values update instantly with gradient effects
- **Checkboxes**: Breeding program status, legal protection
- **Live Results**:
  - Risk category badge (color-coded by IUCN level)
  - Confidence score with visual indicator
  - Probability distribution bars (animated)
- **Status Indicator**: Real-time API connection health
- **Notifications**: Toast notifications for success/error messages
- **Dark Mode**: Beautiful green forest theme with glassmorphism effects

### 🐳 Docker & Orchestration
- **Multi-stage Dockerfile** for optimized image size (~300MB)
- **docker-compose.yml** for local development (API + dependencies)
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

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run API Server

#### Option 1: Native Python (Recommended for Development)
```bash
# Start API server
uvicorn api.main:app --reload

# Output:
# INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
# INFO:     Application startup complete
# ✅ Models loaded successfully
```

**Access:**
- API base: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health: http://127.0.0.1:8000/health

#### Option 2: Docker Compose (Recommended for Deployment)
```bash
# Build and run container
docker-compose up --build

# API available at http://localhost:8000

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Open Web Frontend

#### Option 1: Direct File (Simplest)
```bash
# Windows
start frontend/index.html

# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html
```

#### Option 2: HTTP Server (Recommended)
```bash
cd frontend
python -m http.server 5500

# Then open: http://127.0.0.1:5500/index.html
```

---

## 🧪 Testing the Application

### Test 1: Load a Pre-defined Species
1. Open http://127.0.0.1:5500/index.html (or direct file)
2. Check API status badge (should show ✅ API Connected)
3. Select "🐆 Amazon Jaguar" from dropdown
4. Verify form populates:
   - Population: 173
   - Fragmentation: 0.85
   - Climate: 0.72
   - Hunting: 0.78
   - Conservation: 0.45
   - Habitat: Canopy
   - Breeding: ✓ (checked)
   - Legal: ✓ (checked)
5. Click "Predict Risk"
6. Result should show: **🔴 Critically Endangered** (~85% confidence)

### Test 2: Compare Multiple Species
1. Select "🐭 Capybara"
2. Form updates automatically
3. Click "Predict"
4. Result: **✅ Least Concern** (~90% confidence)

### Test 3: Manual Custom Assessment
1. Click "Reset"
2. Manually adjust sliders for extreme values
3. Set all to maximum risk (pop: 100, frag: 1.0, climate: 1.0, etc.)
4. Result should be: **🔴 Critically Endangered**

### Test 4: API Testing with cURL
```bash
# Health check
curl http://127.0.0.1:8000/health

# Model info
curl http://127.0.0.1:8000/info

# Single prediction (Jaguar parameters)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "population_size": 173,
    "habitat_fragmentation": 0.85,
    "climate_vulnerability": 0.72,
    "illegal_hunting_pressure": 0.78,
    "conservation_efforts_index": 0.45,
    "habitat": "Canopy",
    "breeding_program_exists": 1,
    "legal_protection": 1
  }'

# Expected response:
# {
#   "risk_category": "Critically Endangered",
#   "risk_code": 3,
#   "confidence": 0.85,
#   "probabilities": {
#     "class_0": 0.02,
#     "class_1": 0.05,
#     "class_2": 0.08,
#     "class_3": 0.85
#   }
# }
```

---

## 📖 API Documentation

### Health Check
```bash
GET /health
```

**Response (200):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Model Information
```bash
GET /info
```

**Response (200):**
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
    "habitat_Aquatic",
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
POST /predict
Content-Type: application/json

{
  "population_size": 5000,
  "habitat_fragmentation": 0.5,
  "climate_vulnerability": 0.5,
  "illegal_hunting_pressure": 0.3,
  "conservation_efforts_index": 0.5,
  "habitat": "Canopy",
  "breeding_program_exists": 0,
  "legal_protection": 1
}
```

**Response (200):**
```json
{
  "risk_category": "Vulnerable",
  "risk_code": 1,
  "confidence": 0.76,
  "probabilities": {
    "class_0": 0.15,
    "class_1": 0.76,
    "class_2": 0.09,
    "class_3": 0.00
  }
}
```

### Batch Predictions
```bash
POST /bulk-predict
Content-Type: application/json

[
  {
    "population_size": 173,
    "habitat_fragmentation": 0.85,
    ...
  },
  {
    "population_size": 15000,
    "habitat_fragmentation": 0.55,
    ...
  }
]
```

**Response (200):**
```json
{
  "predictions": [
    {
      "risk_category": "Critically Endangered",
      "confidence": 0.85,
      ...
    },
    {
      "risk_category": "Least Concern",
      "confidence": 0.90,
      ...
    }
  ]
}
```

---

## 📂 Project Structure

```
amazon-rainforest-species-risk/
├── api/
│   ├── main.py                       # FastAPI app & endpoints
│   ├── models.py                     # Pydantic request/response schemas
│   └── mock_data.py                  # Synthetic species database
│
├── frontend/
│   └── index.html                    # Interactive web UI (HTML/CSS/JS)
│   
├── ml/
│   ├── preprocessing.py              # Data preprocessing pipeline
│   ├── models.py                     # Model definitions & training
│   └── train.py                      # Main training script
│
├── models/
│   ├── xgboost_v1.pkl                # Trained XGBoost model
│   ├── random_forest_v1.pkl          # Trained Random Forest model
│   ├── logistic_regression_v1.pkl    # Trained Logistic Regression model
│   ├── preprocessor.pkl              # Sklearn preprocessing pipeline
│   └── metadata.json                 # Model metadata & feature names
│
├── data/
│   ├── synthetic_generator.py        # Generate 1,000 synthetic species
│   ├── raw/
│   │   └── amazon_species.csv        # Generated dataset (1,000 records)
│   └── processed/
│
├── k8s/                              # Kubernetes manifests (ready)
│   ├── deployment.yaml
│   └── service.yaml
│
├── Dockerfile                        # Multi-stage production build
├── docker-compose.yml                # Local dev environment
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusions
├── LICENSE                           # MIT License
├── ARCHITETTURA_PROGETTO.md          # Architecture documentation
└── README.md                         # This file
```

---

## 🤖 Machine Learning Models

### Ensemble Strategy
The API uses a **voting ensemble** combining 3 models:

**Model 1: XGBoost (Primary)**
- Type: Gradient Boosting Classifier
- Hyperparameters: n_estimators=100, max_depth=6, learning_rate=0.1
- Accuracy: ~85% on test set
- Weight in ensemble: 50%

**Model 2: Random Forest (Diversity)**
- Type: Random Forest Classifier
- Hyperparameters: n_estimators=100, max_depth=10
- Accuracy: ~82% on test set
- Weight in ensemble: 30%

**Model 3: Logistic Regression (Baseline)**
- Type: Multinomial Logistic Regression
- Accuracy: ~75% on test set
- Weight in ensemble: 20%

**Combined Performance**: ~85% accuracy, robust to individual model failures

### Feature Importance (XGBoost)
1. Population Size (25%)
2. Habitat Fragmentation (22%)
3. Hunting Pressure (18%)
4. Climate Vulnerability (15%)
5. Conservation Efforts (12%)
6. Other features (8%)

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
| climate_vulnerability | numeric | 0.0 - 1.0 | Climate change vulnerability (0=not, 1=highly) |
| illegal_hunting_pressure | numeric | 0.0 - 1.0 | Hunting/poaching pressure (0=none, 1=extreme) |
| conservation_efforts_index | numeric | 0.0 - 1.0 | Conservation effort level (0=none, 1=maximum) |
| habitat | categorical | 3 types | Canopy / Forest Floor / Aquatic |
| breeding_program_exists | binary | 0 / 1 | Active breeding program (yes/no) |
| legal_protection | binary | 0 / 1 | Legal protection status (yes/no) |

### Target Classes (IUCN Red List)
- **0**: Least Concern (LC) - ✅ Lowest risk
- **1**: Vulnerable (VU) - ⚠️ Medium risk
- **2**: Endangered (EN) - 🚨 High risk
- **3**: Critically Endangered (CR) - 🔴 Highest risk

---

## 🌿 Pre-defined Species

The frontend includes 10 Amazon rainforest species with realistic ecological parameters:

| Species | Emoji | Population | Fragmentation | Risk Level |
|---------|-------|-----------|--------------|-----------|
| Amazon Jaguar | 🐆 | 173 | 0.85 | 🔴 Critically Endangered |
| Green Anaconda | 🐍 | 8,500 | 0.65 | ⚠️ Vulnerable |
| Scarlet Macaw | 🦜 | 5,200 | 0.72 | ⚠️ Vulnerable |
| Pink River Dolphin | 🐬 | 3,800 | 0.58 | ⚠️ Vulnerable |
| Three-Toed Sloth | 🦥 | 12,000 | 0.68 | ✅ Least Concern |
| Poison Dart Frog | 🐸 | 2,500 | 0.75 | 🚨 Endangered |
| Red Howler Monkey | 🐵 | 6,800 | 0.62 | ✅ Least Concern |
| Arapaima Fish | 🐟 | 4,200 | 0.70 | ⚠️ Vulnerable |
| Harpy Eagle | 🦅 | 580 | 0.82 | 🚨 Endangered |
| Capybara | 🐭 | 15,000 | 0.55 | ✅ Least Concern |

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

### Docker Compose (Complete Stack)
```bash
# Start with all dependencies
docker-compose up --build

# Stop
docker-compose down

# View logs
docker-compose logs -f api
```

### Verify Deployment
```bash
# Check container is running
docker ps

# Test health endpoint
curl http://localhost:8000/health

# View logs
docker logs rainforest-api
```

---

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (minikube, kind, or cloud)
- kubectl configured

### Deploy
```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify
kubectl get pods
kubectl get svc
kubectl describe deployment rainforest-api

# View logs
kubectl logs -f deployment/rainforest-api
```

### Port Forward (Local Testing)
```bash
kubectl port-forward svc/rainforest-api 8000:8000
# API available at http://127.0.0.1:8000
```

### Cleanup
```bash
kubectl delete -f k8s/
```

---

## 🧪 Testing

### API Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Load Testing (Apache Bench)
```bash
# 1000 requests, 10 concurrent
ab -n 1000 -c 10 http://127.0.0.1:8000/health
```

### Frontend Testing Checklist
- [ ] API status shows "✅ API Connected"
- [ ] Dropdown loads 10 species
- [ ] Selecting species auto-fills form
- [ ] Sliders update values in real-time
- [ ] Predict button works and shows results
- [ ] Results display correct risk category (color-coded)
- [ ] Probability bars animate smoothly
- [ ] Toast notifications appear
- [ ] Reset button clears form
- [ ] Responsive on mobile (check with DevTools)

---

## 🔧 Troubleshooting

### API not connecting from frontend
```bash
# Check if API is running
curl http://127.0.0.1:8000/health

# If using VPN or local IP, update API_URL in index.html:
# Change: const API_URL = 'http://127.0.0.1:8000';
# To: const API_URL = 'http://192.168.1.100:8000';

# Add CORS middleware in main.py (already included)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Models not loading
```bash
# Verify model files exist
ls -la models/

# Retrain if missing
python ml/train.py

# Check Python path
echo $PYTHONPATH
```

### Docker build fails
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker build --no-cache -t rainforest-api:latest .
```

---

## 📜 License

This project is licensed under the **MIT License** - see LICENSE file for details.

**Code Status**: 100% original, open-source, free from copyright restrictions.

---

## 🎯 Roadmap

### Phase 1 ✅ (Complete)
- [x] Synthetic data generator (1,000 species)
- [x] ML training pipeline (XGBoost, RF, LR ensemble)
- [x] FastAPI REST API with 4 endpoints
- [x] Docker containerization
- [x] Web frontend with species dropdown
- [x] Real-time API status indicator
- [x] Probability distribution visualization
- [x] Comprehensive README

### Phase 2 (Planned)
- [ ] Unit tests & pytest suite
- [ ] CI/CD with GitHub Actions
- [ ] Advanced monitoring dashboard
- [ ] Model versioning (MLflow)
- [ ] A/B testing framework

### Phase 3 (Future)
- [ ] Image classification (CNN for species photos)
- [ ] Temporal forecasting (population trends)
- [ ] Real-time WebSocket predictions
- [ ] Advanced logging & monitoring

---

## 🤝 Contributing

Contributions welcome! Follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style
- **Python**: PEP 8 (use black for formatting)
- **JavaScript**: ES6+ conventions
- **Commit messages**: Conventional Commits format

---

## 📞 Contact

**Author**: Dario Lignana

**Links**:
- GitHub: [@dariolignana96](https://github.com/dariolignana96)
- Project: [amazon-rainforest-species-risk](https://github.com/dariolignana96/amazon-rainforest-species-risk)

**Issues & Support**:
- Open an Issue on GitHub
- Check GitHub Discussions

---

## 🙏 Acknowledgments

Built with ❤️ for:
- 🌿 Environmental conservation awareness
- 📚 Open-source education
- 🏆 ML/MLOps portfolio demonstration
- 🌍 Biodiversity preservation

**Disclaimer**: This is an educational/portfolio project using **synthetic data**. Real conservation decisions require verified ecological data and expert domain knowledge.

---

## 📚 Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **XGBoost**: https://xgboost.readthedocs.io/
- **Docker**: https://docs.docker.com/
- **Kubernetes**: https://kubernetes.io/docs/
- **IUCN Red List**: https://www.iucnredlist.org/
- **Scikit-learn**: https://scikit-learn.org/

---

**Last Updated**: January 7, 2026  
**Status**: ✅ **Production Ready - Phase 1 Complete**  
**Version**: 1.0.0
