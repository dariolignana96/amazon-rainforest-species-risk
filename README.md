# Amazon Rainforest Wildlife Risk Assessment

ML/MLOps full-stack application predicting extinction risk for Amazon rainforest
species. Ensemble model (XGBoost + Random Forest + Logistic Regression) served
via FastAPI, containerized with Docker, and deployable on Kubernetes.
All data is synthetic and generated locally — no external services required.

## Stack

- Python 3.11 — XGBoost, scikit-learn, pandas, FastAPI, Pydantic, Uvicorn
- Frontend: HTML5, CSS3, JavaScript (vanilla)
- Docker, Docker Compose
- Kubernetes (Deployment + Service manifests)
- Terraform (AWS VPC design, local validation only)

## Structure

    amazon-rainforest-species-risk/
    ├── api/
    │   ├── main.py                    # FastAPI app & endpoints
    │   ├── models.py                  # Pydantic schemas
    │   └── mock_data.py               # Synthetic species database
    ├── frontend/
    │   └── index.html                 # Single-page web UI
    ├── ml/
    │   ├── preprocessing.py           # Data preprocessing pipeline
    │   ├── models.py                  # Model definitions & training
    │   └── train.py                   # Training entry point
    ├── models/                        # Serialized models (joblib)
    ├── data/
    │   ├── synthetic_generator.py
    │   └── raw/amazon_species.csv
    ├── k8s/
    │   ├── deployment.yaml
    │   └── service.yaml
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    └── README.md

## Setup

### Local

    git clone https://github.com/dariolignana96/amazon-rainforest-species-risk.git
    cd amazon-rainforest-species-risk
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    uvicorn api.main:app --reload

- API: http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

Frontend: open `frontend/index.html` directly, or:

    cd frontend
    python -m http.server 5500

Then open http://127.0.0.1:5500/index.html.

### Docker

    docker-compose up --build

API available at http://localhost:8000.

### Kubernetes

    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl port-forward svc/rainforest-api 8000:8000

## ML Models

Voting ensemble combining three classifiers trained on 1,000 synthetic species
records with 10 ecological features (population size, habitat fragmentation,
climate vulnerability, hunting pressure, conservation efforts, habitat type,
breeding program, legal protection).

Target: 4-class IUCN category (Least Concern / Vulnerable / Endangered /
Critically Endangered).

| Model | Accuracy | Ensemble Weight |
|---|---|---|
| XGBoost | ~85% | 50% |
| Random Forest | ~82% | 30% |
| Logistic Regression | ~75% | 20% |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| GET | /info | Model metadata & feature names |
| POST | /predict | Single species prediction |
| POST | /bulk-predict | Batch predictions |

## Infrastructure (Terraform)

The `infra/` directory contains an AWS VPC design for local validation only.
No resources are provisioned by default.

    cd infra
    terraform init
    terraform validate

## License

MIT — see [LICENSE](LICENSE) for details.
Dataset: 100% synthetic, generated algorithmically. No third-party data included.