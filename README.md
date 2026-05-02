# 🌿 Amazon Rainforest Wildlife Risk Assessment

ML/MLOps full-stack application predicting extinction risk for Amazon rainforest
species. Ensemble model (XGBoost + Random Forest + Logistic Regression) served
via FastAPI, containerized with Docker, and deployable on Kubernetes.
All data is synthetic and generated locally — no external services required.

> **Progetto formativo/portfolio** — nessun servizio cloud attivo, nessun costo.

---

## ⚠️ Nota costi

| Componente | Costo |
|---|---|
| Python, FastAPI, ML models | ✅ Gratuito — gira in locale |
| Docker / Docker Compose | ✅ Gratuito — gira in locale |
| Kubernetes (minikube/kind) | ✅ Gratuito — gira in locale |
| `terraform init` / `validate` / `plan` | ✅ Gratuito — solo verifica locale |
| `terraform apply` su AWS/Azure | ❌ Genera costi — non eseguire |

**Regola:** tutto fino a `terraform plan` è sicuro e non genera costi.  
`terraform apply` non va eseguito senza un account cloud attivo e consapevolezza dei costi.

I modelli ML (`.pkl`) non sono inclusi nel repository — vanno rigenerati localmente con:

```bash
python ml/train.py
```

---

## Stack

- **Backend:** Python 3, XGBoost, scikit-learn, pandas, FastAPI, Uvicorn
- **Frontend:** HTML5, CSS3, JavaScript (vanilla)
- **Container:** Docker, Docker Compose
- **Orchestrazione:** Kubernetes (Deployment + Service manifests)
- **Infrastruttura:** Terraform — AWS + Azure (solo pianificazione, no deploy attivo)
- **CI/CD:** GitHub Actions — terraform validate + Python lint ad ogni push

---

## Struttura

```
amazon-rainforest-species-risk/
├── api/
│   ├── main.py                  # FastAPI app & endpoints
│   ├── schemas.py               # Pydantic schemas
│   └── routers/                 # Endpoint modulari (espandibile)
├── frontend/
│   └── index.html               # Single-page web UI
├── ml/
│   ├── preprocessing.py         # Data preprocessing pipeline
│   ├── models.py                # Model definitions & training
│   └── train.py                 # Training entry point
├── models/                      # Modelli .pkl — generati con: python ml/train.py
├── data/
│   ├── synthetic_generator.py
│   ├── raw/amazon_species.csv
│   └── processed/               # Output preprocessing (generato localmente)
├── k8s/
│   ├── deployment.yaml          # Kubernetes Deployment (2 repliche)
│   └── service.yaml             # Kubernetes Service (LoadBalancer)
├── .github/
│   └── workflows/
│       └── ci.yml               # CI: terraform validate + Python lint
├── docs/                        # Documentazione (work in progress)
├── tests/                       # Test suite (work in progress)
├── terraform/                   # Infrastructure as Code — AWS + Azure
│   ├── main.tf                  # Entry point multi-cloud
│   ├── variables.tf
│   ├── outputs.tf
│   └── modules/
│       ├── aws/                 # VPC, EC2, S3, IAM, ECS, Lambda, API Gateway
│       └── azure/               # Resource Group, VNet, ACI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

```

---

## Setup locale

```bash
git clone https://github.com/dariolignana96/amazon-rainforest-species-risk.git
cd amazon-rainforest-species-risk
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
python ml/train.py        # genera i modelli .pkl in models/
uvicorn api.main:app --reload
```

- API: http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

Frontend — apri `frontend/index.html` direttamente, oppure:

```bash
cd frontend
python -m http.server 5500
```

Poi apri http://127.0.0.1:5500/index.html.

---

## Docker

```bash
docker-compose up --build
```

API disponibile su http://localhost:8000.

---

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl port-forward svc/rainforest-api 8000:8000
```

---

## Terraform (infrastruttura — solo pianificazione)

Infrastruttura multi-cloud definita come codice. Nessuna risorsa viene creata
eseguendo solo i comandi di verifica.

```bash
cd terraform/
terraform init        # scarica i provider AWS e Azure
terraform validate    # verifica la sintassi
terraform plan        # mostra il piano (nessun costo, nessuna risorsa creata)
```

Architettura pianificata:
- **AWS:** VPC, EC2 t2.micro (free tier), S3, IAM, ECS Fargate, Lambda, API Gateway
- **Azure:** Resource Group, VNet, Container Instance (ACI)

---

## CI/CD

GitHub Actions esegue automaticamente ad ogni push su `main`:
- `terraform validate` — verifica sintassi infrastruttura
- `terraform fmt -check` — verifica formattazione
- `flake8` — lint del codice Python

---

## ML Models

Voting ensemble su 1.000 record sintetici con 10 feature ecologiche.
Target: 4 classi IUCN (Least Concern / Vulnerable / Endangered / Critically Endangered).

| Model | Accuracy | Peso ensemble |
|---|---|---|
| XGBoost | ~85% | 50% |
| Random Forest | ~82% | 30% |
| Logistic Regression | ~75% | 20% |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| GET | /info | Model metadata & feature names |
| POST | /predict | Single species prediction |
| POST | /bulk-predict | Batch predictions |

---

## License

MIT — see [LICENSE](LICENSE) for details.  
Dataset: 100% sintetico, generato algoritmicamente. Nessun dato di terze parti.