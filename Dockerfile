# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copia dipendenze da builder
COPY --from=builder /root/.local /root/.local

# Copia codice
COPY data/ ./data/
COPY ml/ ./ml/
COPY api/ ./api/
COPY models/ ./models/

ENV PATH=/root/.local/bin:
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
