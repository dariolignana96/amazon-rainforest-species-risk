#!/bin/bash
# =============================================================================
# USER DATA SCRIPT - Eseguito al primo avvio dell'istanza EC2
# =============================================================================
set -e
exec > >(tee /var/log/user-data.log) 2>&1

echo "=== [$(date)] Avvio bootstrap Rainforest API ==="

# Aggiorna pacchetti
dnf update -y

# Installa Docker
dnf install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Installa AWS CLI v2 (già presente su Amazon Linux 2023)
aws --version

# Crea directory per i modelli
mkdir -p /opt/rainforest/models

# Scarica modelli ML dal bucket S3
echo "=== Download modelli ML da S3 ==="
aws s3 cp s3://${s3_bucket}/models/ /opt/rainforest/models/ \
    --recursive \
    --region ${aws_region}

echo "Modelli scaricati:"
ls -la /opt/rainforest/models/

# Avvia il container FastAPI
echo "=== Avvio container Docker ==="
docker run -d \
    --name rainforest-api \
    --restart unless-stopped \
    -p 8000:8000 \
    -v /opt/rainforest/models:/app/models:ro \
    -e ENVIRONMENT=${environment} \
    -e AWS_DEFAULT_REGION=${aws_region} \
    dariolignana96/rainforest-api:latest

# Health check
sleep 10
curl -f http://localhost:8000/health && echo "=== API avviata con successo ===" || echo "=== ERRORE: API non risponde ==="

echo "=== Bootstrap completato: $(date) ==="
