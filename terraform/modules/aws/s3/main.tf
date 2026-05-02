# =============================================================================
# MODULO: AWS S3
# =============================================================================
# S3 è lo storage oggetti di AWS: ideale per archiviare i modelli ML (.pkl),
# dati sintetici CSV, e artifact del training pipeline.
#
# Best practice applicate:
#   - Versioning abilitato (rollback a versione precedente del modello)
#   - Encryption at rest (SSE-S3, nessun costo aggiuntivo)
#   - Block public access (modelli non esposti pubblicamente)
#   - Lifecycle policy (elimina versioni vecchie dopo 30gg, risparmio costi)
# =============================================================================

# --- Bucket per i modelli ML ---
resource "aws_s3_bucket" "models" {
  bucket = "${var.project}-${var.environment}-models-${var.aws_region}"
  # I nomi S3 sono globalmente unici: aggiungere region evita collisioni

  tags = {
    Name    = "${var.project}-${var.environment}-models"
    Purpose = "ML model artifacts storage"
  }
}

# Blocca qualsiasi accesso pubblico (ACL, policy pubblica, etc.)
resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Versioning: ogni upload crea una nuova versione, non sovrascrive
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Encryption at rest con chiave gestita da AWS (nessun costo aggiuntivo)
resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Lifecycle: elimina versioni non correnti dopo 30 giorni
resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    id     = "cleanup-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    # Transizione a Glacier dopo 90 giorni (costo ~$0.004/GB invece di $0.023/GB)
    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }
  }
}

# --- Bucket per dati sintetici (CSV del dataset) ---
resource "aws_s3_bucket" "data" {
  bucket = "${var.project}-${var.environment}-data-${var.aws_region}"

  tags = {
    Name    = "${var.project}-${var.environment}-data"
    Purpose = "Training and inference datasets"
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

# Upload automatico dei modelli serializzati dal repository
resource "aws_s3_object" "xgboost_model" {
  bucket = aws_s3_bucket.models.id
  key    = "models/xgboost_v1.pkl"
  source = "${path.root}/../models/xgboost_v1.pkl"
  etag   = filemd5("${path.root}/../models/xgboost_v1.pkl")
}

resource "aws_s3_object" "random_forest_model" {
  bucket = aws_s3_bucket.models.id
  key    = "models/random_forest_v1.pkl"
  source = "${path.root}/../models/random_forest_v1.pkl"
  etag   = filemd5("${path.root}/../models/random_forest_v1.pkl")
}

resource "aws_s3_object" "preprocessor" {
  bucket = aws_s3_bucket.models.id
  key    = "models/preprocessor.pkl"
  source = "${path.root}/../models/preprocessor.pkl"
  etag   = filemd5("${path.root}/../models/preprocessor.pkl")
}
