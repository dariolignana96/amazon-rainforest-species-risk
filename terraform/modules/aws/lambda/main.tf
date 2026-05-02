# =============================================================================
# MODULO: AWS Lambda
# =============================================================================
# Lambda = funzione serverless: eseguita on-demand, paghi solo per le esecuzioni.
# FREE TIER: 1 milione di invocazioni/mese + 400.000 GB-secondi/mese → GRATIS.
#
# Caso d'uso qui: endpoint /predict leggero per singole predizioni,
# senza dover tenere ECS sempre attivo in un ambiente a basso traffico.
#
# Il codice Lambda viene pacchettizzato in un .zip e caricato su S3.
# =============================================================================

# Crea il pacchetto ZIP della Lambda dal codice Python
data "archive_file" "lambda_package" {
  type        = "zip"
  source_dir  = "${path.module}/src"
  output_path = "${path.module}/lambda_package.zip"
}

# --- Lambda Function ---
resource "aws_lambda_function" "predict" {
  function_name = "${var.project}-${var.environment}-predict"
  description   = "Endpoint serverless per predizione rischio specie"

  filename         = data.archive_file.lambda_package.output_path
  source_code_hash = data.archive_file.lambda_package.output_base64sha256
  # source_code_hash cambia solo se il codice cambia → Terraform re-deploya solo se necessario

  runtime = "python3.11"
  handler = "handler.lambda_handler" # file handler.py, funzione lambda_handler

  role        = var.lambda_role_arn
  timeout     = 30  # secondi (max 15 minuti, ma 30s è ok per inference ML)
  memory_size = 512 # MB - più memoria = più CPU proporzionalmente

  environment {
    variables = {
      ENVIRONMENT      = var.environment
      S3_BUCKET        = var.s3_bucket
      MODEL_KEY        = "models/xgboost_v1.pkl"
      PREPROCESSOR_KEY = "models/preprocessor.pkl"
    }
  }

  # X-Ray tracing: analisi performance e latenza
  tracing_config {
    mode = "Active"
  }

  tags = {
    Name = "${var.project}-${var.environment}-predict-lambda"
  }
}

# CloudWatch Logs automatici per Lambda
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${aws_lambda_function.predict.function_name}"
  retention_in_days = 14
}
