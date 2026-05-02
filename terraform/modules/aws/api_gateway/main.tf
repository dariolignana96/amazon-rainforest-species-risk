# =============================================================================
# MODULO: AWS API Gateway (HTTP API v2)
# =============================================================================
# API Gateway espone la Lambda come REST endpoint HTTPS pubblico.
# Vantaggi rispetto all'ALB:
#   - HTTPS automatico (no certificato da gestire)
#   - Rate limiting integrato (throttling)
#   - Logging centralizzato
#   - Costo bassissimo: $1/milione di richieste
#
# HTTP API v2 (vs REST API v1): più economica e a latenza minore.
# =============================================================================

# --- HTTP API ---
resource "aws_apigatewayv2_api" "main" {
  name          = "${var.project}-${var.environment}-api"
  protocol_type = "HTTP"
  description   = "API Gateway per Rainforest Species Risk - Lambda integration"

  # CORS: permette al frontend di chiamare questa API da qualsiasi origine
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["Content-Type", "Authorization"]
    max_age       = 300
  }

  tags = {
    Name = "${var.project}-${var.environment}-apigw"
  }
}

# --- Stage (ambiente di deployment) ---
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = var.environment
  auto_deploy = true  # re-deploya automaticamente quando cambia la configurazione

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.apigw.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      sourceIp       = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      protocol       = "$context.protocol"
      httpMethod     = "$context.httpMethod"
      resourcePath   = "$context.resourcePath"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      responseLength = "$context.responseLength"
      integrationLatency = "$context.integrationLatency"
    })
  }
}

# --- Lambda Integration ---
resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.main.id
  integration_type   = "AWS_PROXY"  # proxy: API GW passa tutto il payload alla Lambda
  integration_uri    = var.lambda_invoke_arn
  integration_method = "POST"
  payload_format_version = "2.0"
}

# --- Routes (endpoint) ---
resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# --- Permesso: API Gateway può invocare la Lambda ---
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = var.lambda_function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

# --- CloudWatch Logs per API Gateway ---
resource "aws_cloudwatch_log_group" "apigw" {
  name              = "/aws/apigateway/${var.project}/${var.environment}"
  retention_in_days = 14
}

# --- Throttling (rate limiting) ---
# Previene abusi: max 100 req/sec con burst fino a 200
resource "aws_apigatewayv2_stage" "throttling" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "${var.environment}-throttled"
  auto_deploy = false

  default_route_settings {
    throttling_rate_limit  = 100
    throttling_burst_limit = 200
  }
}
