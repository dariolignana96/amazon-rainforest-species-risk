# modules/aws/api_gateway/variables.tf
variable "project" { type = string }
variable "environment" { type = string }
variable "lambda_invoke_arn" { type = string }
variable "lambda_function_name" { type = string }

# modules/aws/api_gateway/outputs.tf
output "api_id" { value = aws_apigatewayv2_api.main.id }
output "invoke_url" { value = aws_apigatewayv2_stage.default.invoke_url }
