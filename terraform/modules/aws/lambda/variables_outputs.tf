# modules/aws/lambda/variables.tf
variable "project" { type = string }
variable "environment" { type = string }
variable "s3_bucket" { type = string }
variable "lambda_role_arn" { type = string }

# modules/aws/lambda/outputs.tf
output "lambda_function_name" { value = aws_lambda_function.predict.function_name }
output "lambda_invoke_arn" { value = aws_lambda_function.predict.invoke_arn }
output "lambda_arn" { value = aws_lambda_function.predict.arn }
