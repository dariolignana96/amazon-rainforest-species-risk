# modules/aws/s3/variables.tf
variable "project" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }

# modules/aws/s3/outputs.tf
output "models_bucket_name" { value = aws_s3_bucket.models.bucket }
output "models_bucket_arn" { value = aws_s3_bucket.models.arn }
output "data_bucket_name" { value = aws_s3_bucket.data.bucket }
