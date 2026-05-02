variable "project" {
  type = string
}
variable "environment" {
  type = string
}
output "ec2_role_arn" {
  value = aws_iam_role.ec2_role.arn
}
output "ec2_instance_profile" {
  value = aws_iam_instance_profile.ec2_profile.name
}
output "ecs_task_role_arn" {
  value = aws_iam_role.ecs_task_role.arn
}
output "lambda_role_arn" {
  value = aws_iam_role.lambda_role.arn
}
