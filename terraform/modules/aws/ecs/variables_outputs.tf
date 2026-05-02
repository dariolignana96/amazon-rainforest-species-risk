variable "project" {
  type = string
}
variable "environment" {
  type = string
}
variable "vpc_id" {
  type = string
}
variable "private_subnets" {
  type = list(string)
}
variable "public_subnets" {
  type = list(string)
}
variable "alb_security_group_id" {
  type = string
}
variable "ecs_security_group_id" {
  type = string
}
variable "container_image" {
  type = string
}
variable "container_port" {
  type    = number
  default = 8000
}
variable "cpu" {
  type    = number
  default = 256
}
variable "memory" {
  type    = number
  default = 512
}
variable "task_role_arn" {
  type = string
}
output "cluster_name" {
  value = aws_ecs_cluster.main.name
}
output "service_name" {
  value = aws_ecs_service.api.name
}
output "alb_dns_name" {
  value = aws_lb.main.dns_name
}
output "alb_arn" {
  value = aws_lb.main.arn
}
