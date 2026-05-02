variable "project" {
  type = string
}
variable "environment" {
  type = string
}
variable "vpc_id" {
  type = string
}
variable "public_subnet_id" {
  type = string
}
variable "security_group_id" {
  type = string
}
variable "instance_type" {
  type    = string
  default = "t2.micro"
}
variable "ec2_instance_profile" {
  type = string
}
variable "s3_bucket_name" {
  type = string
}
variable "aws_region" {
  type    = string
  default = "eu-west-1"
}
output "instance_id" {
  value = aws_instance.api_server.id
}
output "public_ip" {
  value = aws_eip.api_server.public_ip
}
output "private_ip" {
  value = aws_instance.api_server.private_ip
}
