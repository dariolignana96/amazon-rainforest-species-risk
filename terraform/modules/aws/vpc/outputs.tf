# modules/aws/vpc/outputs.tf

output "vpc_id" {
  description = "ID della VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Lista degli ID delle subnet pubbliche"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Lista degli ID delle subnet private"
  value       = aws_subnet.private[*].id
}

output "alb_security_group_id" {
  value = aws_security_group.alb.id
}

output "ecs_security_group_id" {
  value = aws_security_group.ecs_tasks.id
}

output "ec2_security_group_id" {
  value = aws_security_group.ec2.id
}
