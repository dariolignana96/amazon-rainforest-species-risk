# modules/aws/vpc/variables.tf

variable "project" {
  description = "Nome del progetto"
  type        = string
}

variable "environment" {
  description = "Ambiente (dev/staging/prod)"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block della VPC"
  type        = string
}

variable "availability_zones" {
  description = "Lista delle AZ da usare"
  type        = list(string)
}
