# =============================================================================
# OUTPUT GLOBALI
# =============================================================================
# Gli output espongono valori calcolati da Terraform al termine dell'apply.
# Utilità:
#   - Leggibili da altri moduli o stack Terraform (remote_state)
#   - Visibili con: terraform output
#   - Utili per CI/CD (es. GitHub Actions legge l'endpoint API)
# =============================================================================

# --- AWS ---

output "aws_vpc_id" {
  description = "ID della VPC principale AWS"
  value       = module.aws_vpc.vpc_id
}

output "aws_ec2_public_ip" {
  description = "IP pubblico dell'istanza EC2 (dev/testing)"
  value       = module.aws_ec2.public_ip
}

output "aws_s3_models_bucket" {
  description = "Nome del bucket S3 che contiene i modelli ML serializzati"
  value       = module.aws_s3.models_bucket_name
}

output "aws_ecs_service_url" {
  description = "URL del Load Balancer davanti al cluster ECS"
  value       = module.aws_ecs.alb_dns_name
}

output "aws_api_gateway_url" {
  description = "Endpoint API Gateway (Lambda integration)"
  value       = module.aws_api_gateway.invoke_url
}

# --- Azure ---

output "azure_resource_group_name" {
  description = "Nome del Resource Group Azure"
  value       = module.azure_resource_group.name
}

output "azure_aci_fqdn" {
  description = "FQDN del Container Instance Azure (accesso diretto al container)"
  value       = module.azure_aci.fqdn
}

output "azure_aci_ip" {
  description = "IP pubblico del Container Instance Azure"
  value       = module.azure_aci.ip_address
}

# --- Summary ---

output "deployment_summary" {
  description = "Riepilogo endpoint di accesso all'applicazione"
  value = {
    aws_api_url   = module.aws_api_gateway.invoke_url
    aws_ecs_url   = "http://${module.aws_ecs.alb_dns_name}"
    aws_ec2_url   = "http://${module.aws_ec2.public_ip}:8000"
    azure_aci_url = "http://${module.azure_aci.fqdn}:8000"
  }
}
