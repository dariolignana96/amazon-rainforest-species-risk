# variables.tf
variable "project" { type = string }
variable "environment" { type = string }
variable "resource_group_name" { type = string }
variable "location" { type = string }
variable "container_image" { type = string }
variable "subnet_id" { type = string }

# outputs.tf
output "fqdn" { value = azurerm_container_group.api.fqdn }
output "ip_address" { value = azurerm_container_group.api.ip_address }
output "name" { value = azurerm_container_group.api.name }
