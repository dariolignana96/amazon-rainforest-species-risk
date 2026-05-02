# variables.tf
variable "project"     { type = string }
variable "environment" { type = string }
variable "location"    { type = string }

# outputs.tf
output "name"     { value = azurerm_resource_group.main.name }
output "location" { value = azurerm_resource_group.main.location }
