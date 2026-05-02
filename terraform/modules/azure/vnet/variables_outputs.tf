# variables.tf
variable "project"             { type = string }
variable "environment"         { type = string }
variable "resource_group_name" { type = string }
variable "location"            { type = string }
variable "vnet_address_space"  { type = string }

# outputs.tf
output "vnet_id"   { value = azurerm_virtual_network.main.id }
output "subnet_id" { value = azurerm_subnet.containers.id }
output "vnet_name" { value = azurerm_virtual_network.main.name }
