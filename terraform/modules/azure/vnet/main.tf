# =============================================================================
# MODULO: Azure Virtual Network (VNet)
# =============================================================================
# VNet = equivalente Azure della AWS VPC.
# Differenze chiave rispetto a AWS:
#   - Subnet Azure NON hanno route table separata di default
#   - I Security Group si chiamano "Network Security Group" (NSG)
#   - Non esiste Internet Gateway separato: il routing pubblico è automatico
# =============================================================================

# --- Virtual Network ---
resource "azurerm_virtual_network" "main" {
  name                = "vnet-${var.project}-${var.environment}"
  resource_group_name = var.resource_group_name
  location            = var.location
  address_space       = [var.vnet_address_space]

  tags = {
    Name        = "vnet-${var.project}-${var.environment}"
    Environment = var.environment
  }
}

# --- Subnet per i container ---
resource "azurerm_subnet" "containers" {
  name                 = "snet-containers-${var.environment}"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_address_space, 8, 1)]

  # Delega la subnet ad Azure Container Instances
  delegation {
    name = "aci-delegation"
    service_delegation {
      name    = "Microsoft.ContainerInstance/containerGroups"
      actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
    }
  }
}

# --- Network Security Group ---
# Equivalente del Security Group AWS, ma associato alla subnet (non all'interfaccia)
resource "azurerm_network_security_group" "containers" {
  name                = "nsg-containers-${var.environment}"
  resource_group_name = var.resource_group_name
  location            = var.location

  # Regola: permetti traffico HTTP sulla porta 8000
  security_rule {
    name                       = "allow-api"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  # Regola: blocca tutto il resto in ingresso
  security_rule {
    name                       = "deny-all-inbound"
    priority                   = 4096
    direction                  = "Inbound"
    access                     = "Deny"
    protocol                   = "*"
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = {
    Environment = var.environment
  }
}

# Associa il NSG alla subnet
resource "azurerm_subnet_network_security_group_association" "containers" {
  subnet_id                 = azurerm_subnet.containers.id
  network_security_group_id = azurerm_network_security_group.containers.id
}
