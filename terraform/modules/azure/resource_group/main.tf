# =============================================================================
# MODULO: Azure Resource Group
# =============================================================================
# Il Resource Group è il contenitore logico di tutte le risorse Azure.
# Analogia AWS: non esiste un equivalente diretto, ma è simile a un "tag"
# che raggruppa tutte le risorse correlate a un progetto.
#
# Vantaggi:
#   - Eliminare il Resource Group elimina TUTTE le risorse dentro (cleanup facile)
#   - Billing separato per progetto/team
#   - Controllo accessi (RBAC) a livello di gruppo
# =============================================================================

resource "azurerm_resource_group" "main" {
  name     = "rg-${var.project}-${var.environment}"
  location = var.location

  tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = "dario-lignana"
  }
}
