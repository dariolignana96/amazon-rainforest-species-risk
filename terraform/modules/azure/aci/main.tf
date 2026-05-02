# =============================================================================
# MODULO: Azure Container Instances (ACI)
# =============================================================================
# ACI = container serverless su Azure. Analogo a AWS Fargate ma più semplice:
# non serve un cluster ECS, lanci il container direttamente.
#
# Ideale per:
#   - Demo e portfolio (avvio rapido, stop altrettanto rapido)
#   - Workload a bassa frequenza (non ha senso pagare un cluster H24)
#   - Testing di immagini Docker prima di passare a Kubernetes
#
# FREE / LOW COST: primo 1 milione di secondi CPU e 1 milione GB/s gratis
# =============================================================================

resource "azurerm_container_group" "api" {
  name                = "aci-${var.project}-${var.environment}-api"
  location            = var.location
  resource_group_name = var.resource_group_name
  ip_address_type     = "Public"
  dns_name_label      = "${var.project}-${var.environment}-api"
  # FQDN risultante: rainforest-dev-api.westeurope.azurecontainer.io
  os_type             = "Linux"
  restart_policy      = "Always"

  container {
    name   = "rainforest-api"
    image  = var.container_image
    cpu    = "0.5"    # 0.5 vCPU
    memory = "0.5"    # 0.5 GB RAM

    ports {
      port     = 8000
      protocol = "TCP"
    }

    environment_variables = {
      ENVIRONMENT = var.environment
      PORT        = "8000"
    }

    # Liveness probe: Azure riavvia il container se non risponde
    liveness_probe {
      http_get {
        path   = "/health"
        port   = 8000
        scheme = "Http"
      }
      initial_delay_seconds = 30
      period_seconds        = 15
      failure_threshold     = 3
    }

    # Readiness probe: Azure non manda traffico finché non è pronto
    readiness_probe {
      http_get {
        path   = "/health"
        port   = 8000
        scheme = "Http"
      }
      initial_delay_seconds = 10
      period_seconds        = 10
    }
  }

  # Diagnostica: log del container su Azure Monitor
  diagnostics {
    log_analytics {
      workspace_id  = azurerm_log_analytics_workspace.main.workspace_id
      workspace_key = azurerm_log_analytics_workspace.main.primary_shared_key
    }
  }

  tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# --- Azure Log Analytics ---
# Equivalente di AWS CloudWatch: raccoglie log e metriche dai container
resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-${var.project}-${var.environment}"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = {
    Environment = var.environment
  }
}
