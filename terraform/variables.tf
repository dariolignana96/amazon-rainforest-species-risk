# =============================================================================
# VARIABILI GLOBALI
# =============================================================================
# Centralizzare le variabili qui permette di riusare lo stesso codice
# in ambienti diversi (dev/prod) cambiando solo i valori nel .tfvars.
# Questo è il pattern raccomandato da HashiCorp per progetti enterprise.
# =============================================================================

# --- Generali ---

variable "project_name" {
  description = "Nome del progetto, usato come prefisso per tutte le risorse"
  type        = string
  default     = "rainforest"

  validation {
    condition     = length(var.project_name) <= 16
    error_message = "Il nome progetto non può superare 16 caratteri (limite naming S3/IAM)."
  }
}

variable "environment" {
  description = "Ambiente di deployment: dev, staging, prod"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "L'ambiente deve essere uno tra: dev, staging, prod."
  }
}

variable "container_image" {
  description = "Immagine Docker da deployare su ECS e ACI"
  type        = string
  default     = "dariolignana96/rainforest-api:latest"
  # In prod: usare un tag immutabile (es. sha digest) invece di :latest
}

# --- AWS ---

variable "aws_region" {
  description = "Regione AWS principale"
  type        = string
  default     = "eu-west-1"  # Irlanda - più vicina all'Italia
}

variable "vpc_cidr" {
  description = "CIDR block per la VPC AWS"
  type        = string
  default     = "10.0.0.0/16"
  # /16 = 65.536 indirizzi IP disponibili
}

variable "availability_zones" {
  description = "AZ da utilizzare (alta disponibilità con almeno 2)"
  type        = list(string)
  default     = ["eu-west-1a", "eu-west-1b"]
}

variable "ec2_instance_type" {
  description = "Tipo istanza EC2. t2.micro è FREE TIER (750h/mese per 12 mesi)"
  type        = string
  default     = "t2.micro"
  # ATTENZIONE: cambiare in t3.small o superiore genera costi!
}

# --- Azure ---

variable "azure_location" {
  description = "Regione Azure principale"
  type        = string
  default     = "West Europe"  # Amsterdam - più vicina all'Italia
}

variable "azure_vnet_cidr" {
  description = "CIDR block per la VNet Azure"
  type        = string
  default     = "10.1.0.0/16"
  # Range diverso da AWS per evitare conflitti in caso di VPN/peering futuro
}
