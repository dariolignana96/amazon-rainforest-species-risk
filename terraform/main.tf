# =============================================================================
# AMAZON RAINFOREST SPECIES RISK - Infrastruttura Multi-Cloud
# =============================================================================
# Autore: Dario Lignana
# Progetto: https://github.com/dariolignana96/amazon-rainforest-species-risk
#
# ATTENZIONE: Questo file è pensato per terraform plan/validate ONLY.
# NON eseguire terraform apply senza aver verificato i costi.
#
# Architettura:
#   AWS  → VPC + EC2 + S3 + IAM + ECS + Lambda + API Gateway
#   Azure → Resource Group + VNet + ACI (Container Instance)
#
# Il design segue il principio "Separation of Concerns":
#   ogni modulo gestisce una sola responsabilità infrastrutturale.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }

  # Backend S3 (commentato: attivare solo se si ha bucket reale)
  # backend "s3" {
  #   bucket = "rainforest-tfstate"
  #   key    = "prod/terraform.tfstate"
  #   region = "eu-west-1"
  # }
}

# =============================================================================
# PROVIDER AWS
# =============================================================================
provider "aws" {
  region = var.aws_region

  # Tag applicati globalmente a tutte le risorse AWS create da Terraform
  default_tags {
    tags = {
      Project     = "amazon-rainforest-species-risk"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "dario-lignana"
    }
  }
}

# =============================================================================
# PROVIDER AZURE
# =============================================================================
provider "azurerm" {
  features {}
  # Credenziali lette da env vars: ARM_SUBSCRIPTION_ID, ARM_CLIENT_ID, etc.
  # oppure da az login (Azure CLI)
}

# =============================================================================
# MODULI AWS
# =============================================================================

module "aws_iam" {
  source      = "./modules/aws/iam"
  environment = var.environment
  project     = var.project_name
}

module "aws_s3" {
  source      = "./modules/aws/s3"
  environment = var.environment
  project     = var.project_name
  aws_region  = var.aws_region
}

module "aws_vpc" {
  source             = "./modules/aws/vpc"
  environment        = var.environment
  project            = var.project_name
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
}

module "aws_ec2" {
  source               = "./modules/aws/ec2"
  environment          = var.environment
  project              = var.project_name
  vpc_id               = module.aws_vpc.vpc_id
  public_subnet_id     = module.aws_vpc.public_subnet_ids[0]
  security_group_id    = module.aws_vpc.ec2_security_group_id
  instance_type        = var.ec2_instance_type
  ec2_instance_profile = module.aws_iam.ec2_instance_profile
  s3_bucket_name       = module.aws_s3.models_bucket_name
  aws_region           = var.aws_region
}

module "aws_ecs" {
  source                = "./modules/aws/ecs"
  environment           = var.environment
  project               = var.project_name
  vpc_id                = module.aws_vpc.vpc_id
  private_subnets       = module.aws_vpc.private_subnet_ids
  public_subnets        = module.aws_vpc.public_subnet_ids
  alb_security_group_id = module.aws_vpc.alb_security_group_id
  ecs_security_group_id = module.aws_vpc.ecs_security_group_id
  container_image       = var.container_image
  container_port        = 8000
  cpu                   = 256
  memory                = 512
  task_role_arn         = module.aws_iam.ecs_task_role_arn
}

module "aws_lambda" {
  source        = "./modules/aws/lambda"
  environment   = var.environment
  project       = var.project_name
  s3_bucket     = module.aws_s3.models_bucket_name
  lambda_role_arn = module.aws_iam.lambda_role_arn
}

module "aws_api_gateway" {
  source          = "./modules/aws/api_gateway"
  environment     = var.environment
  project         = var.project_name
  lambda_invoke_arn = module.aws_lambda.lambda_invoke_arn
  lambda_function_name = module.aws_lambda.lambda_function_name
}

# =============================================================================
# MODULI AZURE
# =============================================================================

module "azure_resource_group" {
  source      = "./modules/azure/resource_group"
  environment = var.environment
  project     = var.project_name
  location    = var.azure_location
}

module "azure_vnet" {
  source              = "./modules/azure/vnet"
  environment         = var.environment
  project             = var.project_name
  resource_group_name = module.azure_resource_group.name
  location            = var.azure_location
  vnet_address_space  = var.azure_vnet_cidr
}

module "azure_aci" {
  source              = "./modules/azure/aci"
  environment         = var.environment
  project             = var.project_name
  resource_group_name = module.azure_resource_group.name
  location            = var.azure_location
  container_image     = var.container_image
  subnet_id           = module.azure_vnet.subnet_id
}
