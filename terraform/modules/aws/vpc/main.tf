# =============================================================================
# MODULO: AWS VPC
# =============================================================================
# Crea la rete isolata (Virtual Private Cloud) con:
#   - Subnet pubbliche  → EC2, ALB (Load Balancer)
#   - Subnet private    → ECS Tasks, Lambda (no accesso diretto da internet)
#   - Internet Gateway  → consente accesso internet alle subnet pubbliche
#   - NAT Gateway       → consente alle subnet private di fare richieste uscenti
#
# Pattern usato: "Hub and Spoke" - una VPC centrale con subnet in 2 AZ
# per alta disponibilità (se una AZ cade, l'altra serve il traffico).
# =============================================================================

# --- VPC principale ---
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true  # necessario per ECS service discovery
  enable_dns_support   = true

  tags = {
    Name = "${var.project}-${var.environment}-vpc"
  }
}

# --- Internet Gateway ---
# Collega la VPC a internet. Senza IGW nessuna risorsa è raggiungibile.
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project}-${var.environment}-igw"
  }
}

# --- Subnet pubbliche (una per AZ) ---
resource "aws_subnet" "public" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  # cidrsubnet("10.0.0.0/16", 8, 0) → "10.0.0.0/24"
  # cidrsubnet("10.0.0.0/16", 8, 1) → "10.0.1.0/24"
  availability_zone = var.availability_zones[count.index]

  map_public_ip_on_launch = true  # EC2 ottengono IP pubblico automaticamente

  tags = {
    Name = "${var.project}-${var.environment}-public-${count.index + 1}"
    Type = "public"
  }
}

# --- Subnet private (una per AZ) ---
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  # Offset +10 per non sovrapporre con le subnet pubbliche
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "${var.project}-${var.environment}-private-${count.index + 1}"
    Type = "private"
  }
}

# --- Elastic IP per NAT Gateway ---
# NAT Gateway richiede un IP pubblico statico dedicato
resource "aws_eip" "nat" {
  count  = 1  # Un solo NAT Gateway per risparmiare (non HA, ma ok per dev)
  domain = "vpc"

  tags = {
    Name = "${var.project}-${var.environment}-nat-eip"
  }
}

# --- NAT Gateway ---
# Permette alle risorse nelle subnet private (ECS) di scaricare immagini Docker,
# chiamare AWS APIs, etc. - senza essere raggiungibili da internet.
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id  # il NAT sta nella subnet pubblica

  tags = {
    Name = "${var.project}-${var.environment}-nat"
  }

  depends_on = [aws_internet_gateway.main]
}

# --- Route table pubblica ---
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project}-${var.environment}-rt-public"
  }
}

# --- Route table privata ---
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name = "${var.project}-${var.environment}-rt-private"
  }
}

# --- Associazioni route table → subnet ---
resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# --- Security Group: ALB (Load Balancer) ---
resource "aws_security_group" "alb" {
  name        = "${var.project}-${var.environment}-sg-alb"
  description = "Traffico HTTP/HTTPS in ingresso verso il Load Balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"  # -1 = tutti i protocolli
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project}-${var.environment}-sg-alb"
  }
}

# --- Security Group: ECS Tasks ---
resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project}-${var.environment}-sg-ecs"
  description = "Traffico dal ALB verso i container ECS"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]  # solo dal ALB, non da internet
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project}-${var.environment}-sg-ecs"
  }
}

# --- Security Group: EC2 ---
resource "aws_security_group" "ec2" {
  name        = "${var.project}-${var.environment}-sg-ec2"
  description = "SSH e porta API per istanza EC2 dev"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    # NOTA: In prod limitare a IP specifici o usare AWS SSM Session Manager
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project}-${var.environment}-sg-ec2"
  }
}
