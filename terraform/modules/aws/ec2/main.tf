# =============================================================================
# MODULO: AWS EC2
# =============================================================================
# Istanza EC2 per sviluppo/testing. FREE TIER: t2.micro (750h/mese per 12 mesi).
# Al boot, lo user_data script:
#   1. Installa Docker
#   2. Scarica i modelli ML da S3
#   3. Avvia il container FastAPI
#
# In produzione si userebbe ECS/EKS invece di EC2 diretto,
# ma EC2 è utile per debugging e SSH diretto.
# =============================================================================

# Recupera l'ultima AMI Amazon Linux 2023 (gratuita, ottimizzata per AWS)
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- EC2 Instance ---
resource "aws_instance" "api_server" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type  # t2.micro = FREE TIER
  subnet_id              = var.public_subnet_id
  vpc_security_group_ids = [var.security_group_id]
  iam_instance_profile   = var.ec2_instance_profile

  # Script eseguito al primo avvio dell'istanza (cloud-init)
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_bucket   = var.s3_bucket_name
    environment = var.environment
    aws_region  = var.aws_region
  }))

  # Root volume 20GB (free tier include 30GB EBS gp2)
  root_block_device {
    volume_type           = "gp3"
    volume_size           = 20
    delete_on_termination = true
    encrypted             = true
  }

  tags = {
    Name = "${var.project}-${var.environment}-api-server"
    Role = "api"
  }
}

# --- Elastic IP (IP statico) ---
# Senza EIP, l'IP cambia ogni volta che l'istanza si riavvia
resource "aws_eip" "api_server" {
  instance = aws_instance.api_server.id
  domain   = "vpc"

  tags = {
    Name = "${var.project}-${var.environment}-api-eip"
  }
}

# --- CloudWatch: Alarm CPU ---
# Notifica se la CPU supera il 80% per 5 minuti consecutivi
resource "aws_cloudwatch_metric_alarm" "ec2_cpu_high" {
  alarm_name          = "${var.project}-${var.environment}-ec2-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300  # 5 minuti
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "CPU EC2 sopra l'80% per 10 minuti"

  dimensions = {
    InstanceId = aws_instance.api_server.id
  }
}
