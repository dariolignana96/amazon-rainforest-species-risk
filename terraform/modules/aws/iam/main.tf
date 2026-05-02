# =============================================================================
# MODULO: AWS IAM (Identity and Access Management)
# =============================================================================
# Principio del "Least Privilege": ogni servizio ottiene SOLO i permessi
# strettamente necessari. Non usare mai AdministratorAccess in produzione.
#
# Risorse create:
#   - Role per EC2 (legge S3 per caricare i modelli ML)
#   - Role per ECS Task (accede a S3 e CloudWatch Logs)
#   - Role per Lambda (accede a S3, CloudWatch, X-Ray)
# =============================================================================

# --- IAM Role per EC2 ---
# Permette all'istanza EC2 di assumere questo ruolo (trust policy)
resource "aws_iam_role" "ec2_role" {
  name = "${var.project}-${var.environment}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Name = "${var.project}-${var.environment}-ec2-role"
  }
}

# Policy custom: EC2 può leggere i modelli ML da S3
resource "aws_iam_role_policy" "ec2_s3_read" {
  name = "${var.project}-${var.environment}-ec2-s3-read"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${var.project}-${var.environment}-models",
        "arn:aws:s3:::${var.project}-${var.environment}-models/*"
      ]
    }]
  })
}

# Instance profile: collega il role all'istanza EC2
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project}-${var.environment}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# --- IAM Role per ECS Task ---
resource "aws_iam_role" "ecs_task_role" {
  name = "${var.project}-${var.environment}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# ECS Task può scrivere log su CloudWatch e leggere da S3
resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${var.project}-${var.environment}-ecs-task-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.project}-${var.environment}-models",
          "arn:aws:s3:::${var.project}-${var.environment}-models/*"
        ]
      }
    ]
  })
}

# Managed policy per ECS execution (pull immagini ECR, scrivere log)
resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- IAM Role per Lambda ---
resource "aws_iam_role" "lambda_role" {
  name = "${var.project}-${var.environment}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project}-${var.environment}-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "arn:aws:s3:::${var.project}-${var.environment}-models/*"
      },
      {
        # X-Ray tracing per performance monitoring
        Effect   = "Allow"
        Action   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
        Resource = "*"
      }
    ]
  })
}
