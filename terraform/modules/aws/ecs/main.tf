# =============================================================================
# MODULO: AWS ECS (Elastic Container Service) con Fargate
# =============================================================================
# ECS Fargate = container serverless: non devi gestire i server sottostanti.
# AWS si occupa del patching, scaling dell'infrastruttura.
#
# Componenti:
#   - Cluster ECS (raggruppamento logico di task)
#   - Task Definition (specifica del container: immagine, CPU, memoria, env vars)
#   - Service (mantiene N repliche del task attive, le riavvia se crashano)
#   - Application Load Balancer (distribuisce il traffico tra le repliche)
#   - CloudWatch Log Group (raccoglie tutti i log dei container)
# =============================================================================

# --- ECS Cluster ---
resource "aws_ecs_cluster" "main" {
  name = "${var.project}-${var.environment}-cluster"

  # Container Insights: metriche dettagliate su CPU/memoria per container
  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.project}-${var.environment}-cluster"
  }
}

# --- CloudWatch Log Group ---
# I log del container vanno qui (stdout/stderr del processo FastAPI)
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project}/${var.environment}"
  retention_in_days = 30 # Elimina log dopo 30 giorni (risparmio costi)

  tags = {
    Name = "${var.project}-${var.environment}-logs"
  }
}

# --- ECS Task Definition ---
# Definisce come far girare il container: immagine, risorse, variabili d'ambiente
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project}-${var.environment}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"   # obbligatorio per Fargate
  cpu                      = var.cpu    # 256 = 0.25 vCPU
  memory                   = var.memory # 512 MB
  task_role_arn            = var.task_role_arn
  execution_role_arn       = var.task_role_arn

  container_definitions = jsonencode([{
    name  = "rainforest-api"
    image = var.container_image

    portMappings = [{
      containerPort = var.container_port
      protocol      = "tcp"
    }]

    environment = [
      { name = "ENVIRONMENT", value = var.environment },
      { name = "PORT", value = tostring(var.container_port) }
    ]

    # Health check: ECS riavvia il container se l'health check fallisce 3 volte
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:${var.container_port}/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60 # dà 60 secondi al container per avviarsi prima dei check
    }

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
        "awslogs-region"        = "eu-west-1"
        "awslogs-stream-prefix" = "api"
      }
    }
  }])
}

# --- Application Load Balancer ---
resource "aws_lb" "main" {
  name               = "${var.project}-${var.environment}-alb"
  internal           = false # pubblico (accessibile da internet)
  load_balancer_type = "application"
  security_groups    = [var.alb_security_group_id]
  subnets            = var.public_subnets

  enable_deletion_protection = false # in prod mettere true

  tags = {
    Name = "${var.project}-${var.environment}-alb"
  }
}

# Target Group: pool di container ECS verso cui l'ALB instrada le richieste
resource "aws_lb_target_group" "api" {
  name        = "${var.project}-${var.environment}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip" # Fargate usa IP, non instance ID

  health_check {
    enabled             = true
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
  }
}

# Listener HTTP sulla porta 80
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# --- ECS Service ---
# Mantiene 2 repliche del container attive (alta disponibilità)
resource "aws_ecs_service" "api" {
  name            = "${var.project}-${var.environment}-api-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 2 # 2 repliche in 2 AZ diverse = HA
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnets # container nelle subnet private
    security_groups  = [var.ecs_security_group_id]
    assign_public_ip = false # accesso solo via ALB, non diretto
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "rainforest-api"
    container_port   = var.container_port
  }

  # Rolling deployment: aggiorna i container senza downtime
  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  depends_on = [aws_lb_listener.http]
}

# --- Auto Scaling ---
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = 4
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# Scale out se CPU > 70% per 2 minuti
resource "aws_appautoscaling_policy" "cpu_scale_out" {
  name               = "${var.project}-${var.environment}-cpu-scale-out"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
