resource "aws_cloudwatch_log_group" "workloads" {
  for_each = toset(["api", "migration", "web", "worker"])

  name              = "/labviz/${var.environment}/${each.key}"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.artifacts.arn
}

locals {
  log_options = {
    awslogs-region        = var.aws_region
    awslogs-stream-prefix = "ecs"
  }
  api_secrets = [
    {
      name      = "LABVIZ_POSTGRES_URL"
      valueFrom = aws_secretsmanager_secret.postgres_url.arn
    },
    {
      name      = "LABVIZ_SHARE_TOKEN_KEYS"
      valueFrom = aws_secretsmanager_secret.share_token_keys.arn
    },
    {
      name      = "LABVIZ_CLIENT_IDENTITY_KEY"
      valueFrom = aws_secretsmanager_secret.client_identity_key.arn
    },
  ]
  database_secret = [
    {
      name      = "LABVIZ_POSTGRES_URL"
      valueFrom = aws_secretsmanager_secret.postgres_url.arn
    },
  ]
  tmp_mount = [
    {
      sourceVolume  = "tmp"
      containerPath = "/tmp"
      readOnly      = false
    },
  ]
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.name_prefix}-${var.environment}-api"
  cpu                      = 512
  memory                   = 1024
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.execution["api"].arn
  task_role_arn            = aws_iam_role.task["api"].arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "tmp"
  }

  container_definitions = jsonencode([
    {
      name                   = "api"
      image                  = local.api_image
      essential              = true
      readonlyRootFilesystem = true
      environment            = local.api_environment
      secrets                = local.api_secrets
      mountPoints            = local.tmp_mount
      linuxParameters = {
        initProcessEnabled = true
      }
      portMappings = [
        {
          name          = "api"
          containerPort = 8000
          protocol      = "tcp"
        },
      ]
      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)\""]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = merge(local.log_options, {
          awslogs-group = aws_cloudwatch_log_group.workloads["api"].name
        })
      }
    },
  ])
}

resource "aws_ecs_task_definition" "web" {
  family                   = "${var.name_prefix}-${var.environment}-web"
  cpu                      = 256
  memory                   = 512
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.execution["web"].arn
  task_role_arn            = aws_iam_role.task["web"].arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "tmp"
  }

  container_definitions = jsonencode([
    {
      name                   = "web"
      image                  = local.web_image
      essential              = true
      readonlyRootFilesystem = true
      environment = [
        {
          name  = "NEXT_PUBLIC_API_BASE_URL"
          value = "${local.public_url}/api"
        },
        {
          name  = "NODE_ENV"
          value = "production"
        },
      ]
      mountPoints = local.tmp_mount
      linuxParameters = {
        initProcessEnabled = true
      }
      portMappings = [
        {
          name          = "web"
          containerPort = 3000
          protocol      = "tcp"
        },
      ]
      healthCheck = {
        command     = ["CMD-SHELL", "node -e \"fetch('http://127.0.0.1:3000/').then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1))\""]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = merge(local.log_options, {
          awslogs-group = aws_cloudwatch_log_group.workloads["web"].name
        })
      }
    },
  ])
}

resource "aws_ecs_task_definition" "migration" {
  family                   = "${var.name_prefix}-${var.environment}-migration"
  cpu                      = 256
  memory                   = 512
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.execution["migration"].arn
  task_role_arn            = aws_iam_role.task["migration"].arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "tmp"
  }

  container_definitions = jsonencode([
    {
      name                   = "migration"
      image                  = local.api_image
      essential              = true
      readonlyRootFilesystem = true
      command                = ["python", "-m", "alembic", "upgrade", "head"]
      environment = [
        {
          name  = "LABVIZ_ENVIRONMENT"
          value = "production"
        },
      ]
      secrets     = local.database_secret
      mountPoints = local.tmp_mount
      linuxParameters = {
        initProcessEnabled = true
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = merge(local.log_options, {
          awslogs-group = aws_cloudwatch_log_group.workloads["migration"].name
        })
      }
    },
  ])
}

resource "aws_ecs_task_definition" "worker" {
  for_each = local.worker_tasks

  family                   = "${var.name_prefix}-${var.environment}-worker-${each.key}"
  cpu                      = 256
  memory                   = 512
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.execution["worker"].arn
  task_role_arn            = aws_iam_role.task["worker"].arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "tmp"
  }

  container_definitions = jsonencode([
    {
      name                   = "worker-${each.key}"
      image                  = local.api_image
      essential              = true
      readonlyRootFilesystem = true
      command                = ["python", "-m", "labviz_api.workers.cli", each.key]
      environment            = local.worker_environment
      secrets                = local.database_secret
      mountPoints            = local.tmp_mount
      linuxParameters = {
        initProcessEnabled = true
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = merge(local.log_options, {
          awslogs-group = aws_cloudwatch_log_group.workloads["worker"].name
        })
      }
    },
  ])
}
