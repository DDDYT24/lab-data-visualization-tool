resource "aws_ecr_repository" "logical_backup" {
  name                 = "${var.name_prefix}-${var.environment}-logical-backup"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.backup.arn
  }
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_cloudwatch_log_group" "logical_backup" {
  name              = "/labviz/${var.environment}/backup"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.backup.arn
}

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "logical_execution" {
  name                 = "${var.name_prefix}-${var.environment}-logical-backup-execution"
  assume_role_policy   = data.aws_iam_policy_document.ecs_tasks_assume.json
  permissions_boundary = var.permissions_boundary_arn
}

data "aws_iam_policy_document" "logical_execution" {
  statement {
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.logical_backup.arn]
  }
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.logical_backup.arn}:*"]
  }
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.postgres_url_secret_arn]
  }
  statement {
    actions   = ["kms:Decrypt"]
    resources = [var.secrets_kms_key_arn]
  }
}

resource "aws_iam_role_policy" "logical_execution" {
  name   = "pull-log-and-read-database-secret"
  role   = aws_iam_role.logical_execution.id
  policy = data.aws_iam_policy_document.logical_execution.json
}

resource "aws_iam_role" "logical_task" {
  name                 = "${var.name_prefix}-${var.environment}-logical-backup-task"
  assume_role_policy   = data.aws_iam_policy_document.ecs_tasks_assume.json
  permissions_boundary = var.permissions_boundary_arn
}

data "aws_iam_policy_document" "logical_task" {
  statement {
    actions   = ["s3:PutObject"]
    resources = ["${var.data_bucket_arn}/${var.logical_backup_prefix}/*"]
  }
  statement {
    actions   = ["kms:DescribeKey", "kms:Encrypt", "kms:GenerateDataKey"]
    resources = [var.object_kms_key_arn]
  }
}

resource "aws_iam_role_policy" "logical_task" {
  name   = "write-encrypted-logical-backups"
  role   = aws_iam_role.logical_task.id
  policy = data.aws_iam_policy_document.logical_task.json
}

resource "aws_ecs_task_definition" "logical_backup" {
  family                   = "${var.name_prefix}-${var.environment}-logical-backup"
  cpu                      = 256
  memory                   = 512
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.logical_execution.arn
  task_role_arn            = aws_iam_role.logical_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "tmp"
  }

  container_definitions = jsonencode([
    {
      name                   = "logical-backup"
      image                  = "${aws_ecr_repository.logical_backup.repository_url}@${var.backup_image_digest}"
      essential              = true
      readonlyRootFilesystem = true
      mountPoints = [{
        sourceVolume  = "tmp"
        containerPath = "/tmp"
        readOnly      = false
      }]
      environment = [
        { name = "LABVIZ_ENVIRONMENT", value = var.environment },
        { name = "LABVIZ_BACKUP_BUCKET", value = var.data_bucket_name },
        { name = "LABVIZ_BACKUP_PREFIX", value = var.logical_backup_prefix },
        { name = "LABVIZ_BACKUP_KMS_KEY_ARN", value = var.object_kms_key_arn },
      ]
      secrets = [{
        name      = "LABVIZ_POSTGRES_URL"
        valueFrom = var.postgres_url_secret_arn
      }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.logical_backup.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    },
  ])
}

data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name                 = "${var.name_prefix}-${var.environment}-logical-backup-scheduler"
  assume_role_policy   = data.aws_iam_policy_document.scheduler_assume.json
  permissions_boundary = var.permissions_boundary_arn
}

data "aws_iam_policy_document" "scheduler" {
  statement {
    actions   = ["ecs:RunTask"]
    resources = [aws_ecs_task_definition.logical_backup.arn]
    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [var.ecs_cluster_arn]
    }
  }
  statement {
    actions = ["iam:PassRole"]
    resources = [
      aws_iam_role.logical_execution.arn,
      aws_iam_role.logical_task.arn,
    ]
  }
}

resource "aws_iam_role_policy" "scheduler" {
  name   = "run-selected-logical-backup-task"
  role   = aws_iam_role.scheduler.id
  policy = data.aws_iam_policy_document.scheduler.json
}

resource "aws_scheduler_schedule" "logical_backup" {
  count = var.activate_logical_backup_schedule ? 1 : 0

  name                         = "${var.name_prefix}-${var.environment}-logical-backup"
  description                  = "Daily encrypted PostgreSQL logical backup"
  schedule_expression          = "cron(30 17 * * ? *)"
  schedule_expression_timezone = "UTC"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = var.ecs_cluster_arn
    role_arn = aws_iam_role.scheduler.arn
    retry_policy {
      maximum_event_age_in_seconds = 3600
      maximum_retry_attempts       = 2
    }
    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.logical_backup.arn
      task_count          = 1
      launch_type         = "FARGATE"
      platform_version    = "LATEST"
      network_configuration {
        subnets          = var.application_subnet_ids
        security_groups  = [var.backup_security_group_id]
        assign_public_ip = false
      }
    }
  }
}
