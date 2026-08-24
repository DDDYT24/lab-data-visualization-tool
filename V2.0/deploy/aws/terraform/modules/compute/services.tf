resource "aws_cloudwatch_metric_alarm" "unhealthy_targets" {
  for_each = var.activate_services ? {
    api = var.api_target_group_arn_suffix
    web = var.web_target_group_arn_suffix
  } : {}

  alarm_name          = "${var.name_prefix}-${var.environment}-${each.key}-unhealthy-targets"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "breaching"

  dimensions = {
    LoadBalancer = var.alb_arn_suffix
    TargetGroup  = each.value
  }
}

resource "aws_ecs_service" "api" {
  name             = "api"
  cluster          = aws_ecs_cluster.this.id
  task_definition  = aws_ecs_task_definition.api.arn
  desired_count    = var.activate_services ? 1 : 0
  launch_type      = "FARGATE"
  platform_version = "LATEST"

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  dynamic "alarms" {
    for_each = var.activate_services ? [1] : []
    content {
      alarm_names = [aws_cloudwatch_metric_alarm.unhealthy_targets["api"].alarm_name]
      enable      = true
      rollback    = true
    }
  }

  network_configuration {
    subnets          = var.application_subnet_ids
    security_groups  = [var.api_security_group_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.api_target_group_arn
    container_name   = "api"
    container_port   = 8000
  }

  health_check_grace_period_seconds = 60
  enable_ecs_managed_tags           = true
  propagate_tags                    = "SERVICE"

  depends_on = [aws_ecs_cluster_capacity_providers.this]
}

resource "aws_ecs_service" "web" {
  name             = "web"
  cluster          = aws_ecs_cluster.this.id
  task_definition  = aws_ecs_task_definition.web.arn
  desired_count    = var.activate_services ? 1 : 0
  launch_type      = "FARGATE"
  platform_version = "LATEST"

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  dynamic "alarms" {
    for_each = var.activate_services ? [1] : []
    content {
      alarm_names = [aws_cloudwatch_metric_alarm.unhealthy_targets["web"].alarm_name]
      enable      = true
      rollback    = true
    }
  }

  network_configuration {
    subnets          = var.application_subnet_ids
    security_groups  = [var.web_security_group_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.web_target_group_arn
    container_name   = "web"
    container_port   = 3000
  }

  health_check_grace_period_seconds = 60
  enable_ecs_managed_tags           = true
  propagate_tags                    = "SERVICE"

  depends_on = [aws_ecs_cluster_capacity_providers.this]
}

resource "aws_ecs_service" "worker" {
  for_each = local.worker_tasks

  name             = "worker-${each.key}"
  cluster          = aws_ecs_cluster.this.id
  task_definition  = aws_ecs_task_definition.worker[each.key].arn
  desired_count    = var.activate_services ? 1 : 0
  launch_type      = "FARGATE"
  platform_version = "LATEST"

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = var.application_subnet_ids
    security_groups  = [var.worker_security_group_id]
    assign_public_ip = false
  }

  enable_ecs_managed_tags = true
  propagate_tags          = "SERVICE"

  depends_on = [aws_ecs_cluster_capacity_providers.this]
}
