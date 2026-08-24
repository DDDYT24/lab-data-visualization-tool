data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  alarm_prefix  = "${var.name_prefix}-${var.environment}"
  alarm_actions = [aws_sns_topic.operations.arn]
  target_groups = {
    api = var.api_target_group_arn_suffix
    web = var.web_target_group_arn_suffix
  }
}

resource "aws_sns_topic" "operations" {
  name              = "${local.alarm_prefix}-operations"
  kms_master_key_id = "alias/aws/sns"
}

resource "aws_sns_topic_subscription" "operations_email" {
  topic_arn = aws_sns_topic.operations.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

data "aws_iam_policy_document" "operations" {
  statement {
    sid       = "AllowCloudWatchAlarms"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.operations.arn]
    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }

  statement {
    sid       = "AllowEventBridgeAccessAlerts"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.operations.arn]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_sns_topic_policy" "operations" {
  arn    = aws_sns_topic.operations.arn
  policy = data.aws_iam_policy_document.operations.json
}

resource "aws_cloudwatch_metric_alarm" "alb_latency" {
  for_each = local.target_groups

  alarm_name          = "${local.alarm_prefix}-${each.key}-p95-latency"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  threshold           = 2
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  dimensions = {
    LoadBalancer = var.alb_arn_suffix
    TargetGroup  = each.value
  }
}

resource "aws_cloudwatch_metric_alarm" "alb_unhealthy" {
  for_each = local.target_groups

  alarm_name          = "${local.alarm_prefix}-${each.key}-unhealthy"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "breaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  dimensions = {
    LoadBalancer = var.alb_arn_suffix
    TargetGroup  = each.value
  }
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx_rate" {
  alarm_name          = "${local.alarm_prefix}-alb-5xx-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  threshold           = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  metric_query {
    id          = "error_rate"
    expression  = "100 * errors / FILL(requests, 1)"
    label       = "ALB 5xx percent"
    return_data = true
  }
  metric_query {
    id          = "errors"
    return_data = false
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_ELB_5XX_Count"
      period      = 300
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
      }
    }
  }
  metric_query {
    id          = "requests"
    return_data = false
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "RequestCount"
      period      = 300
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
      }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "alb_rejected" {
  alarm_name          = "${local.alarm_prefix}-alb-rejected-connections"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "RejectedConnectionCount"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  dimensions = {
    LoadBalancer = var.alb_arn_suffix
  }
}

resource "aws_cloudwatch_metric_alarm" "ecs_capacity" {
  for_each = var.ecs_service_names

  alarm_name          = "${local.alarm_prefix}-${each.key}-running-below-desired"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  threshold           = 0
  treat_missing_data  = "breaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  metric_query {
    id          = "gap"
    expression  = "desired - running"
    label       = "Desired minus running tasks"
    return_data = true
  }
  metric_query {
    id          = "desired"
    return_data = false
    metric {
      namespace   = "ECS/ContainerInsights"
      metric_name = "DesiredTaskCount"
      period      = 60
      stat        = "Maximum"
      dimensions = {
        ClusterName = var.ecs_cluster_name
        ServiceName = each.key
      }
    }
  }
  metric_query {
    id          = "running"
    return_data = false
    metric {
      namespace   = "ECS/ContainerInsights"
      metric_name = "RunningTaskCount"
      period      = 60
      stat        = "Minimum"
      dimensions = {
        ClusterName = var.ecs_cluster_name
        ServiceName = each.key
      }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "ecs_cpu" {
  for_each = var.ecs_service_names

  alarm_name          = "${local.alarm_prefix}-${each.key}-cpu-high"
  namespace           = "AWS/ECS"
  metric_name         = "CPUUtilization"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  threshold           = 85
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  dimensions = {
    ClusterName = var.ecs_cluster_name
    ServiceName = each.key
  }
}

resource "aws_cloudwatch_metric_alarm" "ecs_memory" {
  for_each = var.ecs_service_names

  alarm_name          = "${local.alarm_prefix}-${each.key}-memory-high"
  namespace           = "AWS/ECS"
  metric_name         = "MemoryUtilization"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  threshold           = 85
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  dimensions = {
    ClusterName = var.ecs_cluster_name
    ServiceName = each.key
  }
}

locals {
  rds_alarms = {
    cpu = {
      metric    = "CPUUtilization"
      statistic = "Average"
      threshold = 85
      operator  = "GreaterThanThreshold"
    }
    connections = {
      metric    = "DatabaseConnections"
      statistic = "Maximum"
      threshold = 80
      operator  = "GreaterThanThreshold"
    }
    free_storage = {
      metric    = "FreeStorageSpace"
      statistic = "Minimum"
      threshold = 5368709120
      operator  = "LessThanThreshold"
    }
    read_latency = {
      metric    = "ReadLatency"
      statistic = "Average"
      threshold = 0.1
      operator  = "GreaterThanThreshold"
    }
    write_latency = {
      metric    = "WriteLatency"
      statistic = "Average"
      threshold = 0.1
      operator  = "GreaterThanThreshold"
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "rds" {
  for_each = local.rds_alarms

  alarm_name          = "${local.alarm_prefix}-rds-${each.key}"
  namespace           = "AWS/RDS"
  metric_name         = each.value.metric
  statistic           = each.value.statistic
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  threshold           = each.value.threshold
  comparison_operator = each.value.operator
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
  dimensions = {
    DBInstanceIdentifier = var.database_identifier
  }
}

resource "aws_cloudwatch_metric_alarm" "s3_errors" {
  for_each = toset(["4xxErrors", "5xxErrors"])

  alarm_name          = "${local.alarm_prefix}-s3-${lower(each.key)}"
  namespace           = "AWS/S3"
  metric_name         = each.key
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  dimensions = {
    BucketName = var.data_bucket_name
    FilterId   = "EntireBucket"
  }
}

resource "aws_cloudwatch_metric_alarm" "backup_failed" {
  alarm_name          = "${local.alarm_prefix}-backup-failed"
  namespace           = "AWS/Backup"
  metric_name         = "NumberOfBackupJobsFailed"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  dimensions = {
    BackupVaultName = var.backup_vault_name
  }
}

resource "aws_cloudwatch_metric_alarm" "ses_events" {
  for_each = toset(["Bounce", "Complaint", "DeliveryDelay", "Reject"])

  alarm_name          = "${local.alarm_prefix}-ses-${lower(each.key)}"
  namespace           = "AWS/SES"
  metric_name         = each.key
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  dimensions = {
    environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "ses_reputation" {
  for_each = {
    bounce = {
      metric    = "Reputation.BounceRate"
      threshold = 0.02
    }
    complaint = {
      metric    = "Reputation.ComplaintRate"
      threshold = 0.001
    }
  }

  alarm_name          = "${local.alarm_prefix}-ses-reputation-${each.key}"
  namespace           = "AWS/SES"
  metric_name         = each.value.metric
  statistic           = "Average"
  period              = 900
  evaluation_periods  = 1
  threshold           = each.value.threshold
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  dimensions = {
    ConfigurationSet = var.ses_configuration_set_name
  }
}

resource "aws_cloudwatch_log_metric_filter" "worker_failures" {
  for_each = {
    failure    = { event = "worker-item-failed", metric = "WorkerFailure" }
    quarantine = { event = "worker-item-quarantined", metric = "WorkerQuarantine" }
    lease_loss = { event = "worker-lease-lost", metric = "WorkerLeaseLoss" }
    heartbeat  = { event = "worker-heartbeat-failed", metric = "WorkerHeartbeat" }
  }

  name           = "${local.alarm_prefix}-worker-${each.key}"
  log_group_name = var.worker_log_group_name
  pattern        = "{ $.event = \"${each.value.event}\" }"
  metric_transformation {
    name      = each.value.metric
    namespace = "LabViz/${var.environment}"
    value     = "1"
  }
}

resource "aws_cloudwatch_log_metric_filter" "api_events" {
  for_each = {
    delivery_failure = { event = "authentication-code-delivery-failed", metric = "ApiDeliveryFailure" }
    rate_limit       = { event = "authentication-request-rate-limited", metric = "ApiRateLimit" }
  }

  name           = "${local.alarm_prefix}-api-${each.key}"
  log_group_name = var.api_log_group_name
  pattern        = "\"${each.value.event}\""
  metric_transformation {
    name      = each.value.metric
    namespace = "LabViz/${var.environment}"
    value     = "1"
  }
}

resource "aws_cloudwatch_log_metric_filter" "logical_backup_failure" {
  name           = "${local.alarm_prefix}-logical-backup-failed"
  log_group_name = var.backup_log_group_name
  pattern        = "\"logical-backup-failed\""
  metric_transformation {
    name      = "LogicalBackupFailure"
    namespace = "LabViz/${var.environment}"
    value     = "1"
  }
}

resource "aws_cloudwatch_metric_alarm" "application_events" {
  for_each = {
    api_delivery_failure = "ApiDeliveryFailure"
    api_rate_limit       = "ApiRateLimit"
    logical_backup       = "LogicalBackupFailure"
    worker_failure       = "WorkerFailure"
    worker_quarantine    = "WorkerQuarantine"
    worker_lease_loss    = "WorkerLeaseLoss"
    worker_heartbeat     = "WorkerHeartbeat"
  }

  alarm_name          = "${local.alarm_prefix}-${each.key}"
  namespace           = "LabViz/${var.environment}"
  metric_name         = each.value
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_event_rule" "storage_access_denied" {
  name        = "${local.alarm_prefix}-storage-access-denied"
  description = "Alert on S3 or KMS API access denials observed by CloudTrail/EventBridge"
  event_pattern = jsonencode({
    source        = ["aws.s3", "aws.kms"]
    "detail-type" = ["AWS API Call via CloudTrail"]
    detail = {
      errorCode = [{ prefix = "AccessDenied" }]
    }
  })
}

resource "aws_cloudwatch_event_target" "storage_access_denied" {
  rule = aws_cloudwatch_event_rule.storage_access_denied.name
  arn  = aws_sns_topic.operations.arn
}

resource "aws_cloudwatch_dashboard" "operations" {
  dashboard_name = "${local.alarm_prefix}-operations"
  dashboard_body = jsonencode({
    start          = "-PT6H"
    periodOverride = "inherit"
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 2
        properties = {
          markdown = "# LabViz ${var.environment}\nSLO: availability >=99.9%, RPO <=15m, RTO <=4h."
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 2
        width  = 12
        height = 6
        properties = {
          title  = "ALB requests, errors, and latency"
          region = var.aws_region
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", var.alb_arn_suffix, { stat = "Sum" }],
            [".", "HTTPCode_ELB_5XX_Count", ".", ".", { stat = "Sum" }],
            [".", "TargetResponseTime", ".", ".", { stat = "p95", yAxis = "right" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 2
        width  = 12
        height = 6
        properties = {
          title  = "RDS health"
          region = var.aws_region
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", var.database_identifier],
            [".", "DatabaseConnections", ".", "."],
            [".", "FreeStorageSpace", ".", ".", { yAxis = "right" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 8
        width  = 24
        height = 6
        properties = {
          title  = "Application, worker, SES, and backup failure signals"
          region = var.aws_region
          metrics = [
            ["LabViz/${var.environment}", "WorkerFailure", { stat = "Sum" }],
            [".", "WorkerQuarantine", { stat = "Sum" }],
            [".", "ApiDeliveryFailure", { stat = "Sum" }],
            [".", "LogicalBackupFailure", { stat = "Sum" }],
            ["AWS/Backup", "NumberOfBackupJobsFailed", "BackupVaultName", var.backup_vault_name, { stat = "Sum" }],
            ["AWS/SES", "Bounce", "environment", var.environment, { stat = "Sum" }],
            [".", "Complaint", ".", ".", { stat = "Sum" }],
          ]
        }
      },
    ]
  })
}
