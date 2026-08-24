output "operations_topic_arn" {
  value = aws_sns_topic.operations.arn
}

output "dashboard_name" {
  value = aws_cloudwatch_dashboard.operations.dashboard_name
}

output "alarm_names" {
  value = concat(
    [for alarm in aws_cloudwatch_metric_alarm.alb_latency : alarm.alarm_name],
    [for alarm in aws_cloudwatch_metric_alarm.alb_unhealthy : alarm.alarm_name],
    [for alarm in aws_cloudwatch_metric_alarm.ecs_capacity : alarm.alarm_name],
    [for alarm in aws_cloudwatch_metric_alarm.rds : alarm.alarm_name],
    [for alarm in aws_cloudwatch_metric_alarm.application_events : alarm.alarm_name],
    [aws_cloudwatch_metric_alarm.alb_5xx_rate.alarm_name],
  )
}
