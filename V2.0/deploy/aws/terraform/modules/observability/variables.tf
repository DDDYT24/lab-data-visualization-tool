variable "name_prefix" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "notification_email" {
  type      = string
  sensitive = true
}
variable "alb_arn_suffix" { type = string }
variable "api_target_group_arn_suffix" { type = string }
variable "web_target_group_arn_suffix" { type = string }
variable "ecs_cluster_name" { type = string }
variable "ecs_service_names" { type = set(string) }
variable "database_identifier" { type = string }
variable "data_bucket_name" { type = string }
variable "api_log_group_name" { type = string }
variable "worker_log_group_name" { type = string }
variable "backup_log_group_name" { type = string }
variable "backup_vault_name" { type = string }
variable "ses_configuration_set_name" { type = string }
