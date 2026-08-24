output "vault_name" {
  value = aws_backup_vault.this.name
}

output "logical_backup_repository_url" {
  value = aws_ecr_repository.logical_backup.repository_url
}

output "logical_backup_log_group_name" {
  value = aws_cloudwatch_log_group.logical_backup.name
}

output "logical_backup_task_definition_arn" {
  value = aws_ecs_task_definition.logical_backup.arn
}
