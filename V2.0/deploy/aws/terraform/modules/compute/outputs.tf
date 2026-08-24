output "cluster_arn" { value = aws_ecs_cluster.this.arn }
output "cluster_name" { value = aws_ecs_cluster.this.name }
output "api_repository_url" { value = aws_ecr_repository.api.repository_url }
output "web_repository_url" { value = aws_ecr_repository.web.repository_url }
output "api_service_name" { value = aws_ecs_service.api.name }
output "web_service_name" { value = aws_ecs_service.web.name }
output "worker_service_names" { value = { for task, service in aws_ecs_service.worker : task => service.name } }
output "migration_task_definition_arn" { value = aws_ecs_task_definition.migration.arn }
output "postgres_url_secret_arn" { value = aws_secretsmanager_secret.postgres_url.arn }
output "share_token_keys_secret_arn" { value = aws_secretsmanager_secret.share_token_keys.arn }
output "client_identity_key_secret_arn" { value = aws_secretsmanager_secret.client_identity_key.arn }
output "log_group_names" { value = { for name, group in aws_cloudwatch_log_group.workloads : name => group.name } }
