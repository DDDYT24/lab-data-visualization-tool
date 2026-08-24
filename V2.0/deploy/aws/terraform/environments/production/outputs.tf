output "application_url" {
  value = module.edge.application_url
}

output "alb_dns_name" {
  value = module.edge.alb_dns_name
}

output "trusted_proxy_cidrs" {
  description = "Set LABVIZ_TRUSTED_PROXY_CIDRS to this exact ALB subnet list."
  value       = module.network.public_subnet_cidrs
}

output "data_bucket_name" {
  value = module.data.bucket_name
}

output "database_master_secret_arn" {
  value     = module.data.database_master_secret_arn
  sensitive = true
}

output "ecs_cluster_name" {
  value = module.compute.cluster_name
}

output "api_repository_url" {
  value = module.compute.api_repository_url
}

output "web_repository_url" {
  value = module.compute.web_repository_url
}

output "application_secret_arns" {
  value = {
    postgres_url        = module.compute.postgres_url_secret_arn
    share_token_keys    = module.compute.share_token_keys_secret_arn
    client_identity_key = module.compute.client_identity_key_secret_arn
  }
  sensitive = true
}

output "ses_dkim_tokens" {
  value = module.email.dkim_tokens
}

output "operations_dashboard_name" {
  value = module.observability.dashboard_name
}

output "logical_backup_repository_url" {
  value = module.backup.logical_backup_repository_url
}
