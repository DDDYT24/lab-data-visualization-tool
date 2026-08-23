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
