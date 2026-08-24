data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  worker_tasks = toset([
    "metadata-cleanup",
    "orphan-staging-inventory",
    "pending-reconciliation",
    "project-lifecycle",
    "stored-object-gc",
  ])
  public_url = "https://${var.domain_name}"
  common_environment = [
    { name = "LABVIZ_ENVIRONMENT", value = "production" },
    { name = "LABVIZ_PERSISTENCE_BACKEND", value = "postgresql" },
    { name = "LABVIZ_OBJECT_STORAGE_BACKEND", value = "s3" },
    { name = "LABVIZ_S3_BUCKET", value = var.data_bucket_name },
    { name = "LABVIZ_S3_PREFIX", value = var.object_prefix },
    { name = "LABVIZ_S3_REGION", value = var.aws_region },
  ]
  api_environment = concat(local.common_environment, [
    { name = "LABVIZ_RUNTIME_ROLE", value = "api" },
    { name = "LABVIZ_ALLOWED_ORIGINS", value = local.public_url },
    { name = "LABVIZ_PUBLIC_WEB_URL", value = local.public_url },
    { name = "LABVIZ_AUTH_MODE", value = "ses" },
    { name = "LABVIZ_SES_REGION", value = var.aws_region },
    { name = "LABVIZ_SES_FROM", value = var.ses_from_address },
    { name = "LABVIZ_SES_CONFIGURATION_SET", value = var.ses_configuration_set_name },
    { name = "LABVIZ_COOKIE_SECURE", value = "true" },
    { name = "LABVIZ_SHARE_TOKEN_KEY_VERSION", value = "1" },
    { name = "LABVIZ_TRUSTED_PROXY_CIDRS", value = join(",", var.trusted_proxy_cidrs) },
    { name = "LABVIZ_TRUSTED_PROXY_HOPS", value = "1" },
    { name = "LABVIZ_AUTH_RATE_LIMIT_WINDOW_SECONDS", value = "3600" },
    { name = "LABVIZ_AUTH_CLIENT_REQUEST_LIMIT", value = "30" },
    { name = "LABVIZ_AUTH_EMAIL_REQUEST_LIMIT", value = "10" },
  ])
  worker_environment = concat(local.common_environment, [
    { name = "LABVIZ_RUNTIME_ROLE", value = "worker" },
    { name = "LABVIZ_WORKER_DESTRUCTIVE_MAINTENANCE", value = "false" },
    { name = "LABVIZ_WORKER_DRY_RUN", value = "true" },
    { name = "LABVIZ_WORKER_DELETE_ENABLED", value = "false" },
  ])
  api_image = "${aws_ecr_repository.api.repository_url}@${var.api_image_digest}"
  web_image = "${aws_ecr_repository.web.repository_url}@${var.web_image_digest}"
}

resource "aws_ecs_cluster" "this" {
  name = "${var.name_prefix}-${var.environment}"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "this" {
  cluster_name       = aws_ecs_cluster.this.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}
