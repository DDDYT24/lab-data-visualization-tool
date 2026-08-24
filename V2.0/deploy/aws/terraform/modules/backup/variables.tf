variable "name_prefix" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "vpc_id" { type = string }
variable "application_subnet_ids" { type = list(string) }
variable "database_subnet_ids" { type = list(string) }
variable "backup_security_group_id" { type = string }
variable "ecs_cluster_arn" { type = string }
variable "data_bucket_name" { type = string }
variable "data_bucket_arn" { type = string }
variable "object_kms_key_arn" { type = string }
variable "database_arn" { type = string }
variable "postgres_url_secret_arn" { type = string }
variable "secrets_kms_key_arn" { type = string }
variable "permissions_boundary_arn" { type = string }

variable "backup_image_digest" {
  type = string
  validation {
    condition     = can(regex("^sha256:[0-9a-f]{64}$", var.backup_image_digest))
    error_message = "backup_image_digest must be an immutable sha256 digest."
  }
}

variable "activate_logical_backup_schedule" {
  type    = bool
  default = false
}

variable "enable_restore_testing" {
  type    = bool
  default = false
}

variable "logical_backup_prefix" {
  type    = string
  default = "backups/logical"
}
