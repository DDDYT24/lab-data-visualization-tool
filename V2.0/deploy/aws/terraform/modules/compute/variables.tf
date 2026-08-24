variable "name_prefix" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "domain_name" { type = string }
variable "application_subnet_ids" { type = list(string) }
variable "api_security_group_id" { type = string }
variable "web_security_group_id" { type = string }
variable "worker_security_group_id" { type = string }
variable "api_target_group_arn" { type = string }
variable "web_target_group_arn" { type = string }
variable "alb_arn_suffix" { type = string }
variable "api_target_group_arn_suffix" { type = string }
variable "web_target_group_arn_suffix" { type = string }
variable "trusted_proxy_cidrs" { type = list(string) }
variable "data_bucket_name" { type = string }
variable "data_bucket_arn" { type = string }
variable "object_prefix" { type = string }
variable "object_kms_key_arn" { type = string }
variable "ses_identity_name" { type = string }
variable "ses_from_address" { type = string }
variable "ses_configuration_set_name" { type = string }

variable "api_image_digest" {
  type = string
  validation {
    condition     = can(regex("^sha256:[0-9a-f]{64}$", var.api_image_digest))
    error_message = "api_image_digest must be an immutable sha256 digest."
  }
}

variable "web_image_digest" {
  type = string
  validation {
    condition     = can(regex("^sha256:[0-9a-f]{64}$", var.web_image_digest))
    error_message = "web_image_digest must be an immutable sha256 digest."
  }
}

variable "activate_services" {
  type    = bool
  default = false
}

variable "enable_worker_delete_permission" {
  type    = bool
  default = false
}

variable "log_retention_days" {
  type    = number
  default = 30
}
