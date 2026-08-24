variable "aws_region" {
  type    = string
  default = "ap-southeast-1"

  validation {
    condition     = var.aws_region == "ap-southeast-1"
    error_message = "Phase 6C is approved only for ap-southeast-1."
  }
}

variable "owner" {
  description = "Non-secret accountable environment owner."
  type        = string
}

variable "domain_name" {
  description = "Public staging FQDN, for example staging.example.com."
  type        = string
}

variable "route53_zone_id" {
  description = "Existing public Route 53 hosted-zone ID authoritative for domain_name."
  type        = string
}

variable "database_instance_class" {
  description = "Cost-controlled staging PostgreSQL instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "enable_waf" {
  description = "Enable the fail-closed regional WAF after reviewing recurring cost."
  type        = bool
  default     = false
}

variable "api_image_digest" {
  description = "Immutable API ECR image digest selected by the release workflow."
  type        = string
}

variable "web_image_digest" {
  description = "Immutable Web ECR image digest selected by the release workflow."
  type        = string
}

variable "ses_identity_name" {
  description = "Verified SES identity name; verification is an external Phase 6C gate."
  type        = string
}

variable "ses_from_address" {
  description = "Verified From address authorized for application mail."
  type        = string
}

variable "activate_services" {
  description = "Fail-closed switch set true only after images, secrets, migration, DNS, and cost approval are ready."
  type        = bool
  default     = false
}

variable "enable_worker_delete_permission" {
  description = "Explicit opt-in for worker S3 deletion after destructive-maintenance acceptance."
  type        = bool
  default     = false
}

variable "backup_image_digest" {
  description = "Immutable logical-backup image digest selected by the release workflow."
  type        = string
}

variable "notification_email" {
  description = "Operator mailbox for SES events and CloudWatch alarms; SNS confirmation is external."
  type        = string
  sensitive   = true
}

variable "activate_logical_backup_schedule" {
  description = "Enable only after backup image, secret, alarm, and cost gates pass."
  type        = bool
  default     = false
}

variable "enable_restore_testing" {
  description = "Enable quarterly AWS Backup restore testing after recurring cost approval."
  type        = bool
  default     = false
}

variable "permissions_boundary_arn" {
  description = "Staging runtime-role permissions boundary created by the bootstrap root."
  type        = string

  validation {
    condition     = can(regex(":policy/labviz-staging-runtime-boundary$", var.permissions_boundary_arn))
    error_message = "Staging must use the staging LabViz runtime permissions boundary."
  }
}

locals {
  name_prefix = "labviz"
  required_tags = {
    Environment = "staging"
    ManagedBy   = "Terraform"
    Owner       = var.owner
    Phase       = "6C"
    Project     = "LabViz"
  }
}
