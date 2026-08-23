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
  description = "Public production FQDN, for example app.example.com."
  type        = string
}

variable "route53_zone_id" {
  description = "Existing public Route 53 hosted-zone ID authoritative for domain_name."
  type        = string
}

variable "database_instance_class" {
  description = "Production PostgreSQL instance class."
  type        = string
  default     = "db.t4g.small"
}

variable "enable_waf" {
  description = "Production WAF remains fail-closed when attached."
  type        = bool
  default     = true
}

locals {
  name_prefix = "labviz"
  required_tags = {
    Environment = "production"
    ManagedBy   = "Terraform"
    Owner       = var.owner
    Phase       = "6C"
    Project     = "LabViz"
  }
}
