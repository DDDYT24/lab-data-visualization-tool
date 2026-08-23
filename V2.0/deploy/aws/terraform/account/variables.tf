variable "aws_region" {
  type    = string
  default = "ap-southeast-1"

  validation {
    condition     = var.aws_region == "ap-southeast-1"
    error_message = "Phase 6C is approved only for ap-southeast-1."
  }
}

variable "budget_owner" {
  description = "Non-secret accountable cost owner label."
  type        = string

  validation {
    condition     = length(trimspace(var.budget_owner)) > 0
    error_message = "budget_owner is required."
  }
}

variable "budget_owner_email" {
  description = "Private notification destination supplied outside Git."
  type        = string
  sensitive   = true

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.budget_owner_email))
    error_message = "budget_owner_email must be a valid email address."
  }
}

variable "bootstrap_user_name" {
  description = "MFA-protected IAM user trusted to assume the break-glass role."
  type        = string
  default     = "labviz-bootstrap"
}

locals {
  name_prefix = "labviz"
  required_tags = {
    Environment = "account"
    ManagedBy   = "Terraform"
    Owner       = var.budget_owner
    Phase       = "6C"
    Project     = "LabViz"
  }
}
