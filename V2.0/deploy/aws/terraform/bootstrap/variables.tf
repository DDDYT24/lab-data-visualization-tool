variable "aws_region" {
  description = "AWS Region selected for LabViz production data and services."
  type        = string
  default     = "ap-southeast-1"

  validation {
    condition     = var.aws_region == "ap-southeast-1"
    error_message = "Phase 6C is approved only for ap-southeast-1."
  }
}

variable "budget_owner" {
  description = "Non-secret owner label used on shared AWS resources."
  type        = string

  validation {
    condition     = length(trimspace(var.budget_owner)) > 0
    error_message = "budget_owner must identify the accountable cost owner."
  }
}

variable "github_repository" {
  description = "GitHub owner/repository allowed to request OIDC sessions."
  type        = string
  default     = "DDDYT24/lab-data-visualization-tool"

  validation {
    condition     = var.github_repository == "DDDYT24/lab-data-visualization-tool"
    error_message = "OIDC trust is pinned to the approved GitHub repository."
  }
}

locals {
  name_prefix = "labviz"
  required_tags = {
    Environment = "shared"
    ManagedBy   = "Terraform"
    Owner       = var.budget_owner
    Phase       = "6C"
    Project     = "LabViz"
  }
}
