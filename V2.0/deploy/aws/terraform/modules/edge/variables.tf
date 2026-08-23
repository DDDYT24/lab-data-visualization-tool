variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_ids" {
  type = list(string)
}

variable "alb_security_group_id" {
  type = string
}

variable "domain_name" {
  description = "Public application FQDN covered by ACM and Route 53."
  type        = string

  validation {
    condition = (
      length(var.domain_name) <= 253
      && can(regex("^[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$", var.domain_name))
      && strcontains(var.domain_name, ".")
    )
    error_message = "domain_name must be a valid public FQDN."
  }
}

variable "route53_zone_id" {
  description = "Existing public Route 53 hosted-zone ID with delegated DNS authority."
  type        = string

  validation {
    condition     = length(trimspace(var.route53_zone_id)) > 0
    error_message = "route53_zone_id is required before 6C-1 apply."
  }
}

variable "api_port" {
  type    = number
  default = 8000
}

variable "web_port" {
  type    = number
  default = 3000
}

variable "enable_deletion_protection" {
  type    = bool
  default = true
}

variable "enable_waf" {
  type    = bool
  default = false
}
