variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "aws_region" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "application_route_table_ids" {
  type = list(string)
}

variable "database_subnet_ids" {
  type = list(string)
}

variable "application_security_group_ids" {
  type = map(string)
}

variable "database_security_group_id" {
  type = string
}

variable "database_instance_class" {
  type = string
}

variable "database_multi_az" {
  type = bool
}

variable "database_deletion_protection" {
  type    = bool
  default = true
}

variable "database_backup_retention_days" {
  type = number

  validation {
    condition     = var.database_backup_retention_days >= 7
    error_message = "RDS PITR retention must be at least seven days."
  }
}

variable "object_prefix" {
  type    = string
  default = "labviz/"

  validation {
    condition     = endswith(var.object_prefix, "/") && !startswith(var.object_prefix, "/")
    error_message = "object_prefix must be relative and end with a slash."
  }
}
