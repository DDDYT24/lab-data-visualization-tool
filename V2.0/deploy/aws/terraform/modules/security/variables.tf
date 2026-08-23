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

variable "vpc_cidr" {
  type = string
}

variable "application_subnet_ids" {
  type = list(string)
}

variable "api_port" {
  type    = number
  default = 8000
}

variable "web_port" {
  type    = number
  default = 3000
}

variable "database_port" {
  type    = number
  default = 5432
}
