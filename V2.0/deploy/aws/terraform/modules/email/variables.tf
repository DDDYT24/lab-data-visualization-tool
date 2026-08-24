variable "name_prefix" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "identity_name" { type = string }
variable "notification_email" {
  type      = string
  sensitive = true
}
