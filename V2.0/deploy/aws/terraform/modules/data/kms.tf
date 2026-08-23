data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

resource "aws_kms_key" "objects" {
  description             = "${var.name_prefix} ${var.environment} object storage"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "objects" {
  name          = "alias/${var.name_prefix}-${var.environment}-objects"
  target_key_id = aws_kms_key.objects.key_id
}

resource "aws_kms_key" "database" {
  description             = "${var.name_prefix} ${var.environment} PostgreSQL"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "database" {
  name          = "alias/${var.name_prefix}-${var.environment}-database"
  target_key_id = aws_kms_key.database.key_id
}
