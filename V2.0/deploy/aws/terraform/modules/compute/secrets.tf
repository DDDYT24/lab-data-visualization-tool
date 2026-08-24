resource "aws_kms_key" "secrets" {
  description             = "${var.name_prefix} ${var.environment} application secrets"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "secrets" {
  name          = "alias/${var.name_prefix}-${var.environment}-secrets"
  target_key_id = aws_kms_key.secrets.key_id
}

resource "aws_secretsmanager_secret" "postgres_url" {
  name                    = "${var.name_prefix}/${var.environment}/postgres-url"
  description             = "SQLAlchemy PostgreSQL URL assembled from the RDS-managed credential"
  kms_key_id              = aws_kms_key.secrets.arn
  recovery_window_in_days = 30
}

resource "aws_secretsmanager_secret" "share_token_keys" {
  name                    = "${var.name_prefix}/${var.environment}/share-token-keys"
  description             = "Versioned LabViz share-token key ring"
  kms_key_id              = aws_kms_key.secrets.arn
  recovery_window_in_days = 30
}

resource "aws_secretsmanager_secret" "client_identity_key" {
  name                    = "${var.name_prefix}/${var.environment}/client-identity-key"
  description             = "Independent keyed client-identity digest secret"
  kms_key_id              = aws_kms_key.secrets.arn
  recovery_window_in_days = 30
}
