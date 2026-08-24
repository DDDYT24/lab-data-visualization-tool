resource "aws_db_subnet_group" "this" {
  name       = "${var.name_prefix}-${var.environment}"
  subnet_ids = var.database_subnet_ids

  tags = {
    Name = "${var.name_prefix}-${var.environment}"
  }
}

resource "aws_db_parameter_group" "postgres17" {
  name   = "${var.name_prefix}-${var.environment}-postgres17"
  family = "postgres17"

  parameter {
    name         = "rds.force_ssl"
    value        = "1"
    apply_method = "immediate"
  }
}

data "aws_iam_policy_document" "rds_monitoring_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["monitoring.rds.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "rds_monitoring" {
  name                 = "${var.name_prefix}-${var.environment}-rds-monitoring"
  assume_role_policy   = data.aws_iam_policy_document.rds_monitoring_assume.json
  max_session_duration = 3600
  permissions_boundary = var.permissions_boundary_arn
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

resource "aws_db_instance" "postgres" {
  identifier = "${var.name_prefix}-${var.environment}"

  engine         = "postgres"
  engine_version = "17"
  instance_class = var.database_instance_class

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.database.arn

  db_name                       = "labviz"
  username                      = "labviz_admin"
  manage_master_user_password   = true
  master_user_secret_kms_key_id = aws_kms_key.database.arn
  port                          = 5432

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [var.database_security_group_id]
  publicly_accessible    = false
  multi_az               = var.database_multi_az

  parameter_group_name = aws_db_parameter_group.postgres17.name
  enabled_cloudwatch_logs_exports = [
    "postgresql",
    "upgrade",
  ]
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn

  performance_insights_enabled          = true
  performance_insights_kms_key_id       = aws_kms_key.database.arn
  performance_insights_retention_period = 7

  backup_retention_period  = var.database_backup_retention_days
  backup_window            = "17:00-17:30"
  maintenance_window       = "sun:18:00-sun:19:00"
  copy_tags_to_snapshot    = true
  delete_automated_backups = false

  auto_minor_version_upgrade  = true
  allow_major_version_upgrade = false
  apply_immediately           = false
  deletion_protection         = var.database_deletion_protection
  skip_final_snapshot         = false
  final_snapshot_identifier   = "${var.name_prefix}-${var.environment}-final"

  lifecycle {
    prevent_destroy = true
  }
}
