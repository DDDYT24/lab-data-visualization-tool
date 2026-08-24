data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

data "aws_iam_policy_document" "key" {
  statement {
    sid       = "EnableAccountAdministration"
    effect    = "Allow"
    actions   = ["kms:*"]
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
  }

  statement {
    sid    = "AllowCloudWatchLogsEncryption"
    effect = "Allow"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey*",
      "kms:ReEncrypt*",
    ]
    resources = ["*"]
    principals {
      type        = "Service"
      identifiers = ["logs.${var.aws_region}.amazonaws.com"]
    }
    condition {
      test     = "ArnLike"
      variable = "kms:EncryptionContext:aws:logs:arn"
      values   = ["arn:${data.aws_partition.current.partition}:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/labviz/${var.environment}/backup"]
    }
  }
}

resource "aws_kms_key" "backup" {
  description             = "${var.name_prefix} ${var.environment} backup vault, image, and logs"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  policy                  = data.aws_iam_policy_document.key.json

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_kms_alias" "backup" {
  name          = "alias/${var.name_prefix}-${var.environment}-backup"
  target_key_id = aws_kms_key.backup.key_id
}

resource "aws_backup_vault" "this" {
  name        = "${var.name_prefix}-${var.environment}"
  kms_key_arn = aws_kms_key.backup.arn

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_backup_plan" "this" {
  name = "${var.name_prefix}-${var.environment}"

  rule {
    rule_name                    = "continuous-and-daily-30-days"
    target_vault_name            = aws_backup_vault.this.name
    schedule                     = "cron(0 17 * * ? *)"
    schedule_expression_timezone = "UTC"
    start_window                 = 60
    completion_window            = 240
    enable_continuous_backup     = true
    lifecycle {
      delete_after = 30
    }
  }
}

data "aws_iam_policy_document" "backup_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["backup.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "backup" {
  name               = "${var.name_prefix}-${var.environment}-aws-backup"
  assume_role_policy = data.aws_iam_policy_document.backup_assume.json
}

resource "aws_iam_role_policy_attachment" "backup" {
  for_each = toset([
    "AWSBackupServiceRolePolicyForBackup",
    "AWSBackupServiceRolePolicyForS3Backup",
  ])
  role       = aws_iam_role.backup.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/${each.key}"
}

resource "aws_backup_selection" "this" {
  name         = "rds-and-versioned-s3"
  plan_id      = aws_backup_plan.this.id
  iam_role_arn = aws_iam_role.backup.arn
  resources    = [var.database_arn, var.data_bucket_arn]
}

resource "aws_security_group" "restore_testing" {
  name_prefix = "${var.name_prefix}-${var.environment}-restore-testing-"
  description = "No-ingress isolated RDS restore-testing group"
  vpc_id      = var.vpc_id

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_db_subnet_group" "restore_testing" {
  name       = "${var.name_prefix}-${var.environment}-restore-testing"
  subnet_ids = var.database_subnet_ids
}

resource "aws_backup_restore_testing_plan" "quarterly" {
  count = var.enable_restore_testing ? 1 : 0

  name                         = "${replace(var.name_prefix, "-", "_")}_${replace(var.environment, "-", "_")}_quarterly"
  schedule_expression          = "cron(0 3 1 1,4,7,10 ? *)"
  schedule_expression_timezone = "UTC"
  start_window_hours           = 24

  recovery_point_selection {
    algorithm             = "LATEST_WITHIN_WINDOW"
    include_vaults        = [aws_backup_vault.this.arn]
    recovery_point_types  = ["CONTINUOUS", "SNAPSHOT"]
    selection_window_days = 7
  }
}

resource "aws_iam_role" "restore_testing" {
  name               = "${var.name_prefix}-${var.environment}-restore-testing"
  assume_role_policy = data.aws_iam_policy_document.backup_assume.json
}

resource "aws_iam_role_policy_attachment" "restore_testing" {
  role       = aws_iam_role.restore_testing.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForRestores"
}

resource "aws_backup_restore_testing_selection" "rds" {
  count = var.enable_restore_testing ? 1 : 0

  name                      = "rds_isolated"
  restore_testing_plan_name = aws_backup_restore_testing_plan.quarterly[0].name
  iam_role_arn              = aws_iam_role.restore_testing.arn
  protected_resource_type   = "AWS::RDS::DBInstance"
  protected_resource_arns   = [var.database_arn]
  validation_window_hours   = 1
  restore_metadata_overrides = {
    DBInstanceClass     = "db.t4g.micro"
    DBSubnetGroupName   = aws_db_subnet_group.restore_testing.name
    MultiAZ             = "false"
    PubliclyAccessible  = "false"
    VpcSecurityGroupIds = aws_security_group.restore_testing.id
  }
}
