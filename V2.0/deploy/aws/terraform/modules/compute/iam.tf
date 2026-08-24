data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  for_each           = toset(["api", "migration", "web", "worker"])
  name               = "${var.name_prefix}-${var.environment}-${each.key}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "execution" {
  for_each = toset(["api", "migration", "web", "worker"])
  statement {
    sid       = "EcrAuthorization"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "PullImages"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.api.arn, aws_ecr_repository.web.arn]
  }
  statement {
    sid       = "WriteSelectedLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.workloads[each.key].arn}:*"]
  }
  dynamic "statement" {
    for_each = each.key == "api" ? [1] : []
    content {
      sid       = "ReadApiSecrets"
      actions   = ["secretsmanager:GetSecretValue"]
      resources = [aws_secretsmanager_secret.postgres_url.arn, aws_secretsmanager_secret.share_token_keys.arn, aws_secretsmanager_secret.client_identity_key.arn]
    }
  }
  dynamic "statement" {
    for_each = contains(["migration", "worker"], each.key) ? [1] : []
    content {
      sid       = "ReadDatabaseSecret"
      actions   = ["secretsmanager:GetSecretValue"]
      resources = [aws_secretsmanager_secret.postgres_url.arn]
    }
  }
  dynamic "statement" {
    for_each = each.key == "web" ? [] : [1]
    content {
      sid       = "DecryptSelectedSecrets"
      actions   = ["kms:Decrypt"]
      resources = [aws_kms_key.secrets.arn]
    }
  }
}

resource "aws_iam_role_policy" "execution" {
  for_each = aws_iam_role.execution
  name     = "least-privilege-execution"
  role     = each.value.id
  policy   = data.aws_iam_policy_document.execution[each.key].json
}

resource "aws_iam_role" "task" {
  for_each           = toset(["api", "migration", "web", "worker"])
  name               = "${var.name_prefix}-${var.environment}-${each.key}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "object_access" {
  for_each = toset(["api", "worker"])
  statement {
    sid       = "ListApplicationPrefix"
    actions   = ["s3:ListBucket"]
    resources = [var.data_bucket_arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${var.object_prefix}*"]
    }
  }
  statement {
    sid = "ApplicationObjects"
    actions = concat([
      "s3:AbortMultipartUpload",
      "s3:GetObject",
      "s3:ListMultipartUploadParts",
      "s3:PutObject",
    ], each.key == "worker" && var.enable_worker_delete_permission ? ["s3:DeleteObject"] : [])
    resources = ["${var.data_bucket_arn}/${var.object_prefix}*"]
  }
  statement {
    sid       = "ApplicationObjectKey"
    actions   = ["kms:Decrypt", "kms:DescribeKey", "kms:GenerateDataKey"]
    resources = [var.object_kms_key_arn]
  }
}

resource "aws_iam_role_policy" "object_access" {
  for_each = data.aws_iam_policy_document.object_access
  name     = "selected-object-prefix"
  role     = aws_iam_role.task[each.key].id
  policy   = each.value.json
}

data "aws_iam_policy_document" "api_email" {
  statement {
    sid       = "SendFromVerifiedIdentity"
    actions   = ["ses:SendEmail"]
    resources = ["arn:${data.aws_partition.current.partition}:ses:${var.aws_region}:${data.aws_caller_identity.current.account_id}:identity/${var.ses_identity_name}"]
    condition {
      test     = "StringEquals"
      variable = "ses:FromAddress"
      values   = [var.ses_from_address]
    }
  }
}

resource "aws_iam_role_policy" "api_email" {
  name   = "verified-identity-send-only"
  role   = aws_iam_role.task["api"].id
  policy = data.aws_iam_policy_document.api_email.json
}
