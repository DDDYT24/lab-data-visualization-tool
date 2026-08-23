data "aws_partition" "current" {}

data "aws_iam_policy_document" "break_glass_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type = "AWS"
      identifiers = [
        "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:user/${var.bootstrap_user_name}",
      ]
    }
    condition {
      test     = "Bool"
      variable = "aws:MultiFactorAuthPresent"
      values   = ["true"]
    }
  }
}

resource "aws_iam_role" "break_glass" {
  name                 = "${local.name_prefix}-break-glass"
  description          = "MFA-only, one-hour emergency account recovery role"
  assume_role_policy   = data.aws_iam_policy_document.break_glass_assume.json
  max_session_duration = 3600
}

resource "aws_iam_role_policy_attachment" "break_glass_admin" {
  role       = aws_iam_role.break_glass.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AdministratorAccess"
}
