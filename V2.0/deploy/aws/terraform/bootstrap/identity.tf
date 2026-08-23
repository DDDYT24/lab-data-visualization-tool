resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]
}

locals {
  github_subjects = {
    plan = [
      "repo:${var.github_repository}:pull_request",
      "repo:${var.github_repository}:ref:refs/heads/main",
    ]
    staging    = ["repo:${var.github_repository}:environment:staging"]
    production = ["repo:${var.github_repository}:environment:production"]
  }
  state_prefixes = {
    plan       = ["account", "staging", "production"]
    staging    = ["staging"]
    production = ["production"]
  }
}

data "aws_iam_policy_document" "github_assume" {
  for_each = local.github_subjects

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = each.value
    }
  }
}

resource "aws_iam_role" "github_state" {
  for_each = local.github_subjects

  name                 = "${local.name_prefix}-github-${each.key}"
  assume_role_policy   = data.aws_iam_policy_document.github_assume[each.key].json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "github_state" {
  for_each = local.state_prefixes

  statement {
    sid       = "ListSelectedStatePrefixes"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.terraform_state.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values = concat(
        [for prefix in each.value : "${prefix}/terraform.tfstate"],
        [for prefix in each.value : "${prefix}/terraform.tfstate.tflock"],
      )
    }
  }

  statement {
    sid     = "ReadSelectedState"
    effect  = "Allow"
    actions = ["s3:GetObject"]
    resources = [
      for prefix in each.value : "${aws_s3_bucket.terraform_state.arn}/${prefix}/terraform.tfstate"
    ]
  }

  dynamic "statement" {
    for_each = each.key == "plan" ? [] : [1]
    content {
      sid     = "WriteSelectedState"
      effect  = "Allow"
      actions = ["s3:PutObject"]
      resources = [
        for prefix in each.value : "${aws_s3_bucket.terraform_state.arn}/${prefix}/terraform.tfstate"
      ]
    }
  }

  statement {
    sid     = "ManageSelectedLockFiles"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [
      for prefix in each.value : "${aws_s3_bucket.terraform_state.arn}/${prefix}/terraform.tfstate.tflock"
    ]
  }
}

resource "aws_iam_role_policy" "github_state" {
  for_each = local.state_prefixes

  name   = "terraform-state-${each.key}"
  role   = aws_iam_role.github_state[each.key].id
  policy = data.aws_iam_policy_document.github_state[each.key].json
}
