resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]
}

locals {
  github_subjects = {
    plan = [
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

data "aws_iam_policy_document" "runtime_boundary" {
  for_each = toset(["staging", "production"])

  statement {
    sid = "RuntimeServiceCeiling"
    actions = [
      "backup:*",
      "cloudwatch:*",
      "ec2:*",
      "ecr:*",
      "ecs:*",
      "events:*",
      "kms:*",
      "logs:*",
      "rds:*",
      "scheduler:*",
      "secretsmanager:*",
      "ses:*",
      "sns:*",
    ]
    resources = ["*"]
  }

  statement {
    sid     = "ManageEnvironmentDataBucketsOnly"
    actions = ["s3:*"]
    resources = [
      "arn:${data.aws_partition.current.partition}:s3:::${local.name_prefix}-${each.key}-*",
      "arn:${data.aws_partition.current.partition}:s3:::${local.name_prefix}-${each.key}-*/*",
    ]
  }

  statement {
    sid       = "PassEnvironmentRuntimeRoles"
    actions   = ["iam:PassRole"]
    resources = ["arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${local.name_prefix}-${each.key}-*"]
  }
}

resource "aws_iam_policy" "runtime_boundary" {
  for_each = data.aws_iam_policy_document.runtime_boundary

  name   = "${local.name_prefix}-${each.key}-runtime-boundary"
  policy = each.value.json
}

data "aws_iam_policy_document" "github_infrastructure" {
  for_each = toset(["staging", "production"])

  statement {
    sid = "ManageRegionalLabVizInfrastructure"
    actions = [
      "acm:*",
      "backup:*",
      "cloudwatch:*",
      "ec2:*",
      "ecr:*",
      "ecs:*",
      "elasticloadbalancing:*",
      "events:*",
      "kms:*",
      "logs:*",
      "rds:*",
      "scheduler:*",
      "secretsmanager:*",
      "ses:*",
      "sesv2:*",
      "sns:*",
      "wafv2:*",
    ]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = [var.aws_region]
    }
  }

  statement {
    sid     = "ManageEnvironmentDataBucketsOnly"
    actions = ["s3:*"]
    resources = [
      "arn:${data.aws_partition.current.partition}:s3:::${local.name_prefix}-${each.key}-*",
      "arn:${data.aws_partition.current.partition}:s3:::${local.name_prefix}-${each.key}-*/*",
    ]
  }

  statement {
    sid = "GlobalControlPlaneReads"
    actions = [
      "iam:Get*",
      "iam:List*",
      "route53:Get*",
      "route53:List*",
      "s3:ListAllMyBuckets",
      "tag:GetResources",
    ]
    resources = ["*"]
  }


  statement {
    sid       = "ChangeApprovedHostedZoneOnly"
    actions   = ["route53:ChangeResourceRecordSets"]
    resources = ["arn:${data.aws_partition.current.partition}:route53:::hostedzone/${var.route53_zone_ids[each.key]}"]
  }

  statement {
    sid = "ManageEnvironmentRuntimeRoles"
    actions = [
      "iam:AttachRolePolicy",
      "iam:DeleteRole",
      "iam:DeleteRolePermissionsBoundary",
      "iam:DeleteRolePolicy",
      "iam:DetachRolePolicy",
      "iam:PassRole",
      "iam:PutRolePolicy",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:UpdateAssumeRolePolicy",
      "iam:UpdateRole",
      "iam:UpdateRoleDescription",
    ]
    resources = ["arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${local.name_prefix}-${each.key}-*"]
  }

  statement {
    sid       = "UseOnlyEnvironmentBoundary"
    actions   = ["iam:CreateRole", "iam:PutRolePermissionsBoundary"]
    resources = ["arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${local.name_prefix}-${each.key}-*"]
    condition {
      test     = "ArnEquals"
      variable = "iam:PermissionsBoundary"
      values   = [aws_iam_policy.runtime_boundary[each.key].arn]
    }
  }

  statement {
    sid       = "CreateRequiredServiceLinkedRoles"
    actions   = ["iam:CreateServiceLinkedRole"]
    resources = ["*"]
    condition {
      test     = "StringLike"
      variable = "iam:AWSServiceName"
      values = [
        "backup.amazonaws.com",
        "ecs.amazonaws.com",
        "elasticloadbalancing.amazonaws.com",
        "rds.amazonaws.com",
      ]
    }
  }
}

resource "aws_iam_role_policy" "github_infrastructure" {
  for_each = data.aws_iam_policy_document.github_infrastructure

  name   = "terraform-${each.key}-infrastructure"
  role   = aws_iam_role.github_state[each.key].id
  policy = each.value.json
}

data "aws_iam_policy_document" "github_plan_read" {
  statement {
    sid = "ReadInfrastructureControlPlanes"
    actions = [
      "acm:Describe*",
      "acm:Get*",
      "acm:List*",
      "backup:Describe*",
      "backup:Get*",
      "backup:List*",
      "cloudwatch:Describe*",
      "cloudwatch:Get*",
      "cloudwatch:List*",
      "ec2:Describe*",
      "ecr:Describe*",
      "ecr:GetLifecyclePolicy",
      "ecr:GetRepositoryPolicy",
      "ecr:List*",
      "ecs:Describe*",
      "ecs:List*",
      "elasticloadbalancing:Describe*",
      "events:Describe*",
      "events:List*",
      "iam:Get*",
      "iam:List*",
      "kms:Describe*",
      "kms:Get*",
      "kms:List*",
      "logs:Describe*",
      "logs:Get*",
      "logs:List*",
      "rds:Describe*",
      "rds:List*",
      "route53:Get*",
      "route53:List*",
      "scheduler:Get*",
      "scheduler:List*",
      "secretsmanager:DescribeSecret",
      "secretsmanager:GetResourcePolicy",
      "secretsmanager:List*",
      "ses:Get*",
      "ses:List*",
      "sesv2:Get*",
      "sesv2:List*",
      "sns:Get*",
      "sns:List*",
      "wafv2:Get*",
      "wafv2:List*",
    ]
    resources = ["*"]
  }

  statement {
    sid = "ReadEnvironmentDataBucketConfiguration"
    actions = [
      "s3:GetBucket*",
      "s3:GetEncryptionConfiguration",
      "s3:GetLifecycleConfiguration",
      "s3:GetObjectAttributes",
      "s3:GetObjectTagging",
      "s3:GetObjectVersionAttributes",
      "s3:GetObjectVersionTagging",
      "s3:ListBucket",
      "s3:ListBucketVersions",
    ]
    resources = [
      "arn:${data.aws_partition.current.partition}:s3:::${local.name_prefix}-staging-*",
      "arn:${data.aws_partition.current.partition}:s3:::${local.name_prefix}-staging-*/*",
      "arn:${data.aws_partition.current.partition}:s3:::${local.name_prefix}-production-*",
      "arn:${data.aws_partition.current.partition}:s3:::${local.name_prefix}-production-*/*",
    ]
  }
}

resource "aws_iam_role_policy" "github_plan_read" {
  name   = "terraform-plan-read-without-state-bypass"
  role   = aws_iam_role.github_state["plan"].id
  policy = data.aws_iam_policy_document.github_plan_read.json
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
