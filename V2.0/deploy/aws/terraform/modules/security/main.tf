locals {
  application_security_groups = {
    api    = aws_security_group.api.id
    web    = aws_security_group.web.id
    worker = aws_security_group.worker.id
  }
  interface_endpoint_actions = {
    "ecr.api" = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:DescribeImages",
      "ecr:GetAuthorizationToken",
      "ecr:GetDownloadUrlForLayer",
    ]
    "ecr.dkr" = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    logs = [
      "logs:CreateLogStream",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents",
    ]
    secretsmanager = [
      "secretsmanager:DescribeSecret",
      "secretsmanager:GetSecretValue",
    ]
    kms = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:GenerateDataKey",
    ]
    email = [
      "ses:SendEmail",
    ]
  }
}

resource "aws_security_group" "alb" {
  name        = "${var.name_prefix}-${var.environment}-alb"
  description = "Public HTTPS entry point; HTTP exists only for redirect"
  vpc_id      = var.vpc_id
}

resource "aws_security_group" "api" {
  name        = "${var.name_prefix}-${var.environment}-api"
  description = "Private API tasks"
  vpc_id      = var.vpc_id
}

resource "aws_security_group" "web" {
  name        = "${var.name_prefix}-${var.environment}-web"
  description = "Private Web tasks"
  vpc_id      = var.vpc_id
}

resource "aws_security_group" "worker" {
  name        = "${var.name_prefix}-${var.environment}-worker"
  description = "Private Worker tasks"
  vpc_id      = var.vpc_id
}

resource "aws_security_group" "database" {
  name        = "${var.name_prefix}-${var.environment}-database"
  description = "Isolated PostgreSQL ingress from API and Worker only"
  vpc_id      = var.vpc_id
}

resource "aws_security_group" "endpoints" {
  name        = "${var.name_prefix}-${var.environment}-endpoints"
  description = "PrivateLink HTTPS from application tasks only"
  vpc_id      = var.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "alb_http" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 80
  to_port           = 80
  ip_protocol       = "tcp"
  description       = "Redirect public HTTP to HTTPS"
}

resource "aws_vpc_security_group_ingress_rule" "alb_https" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
  description       = "Public HTTPS"
}

resource "aws_vpc_security_group_ingress_rule" "web_from_alb" {
  security_group_id            = aws_security_group.web.id
  referenced_security_group_id = aws_security_group.alb.id
  from_port                    = var.web_port
  to_port                      = var.web_port
  ip_protocol                  = "tcp"
  description                  = "Web traffic from ALB only"
}

resource "aws_vpc_security_group_ingress_rule" "api_from_alb" {
  security_group_id            = aws_security_group.api.id
  referenced_security_group_id = aws_security_group.alb.id
  from_port                    = var.api_port
  to_port                      = var.api_port
  ip_protocol                  = "tcp"
  description                  = "API traffic from ALB only"
}

resource "aws_vpc_security_group_egress_rule" "alb_to_web" {
  security_group_id            = aws_security_group.alb.id
  referenced_security_group_id = aws_security_group.web.id
  from_port                    = var.web_port
  to_port                      = var.web_port
  ip_protocol                  = "tcp"
  description                  = "ALB to Web targets"
}

resource "aws_vpc_security_group_egress_rule" "alb_to_api" {
  security_group_id            = aws_security_group.alb.id
  referenced_security_group_id = aws_security_group.api.id
  from_port                    = var.api_port
  to_port                      = var.api_port
  ip_protocol                  = "tcp"
  description                  = "ALB to API targets"
}

resource "aws_vpc_security_group_ingress_rule" "database_from_api" {
  security_group_id            = aws_security_group.database.id
  referenced_security_group_id = aws_security_group.api.id
  from_port                    = var.database_port
  to_port                      = var.database_port
  ip_protocol                  = "tcp"
  description                  = "PostgreSQL from API tasks"
}

resource "aws_vpc_security_group_ingress_rule" "database_from_worker" {
  security_group_id            = aws_security_group.database.id
  referenced_security_group_id = aws_security_group.worker.id
  from_port                    = var.database_port
  to_port                      = var.database_port
  ip_protocol                  = "tcp"
  description                  = "PostgreSQL from Worker tasks"
}

resource "aws_vpc_security_group_egress_rule" "api_to_database" {
  security_group_id            = aws_security_group.api.id
  referenced_security_group_id = aws_security_group.database.id
  from_port                    = var.database_port
  to_port                      = var.database_port
  ip_protocol                  = "tcp"
  description                  = "API to PostgreSQL"
}

resource "aws_vpc_security_group_egress_rule" "worker_to_database" {
  security_group_id            = aws_security_group.worker.id
  referenced_security_group_id = aws_security_group.database.id
  from_port                    = var.database_port
  to_port                      = var.database_port
  ip_protocol                  = "tcp"
  description                  = "Worker to PostgreSQL"
}

resource "aws_vpc_security_group_ingress_rule" "endpoint_from_application" {
  for_each = local.application_security_groups

  security_group_id            = aws_security_group.endpoints.id
  referenced_security_group_id = each.value
  from_port                    = 443
  to_port                      = 443
  ip_protocol                  = "tcp"
  description                  = "HTTPS from ${each.key} tasks"
}

resource "aws_vpc_security_group_egress_rule" "application_to_endpoints" {
  for_each = local.application_security_groups

  security_group_id            = each.value
  referenced_security_group_id = aws_security_group.endpoints.id
  from_port                    = 443
  to_port                      = 443
  ip_protocol                  = "tcp"
  description                  = "${each.key} tasks to AWS interface endpoints"
}

resource "aws_vpc_security_group_egress_rule" "application_dns_udp" {
  for_each = local.application_security_groups

  security_group_id = each.value
  cidr_ipv4         = var.vpc_cidr
  from_port         = 53
  to_port           = 53
  ip_protocol       = "udp"
  description       = "DNS through the VPC resolver"
}

resource "aws_vpc_security_group_egress_rule" "application_dns_tcp" {
  for_each = local.application_security_groups

  security_group_id = each.value
  cidr_ipv4         = var.vpc_cidr
  from_port         = 53
  to_port           = 53
  ip_protocol       = "tcp"
  description       = "Large DNS responses through the VPC resolver"
}

data "aws_iam_policy_document" "interface_endpoint" {
  for_each = local.interface_endpoint_actions

  statement {
    sid       = "SelectedServiceActions"
    effect    = "Allow"
    actions   = each.value
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
  }
}

resource "aws_vpc_endpoint" "interface" {
  for_each = local.interface_endpoint_actions

  vpc_id              = var.vpc_id
  service_name        = "com.amazonaws.${var.aws_region}.${each.key}"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true
  subnet_ids          = var.application_subnet_ids
  security_group_ids  = [aws_security_group.endpoints.id]
  policy              = data.aws_iam_policy_document.interface_endpoint[each.key].json

  tags = {
    Name = "${var.name_prefix}-${var.environment}-${replace(each.key, ".", "-")}"
  }
}
