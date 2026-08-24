data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

resource "aws_sesv2_configuration_set" "this" {
  configuration_set_name = "${var.name_prefix}-${var.environment}"

  delivery_options {
    tls_policy = "REQUIRE"
  }
  reputation_options {
    reputation_metrics_enabled = true
  }
  sending_options {
    sending_enabled = true
  }
  suppression_options {
    suppressed_reasons = ["BOUNCE", "COMPLAINT"]
  }
  vdm_options {
    dashboard_options {
      engagement_metrics = "ENABLED"
    }
    guardian_options {
      optimized_shared_delivery = "ENABLED"
    }
  }
}

resource "aws_sesv2_email_identity" "this" {
  email_identity         = var.identity_name
  configuration_set_name = aws_sesv2_configuration_set.this.configuration_set_name
}

resource "aws_sns_topic" "events" {
  name              = "${var.name_prefix}-${var.environment}-ses-events"
  kms_master_key_id = "alias/aws/sns"
}

data "aws_iam_policy_document" "events" {
  statement {
    sid       = "AllowSesEventPublishing"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.events.arn]
    principals {
      type        = "Service"
      identifiers = ["ses.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "AWS:SourceArn"
      values   = [aws_sesv2_configuration_set.this.arn]
    }
  }
}

resource "aws_sns_topic_policy" "events" {
  arn    = aws_sns_topic.events.arn
  policy = data.aws_iam_policy_document.events.json
}

resource "aws_sns_topic_subscription" "events_email" {
  topic_arn = aws_sns_topic.events.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

resource "aws_sesv2_configuration_set_event_destination" "sns" {
  configuration_set_name = aws_sesv2_configuration_set.this.configuration_set_name
  event_destination_name = "ses-events"

  event_destination {
    enabled = true
    matching_event_types = [
      "BOUNCE",
      "COMPLAINT",
      "DELIVERY",
      "DELIVERY_DELAY",
      "REJECT",
      "SEND",
      "SUBSCRIPTION",
    ]
    sns_destination {
      topic_arn = aws_sns_topic.events.arn
    }
  }

  depends_on = [aws_sns_topic_policy.events]
}

resource "aws_sesv2_configuration_set_event_destination" "cloudwatch" {
  configuration_set_name = aws_sesv2_configuration_set.this.configuration_set_name
  event_destination_name = "cloudwatch-metrics"

  event_destination {
    enabled = true
    matching_event_types = [
      "BOUNCE",
      "COMPLAINT",
      "DELIVERY",
      "DELIVERY_DELAY",
      "REJECT",
      "SEND",
    ]
    cloud_watch_destination {
      dimension_configuration {
        default_dimension_value = var.environment
        dimension_name          = "environment"
        dimension_value_source  = "MESSAGE_TAG"
      }
    }
  }
}
