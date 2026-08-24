mock_provider "aws" {
  override_during = plan

  mock_data "aws_availability_zones" {
    defaults = {
      names = ["ap-southeast-1a", "ap-southeast-1b"]
    }
  }

  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "000000000000"
      arn        = "arn:aws:iam::000000000000:user/terraform-test"
      user_id    = "terraform-test"
    }
  }

  mock_data "aws_partition" {
    defaults = {
      partition  = "aws"
      dns_suffix = "amazonaws.com"
    }
  }

  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}"
    }
  }

  mock_resource "aws_acm_certificate" {
    defaults = {
      arn = "arn:aws:acm:ap-southeast-1:000000000000:certificate/00000000-0000-0000-0000-000000000000"
      domain_validation_options = [{
        domain_name           = "app.example.com"
        resource_record_name  = "_mock.app.example.com"
        resource_record_type  = "CNAME"
        resource_record_value = "mock.acm-validations.aws"
      }]
    }
  }

  mock_resource "aws_acm_certificate_validation" {
    defaults = {
      certificate_arn = "arn:aws:acm:ap-southeast-1:000000000000:certificate/00000000-0000-0000-0000-000000000000"
    }
  }

}

run "phase6c1_production_plan" {
  command = plan

  variables {
    owner                            = "phase6c1-test"
    domain_name                      = "app.example.com"
    route53_zone_id                  = "Z0000000000000"
    enable_waf                       = true
    api_image_digest                 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    web_image_digest                 = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    backup_image_digest              = "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    ses_identity_name                = "example.com"
    ses_from_address                 = "noreply@example.com"
    notification_email               = "ops@example.com"
    activate_services                = false
    activate_logical_backup_schedule = true
    enable_restore_testing           = true
    permissions_boundary_arn         = "arn:aws:iam::000000000000:policy/labviz-production-runtime-boundary"
  }

  assert {
    condition     = var.aws_region == "ap-southeast-1"
    error_message = "Production must remain in the approved Phase 6C Region."
  }
}
