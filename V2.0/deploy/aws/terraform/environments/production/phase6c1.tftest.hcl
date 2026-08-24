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
  }

  assert {
    condition     = var.aws_region == "ap-southeast-1"
    error_message = "Production must remain in the approved Phase 6C Region."
  }
}
