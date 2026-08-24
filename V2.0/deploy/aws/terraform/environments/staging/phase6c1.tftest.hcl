run "phase6c1_staging_plan" {
  command = plan

  variables {
    owner             = "phase6c1-test"
    domain_name       = "staging.example.com"
    route53_zone_id   = "Z0000000000000"
    enable_waf        = false
    api_image_digest  = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    web_image_digest  = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    ses_identity_name = "example.com"
    ses_from_address  = "noreply@example.com"
    activate_services = false
  }

  assert {
    condition     = var.aws_region == "ap-southeast-1"
    error_message = "Staging must remain in the approved Phase 6C Region."
  }
}
