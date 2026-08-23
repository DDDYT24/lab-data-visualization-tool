run "phase6c1_staging_plan" {
  command = plan

  variables {
    owner           = "phase6c1-test"
    domain_name     = "staging.example.com"
    route53_zone_id = "Z0000000000000"
    enable_waf      = false
  }

  assert {
    condition     = var.aws_region == "ap-southeast-1"
    error_message = "Staging must remain in the approved Phase 6C Region."
  }
}
