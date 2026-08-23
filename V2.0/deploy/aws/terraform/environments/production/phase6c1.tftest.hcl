run "phase6c1_production_plan" {
  command = plan

  variables {
    owner           = "phase6c1-test"
    domain_name     = "app.example.com"
    route53_zone_id = "Z0000000000000"
    enable_waf      = true
  }

  assert {
    condition     = var.aws_region == "ap-southeast-1"
    error_message = "Production must remain in the approved Phase 6C Region."
  }
}
