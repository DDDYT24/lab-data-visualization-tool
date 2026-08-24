output "state_bucket_name" {
  description = "Encrypted, versioned Terraform state bucket."
  value       = aws_s3_bucket.terraform_state.id
}

output "github_state_role_arns" {
  description = "OIDC role ARNs for state-only plan and environment access."
  value       = { for name, role in aws_iam_role.github_state : name => role.arn }
}

output "runtime_permissions_boundary_arns" {
  description = "Environment-specific boundaries required on every Terraform-created runtime role."
  value       = { for name, policy in aws_iam_policy.runtime_boundary : name => policy.arn }
}
