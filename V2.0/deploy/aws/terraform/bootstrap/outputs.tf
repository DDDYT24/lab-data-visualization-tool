output "state_bucket_name" {
  description = "Encrypted, versioned Terraform state bucket."
  value       = aws_s3_bucket.terraform_state.id
}

output "github_state_role_arns" {
  description = "OIDC role ARNs for state-only plan and environment access."
  value       = { for name, role in aws_iam_role.github_state : name => role.arn }
}
