output "break_glass_role_arn" {
  description = "MFA-required emergency role ARN."
  value       = aws_iam_role.break_glass.arn
}

output "cloudtrail_name" {
  description = "Multi-Region account trail name."
  value       = aws_cloudtrail.account.name
}
