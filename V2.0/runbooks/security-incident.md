# Security incident triage

1. Preserve CloudTrail, CloudWatch, ALB/WAF, ECS, RDS, S3 version, SES, IAM, KMS, and GitHub Actions
   evidence with UTC timestamps. Do not destroy tasks, keys, logs, or versions before capture.
2. Contain narrowly: disable the affected principal/session, stop an affected service or sending
   configuration, and block the specific indicator. Do not use root except account recovery and do
   not broaden public/network access.
3. Determine affected accounts, roles, secrets, data prefixes/versions, recipients, task/image
   digests, and time window. Treat unexplained KMS grants/policies or CloudTrail gaps as high severity.
4. Rotate affected credentials using the rotation runbook; rebuild from accepted signed digests and
   reviewed Terraform. Restore data only through the backup runbook.
5. Validate IAM least privilege, no static keys, no drift, alarms, and representative user flows.
6. Retain the incident record and complete notification/legal review before closure.
