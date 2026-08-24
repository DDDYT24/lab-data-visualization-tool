# Email module

Creates an SES v2 domain identity, configuration set, event publishing, and an encrypted SNS event
topic. Terraform exposes DKIM tokens but does not create DNS records; the DNS owner must publish
the returned CNAME records and wait for `verified_for_sending_status=true`.

The notification subscription remains `PendingConfirmation` until its recipient confirms the AWS
email. That confirmation and real accepted/bounce/complaint simulator evidence are Phase 6C live
gates, not code-plan evidence.
