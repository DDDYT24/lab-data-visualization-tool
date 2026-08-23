# Phase 6C External Prerequisites

These values are account inputs, not source-code defaults. Do not commit the real email address,
hosted-zone ID, domain registrar credentials, Terraform variables, plans, state, or AWS account ID.

## 1. Public domain and DNS authority

Choose one path:

1. **Existing domain already using Route 53:** open Route 53 > Hosted zones, select the public hosted
   zone, and copy its Hosted zone ID.
2. **Existing domain using another DNS provider:** either keep that provider and delegate a LabViz
   subdomain to a new Route 53 public hosted zone, or move the whole domain's authoritative DNS to
   Route 53. For subdomain delegation, create a public hosted zone such as `labviz.example.com`,
   copy its four Route 53 name servers, then add an `NS` record for `labviz.example.com` at the
   current authoritative provider. Do not delete unrelated DNS records.
3. **No domain:** register a domain with a registrar first. Domain registration is a separate paid
   purchase. A Route 53 public hosted zone also has a recurring charge.

Verify delegation from Windows before Terraform apply:

```powershell
Resolve-DnsName -Type NS example.com
Resolve-DnsName -Type NS labviz.example.com
```

The returned authoritative servers must match the four Route 53 name servers. Then select distinct
names, for example `staging.labviz.example.com` and `app.labviz.example.com`. Copy the environment
example to an ignored `terraform.tfvars` and set only locally:

```hcl
owner           = "your-owner-label"
domain_name     = "staging.labviz.example.com"
route53_zone_id = "your-public-hosted-zone-id"
```

Terraform will create the ACM validation CNAME and the ALB alias record. ACM public certificate DNS
validation must resolve through a public hosted zone; a private hosted zone is insufficient.

## 2. SES sender identity

In the AWS console, switch to `ap-southeast-1`, open Amazon SES, then create a verified **domain**
identity for the same delegated domain or a dedicated mail subdomain. Enable Easy DKIM and publish
all generated CNAME records in the authoritative public zone. When Route 53 owns the zone, allow SES
to create the records; otherwise copy them exactly to the external DNS provider.

Keep the account in the SES sandbox for the first simulator/verified-recipient tests. Request
production access only after the sender domain, bounce/complaint handling, suppression, alarms, and
the sending use case are ready. Record the identity ARN and chosen `From` address privately for the
later email module; do not create SMTP credentials because the application uses the SES v2 API task
role.

## 3. Alarm destination

Use a monitored mailbox owned by the budget/operations owner. The later observability unit creates
an SNS topic and email subscription. AWS sends a confirmation message; the owner must open it and
select **Confirm subscription** before alarm delivery can pass. The budget email alone does not
confirm the future CloudWatch/SNS subscription.

## 4. Cost approval before apply

Create an AWS Pricing Calculator estimate for `ap-southeast-1` covering at least:

- six interface endpoint services in two Availability Zones;
- one ALB plus expected LCU usage;
- staging RDS PostgreSQL and storage/backups;
- ECS Web/API/Worker task hours;
- two customer-managed KMS keys per environment;
- Route 53 hosted zone/queries, optional WAF, CloudWatch logs/metrics, S3, and data transfer.

The current USD 30 budget is an alert, not a hard cap. Do not apply if the gross monthly estimate is
above the approved limit, even when promotional credits could offset the invoice.
