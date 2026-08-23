# Phase 6C-1 Implementation Record

**Status:** Code-complete locally; no AWS environment was applied and Phase 6C acceptance remains
pending.

## Git boundary

- Direct parent and Phase 6C-0 candidate: `b17595ace4159284253f07801e5402ea1ac34342`.
- Candidate: the commit containing this record on `codex/phase-6c1-network-edge-data`.
- Scope: Phase 6C-1 network, security, data, and edge code only.

## Implemented boundary

- Two-Availability-Zone VPCs with public ALB, private application, and isolated database subnets.
- No NAT, no application/database default route, and no public IP assignment for ECS/RDS subnets.
- Restricted security-group paths for ALB-to-Web/API, API/Worker-to-RDS, private endpoints, DNS, and
  the S3 gateway prefix list.
- Private ECR API/Docker, CloudWatch Logs, Secrets Manager, KMS, and SES API endpoints with selected
  endpoint actions; S3 gateway access includes ECR layers and the application prefix only.
- PostgreSQL 17 with private networking, KMS encryption, forced TLS, managed master secret,
  enhanced monitoring, Performance Insights, at least seven-day PITR, final snapshot policy,
  deletion protection, and production Multi-AZ.
- S3 Block Public Access, ownership enforcement, versioning, SSE-KMS, TLS-only policy, incomplete
  multipart cleanup, noncurrent-version retention, and `prevent_destroy`.
- Route 53 DNS validation, ACM certificate, HTTPS ALB, HTTP redirect, strict invalid/desync header
  handling, append-mode `X-Forwarded-For`, readiness target groups, and optional fail-closed WAF.
- Separate staging and production CIDRs, buckets, databases, KMS keys, certificates, load balancers,
  and Terraform state.

## Evidence boundary

Terraform validation and native plan tests prove only that the locked Provider accepts both graphs.
They do not prove DNS delegation, certificate issuance, endpoint reachability, RDS restore, ALB
forwarding behavior, SES delivery, alarm delivery, or any other real AWS behavior. Those remain
blocking live evidence after reviewed apply.

## Cost boundary

The full private topology cannot be assumed to fit the USD 30 monthly alert. Interface endpoints,
ALB, RDS, ECS, KMS, WAF, logs, storage, queries, and data transfer are separately billable. No 6C-1
resource is created until an AWS Pricing Calculator estimate is reviewed and explicitly approved.
