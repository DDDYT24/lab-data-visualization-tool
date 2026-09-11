# Phase 6C GitHub and DNS setup

The workflow definitions are present on remote `main` but remain intentionally inert until
bootstrap, GitHub environments, DNS, secrets, and cost approval exist. Never store AWS access
keys: all jobs use GitHub OIDC.

The desired branch protection, environment, variable, and secret-name structure is versioned in
[`contracts/phase6-github-governance-v1.json`](contracts/phase6-github-governance-v1.json). Its
The `prepared-not-applied` state is now due to missing external prerequisites and GitHub settings,
not missing workflow files. Current status is tracked in [`TODO.md`](TODO.md).

## 1. Select hosted zones, then bootstrap once from the MFA-backed IAM profile

Create or select the staging/production public hosted zones first and pass their exact IDs through
the bootstrap `route53_zone_ids` variable. This scopes each deployment role to its own zone; DNS
delegation itself may be completed later. Apply only `deploy/aws/terraform/bootstrap` after
reviewing its plan. Record these outputs without posting them publicly:

- `state_bucket_name`
- `github_state_role_arns.plan`, `.staging`, and `.production`
- `runtime_permissions_boundary_arns.staging` and `.production`

The bootstrap removes pull-request trust from the state-reading role. Pull requests run static
Terraform checks without AWS/state access; only `main` can obtain the plan role.

Do not apply the `account` root until its already-created USD 30 budget has been adopted into state
through a separate reviewed import. Otherwise Terraform would attempt to create a duplicate.

## 2. Configure repository-level GitHub values

Repository variables used by the main-branch plan workflow:

- `AWS_PLAN_ROLE_ARN`, `TF_STATE_BUCKET`, `OWNER`
- `STAGING_DOMAIN_NAME`, `STAGING_ROUTE53_ZONE_ID`, `STAGING_SES_IDENTITY_NAME`,
  `STAGING_SES_FROM_ADDRESS`, `STAGING_PERMISSIONS_BOUNDARY_ARN`
- `PRODUCTION_DOMAIN_NAME`, `PRODUCTION_ROUTE53_ZONE_ID`, `PRODUCTION_SES_IDENTITY_NAME`,
  `PRODUCTION_SES_FROM_ADDRESS`, `PRODUCTION_PERMISSIONS_BOUNDARY_ARN`
- the last accepted `STAGING_*_IMAGE_DIGEST` and `PRODUCTION_*_IMAGE_DIGEST` for API, Web, and
  Backup images

Repository secret: `NOTIFICATION_EMAIL`. Its value is not committed or included in release
evidence.

## 3. Configure protected environments

Create GitHub environments named exactly `staging` and `production`. In each environment set:

- variables `AWS_ROLE_ARN`, `TF_STATE_BUCKET`, `OWNER`, `DOMAIN_NAME`, `ROUTE53_ZONE_ID`,
  `SES_IDENTITY_NAME`, `SES_FROM_ADDRESS`, `PERMISSIONS_BOUNDARY_ARN`, and
  `COST_APPROVAL_REFERENCE`;
- secret `NOTIFICATION_EMAIL`;
- deployment branch `main` only.

Production must require a reviewer other than the release author and prevent self-review. Staging
should also require review until the first restore, alarm, SES, and rollback evidence passes.

## 4. DNS authority required before foundation apply

Create or choose the public Route 53 hosted zone that is authoritative for the application domain.
At the registrar/DNS parent, delegate the domain/subdomain to the four Route 53 name servers. Put
the hosted-zone ID and exact FQDN into GitHub environment variables. The foundation workflow then
creates ACM validation and application alias records.

After foundation apply, publish the three SES DKIM CNAME records represented by the
`ses_dkim_tokens` output if the SES identity is not in the same Terraform-controlled zone. Wait for
ACM `ISSUED` and SES `verified_for_sending_status=true`.

## 5. Safe execution order

1. Obtain a current AWS Pricing Calculator estimate and explicit approval; the USD 30 budget is an
   alert, not a hard cap. Record the approval reference in `COST_APPROVAL_REFERENCE`; keep it absent
   for the default local-only mode. See [`PHASE6_COST_BOUNDARY.md`](PHASE6_COST_BOUNDARY.md).
2. Run `Phase 6C foundation apply` for staging. It uses placeholder digests and verifies every ECS
   service remains at desired count zero; backup/restore schedules remain off.
3. Populate the three application Secrets Manager values outside Terraform and confirm both SNS
   email subscriptions.
4. Let the exact `main` commit pass the full `CI` workflow.
5. Run `Phase 6C release` for staging. It builds/pushes/scans/signs images, saves a reviewed plan,
   runs the candidate migration, applies, waits for all services, checks HTTPS/alarms/drift, and
   rolls services back on failure.
6. Collect every item in `contracts/phase6c-evidence-v1.json`. Production remains blocked until the
   staging gate passes.
