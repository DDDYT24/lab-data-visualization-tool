# LabViz AWS Terraform

This directory implements Phase 6C in `ap-southeast-1`. Phase 6C-0 establishes repository, state,
identity, audit, and cost-control boundaries only. It does not prove that staging exists and it does
not satisfy the Phase 6C acceptance gate.

Phase 6C-1 adds the network, security, data, and edge modules. The application and database subnets
have no internet route and never auto-assign public IPs. Private application tasks reach ECR,
CloudWatch Logs, Secrets Manager, KMS, and the SES API through interface endpoints and reach S3
through a gateway endpoint. Only the ALB subnets route to the internet gateway; no NAT resource is
created.

Phase 6C-2 adds immutable image repositories, digest-pinned ECS task definitions, independent API,
Web, and worker services, the one-shot migration task, least-privilege runtime roles, and
Secrets Manager metadata. Services default off. Terraform deliberately does not own application
secret values; see [`modules/compute/README.md`](modules/compute/README.md) for the release order.

Phase 6C-3 adds SES operational resources, dashboards/alarms, continuous and daily managed backup,
an encrypted 30-day logical-backup path, opt-in quarterly restore testing, and operator runbooks.
Notification confirmation, schedule activation, and all real restore/alarm evidence remain explicit
post-apply gates.

## Roots and destruction boundaries

| Root | State key | Ownership |
| --- | --- | --- |
| `bootstrap` | local during first creation; migrate to `bootstrap/terraform.tfstate` | state bucket and state-only GitHub OIDC roles |
| `account` | `account/terraform.tfstate` | budget, cost anomaly notification, CloudTrail, break-glass role |
| `environments/staging` | `staging/terraform.tfstate` | staging application infrastructure |
| `environments/production` | `production/terraform.tfstate` | production application infrastructure |

The bootstrap root has `prevent_destroy` on its state bucket and is never included in an
application destroy. State objects cannot be deleted by any generated state policy; only the
corresponding `.tflock` object can be deleted to release a lock.

## Identity

Normal GitHub Actions access uses `https://token.actions.githubusercontent.com` with audience
`sts.amazonaws.com`. Trust is limited to this repository and to the `main`, pull-request,
`staging`, or `production` subject appropriate to each role. No static AWS access key is created.

The `labviz-break-glass` role requires MFA, has a one-hour maximum session, and trusts only the
bootstrap IAM user ARN. After the bootstrap and break-glass path are tested, remove direct
`AdministratorAccess` from the bootstrap user's group. Keep console MFA enabled and use the role
only for a documented incident or account recovery.

## Initial execution

Use Terraform `1.15.x` and the locked AWS provider. Authenticate interactively with the named IAM
profile; do not place credentials in any Terraform file.

```powershell
$env:AWS_PROFILE = "labviz-bootstrap"
terraform -chdir=V2.0/deploy/aws/terraform/bootstrap init
terraform -chdir=V2.0/deploy/aws/terraform/bootstrap validate
terraform -chdir=V2.0/deploy/aws/terraform/bootstrap plan -var="budget_owner=<owner>"
```

Apply bootstrap only after reviewing the plan. Then initialize the other roots with the created
bucket name supplied at runtime:

```powershell
$stateBucket = terraform -chdir=V2.0/deploy/aws/terraform/bootstrap output -raw state_bucket_name
terraform -chdir=V2.0/deploy/aws/terraform/account init -backend-config="bucket=$stateBucket"
terraform -chdir=V2.0/deploy/aws/terraform/environments/staging init -backend-config="bucket=$stateBucket"
terraform -chdir=V2.0/deploy/aws/terraform/environments/production init -backend-config="bucket=$stateBucket"
```

Supply `budget_owner_email` at the command line or through an ignored local `.tfvars` file. The
monthly budget is an alert, not a hard spending cap. The account root deliberately uses gross
resource cost (`include_credit = false`, `include_refund = false`) so free-plan credits do not hide
resource consumption. Never commit `.tfvars`, plans, state, outputs, credentials, email addresses,
or account IDs.

The live budget created before Terraform adoption is project-owned. Reconcile it into the account
root only in a separately reviewed state-adoption step; do not attempt to create a duplicate and do
not import unrelated account resources.

## Validation

```powershell
terraform fmt -check -recursive V2.0/deploy/aws/terraform
terraform -chdir=V2.0/deploy/aws/terraform/bootstrap validate
terraform -chdir=V2.0/deploy/aws/terraform/account validate
terraform -chdir=V2.0/deploy/aws/terraform/environments/staging validate
terraform -chdir=V2.0/deploy/aws/terraform/environments/production validate
```

A complete environment plan remains blocked until bootstrap has been applied and its bucket exists.
No plan, local state, or provider cache is release evidence.

For a backend-independent code plan, Terraform's native tests use an explicit mock AWS provider to
exercise both complete environment graphs without credentials, state access, or resource changes:

```powershell
terraform -chdir=V2.0/deploy/aws/terraform/environments/staging test -filter=phase6c1.tftest.hcl
terraform -chdir=V2.0/deploy/aws/terraform/environments/production test -filter=phase6c1.tftest.hcl
```

The mocked account, domain, hosted-zone ID, certificate, and image digests are non-deployment
values. They are never accepted as live evidence. Main-branch OIDC plans still use the real AWS
provider and remote state. Real apply remains blocked until the public DNS authority described in
[`PHASE6C_EXTERNAL_PREREQUISITES.md`](../../../PHASE6C_EXTERNAL_PREREQUISITES.md) is configured.

## Cost boundary

This topology optimizes for private networking and the Phase 6 contract, not for a USD 30 monthly
runtime. Six interface endpoint services across two Availability Zones are billed per endpoint-AZ
hour before ALB, RDS, ECS, WAF, KMS, logging, storage, and data transfer. The monthly budget is an
alert, not permission to apply. Obtain a current AWS Pricing Calculator estimate and explicit cost
approval before any environment apply. Staging WAF defaults off solely to avoid its recurring cost;
all TLS, header, encryption, public-access, database, and identity controls remain unchanged.
