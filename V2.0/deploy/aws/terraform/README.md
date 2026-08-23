# LabViz AWS Terraform

This directory implements Phase 6C in `ap-southeast-1`. Phase 6C-0 establishes repository, state,
identity, audit, and cost-control boundaries only. It does not prove that staging exists and it does
not satisfy the Phase 6C acceptance gate.

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
