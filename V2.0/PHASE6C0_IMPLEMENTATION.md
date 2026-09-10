# Phase 6C-0 Implementation Record

**Status:** Implemented locally; live bootstrap apply and Phase 6C acceptance remain pending.

## Git boundary

- Direct parent and accepted Phase 6B boundary: `75adb8a0cc588fa5056da5725c504190978aa0a4`.
- Candidate: the commit containing this record on the Phase 6C-0 local development branch.
- Scope: Phase 6C-0 state, identity, account guardrails, and repository layout only.

This unit does not claim a deployed state bucket, OIDC role, CloudTrail, cost-anomaly monitor,
staging environment, SES delivery, ALB behavior, or CloudWatch alarm. Those claims require a
reviewed real-account apply and retained read-only evidence.

## Implemented boundary

- Locked Terraform roots for bootstrap, account guardrails, staging, and production.
- Encrypted, versioned, public-blocked state bucket with `prevent_destroy` and TLS-only access.
- GitHub Actions OIDC trust constrained by audience, repository, branch/pull request, and protected
  environment subjects; no static access key resource exists.
- State policies can delete only `.tflock` objects, never state objects.
- MFA-required, one-hour break-glass role and documented removal path for direct bootstrap admin.
- USD 30 gross-resource budget model, four notifications, daily cost anomaly notification, and
  multi-Region validated CloudTrail model.
- Reserved Phase 6C modules and isolated state keys for staging and production.

## Remaining external steps

1. Review and apply `bootstrap` with the MFA-protected `labviz-bootstrap` profile.
2. Migrate the bootstrap state into its protected S3 key and initialize `account` with the bucket
   name supplied outside Git.
3. Reconcile the already-created project budget into the account root in a separately reviewed
   state-adoption step; do not create a duplicate.
4. Apply the remaining account guardrails, verify OIDC sessions, and only then remove direct
   administrator policy from the bootstrap group.
5. Obtain the domain/DNS authority, SES sender identity, and alarm destination before 6C-1 live
   planning or apply.
