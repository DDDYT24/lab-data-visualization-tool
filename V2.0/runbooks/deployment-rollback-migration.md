# Deployment, migration, and rollback

## Preconditions

- CI has accepted all application/Terraform/container gates and signed each immutable digest.
- The reviewed Terraform plan contains only the intended environment and release.
- Secrets are current, SNS subscriptions are confirmed, budget approval is recorded, and no
  unresolved incident or backup failure exists.
- Production uses the protected GitHub environment and a different approval from the author.

## Deploy

1. Record current task-definition revisions, service desired/running counts, database revision, and
   the last successful backup.
2. Register the digest-pinned migration task and run it once in private application subnets. Do not
   update services while it runs.
3. Require `STOPPED`, essential-container exit code `0`, and expected Alembic head. On any failure,
   stop: retain logs, do not retry blindly, and follow Migration failure below.
4. Update API/Web/workers by immutable digest. Wait for ECS steady state and require circuit-breaker
   and deployment alarms to remain healthy.
5. Verify HTTPS, `/health`, ALB `/api/v1/ready`, authentication, upload/download, chart/share, and
   one dry-run cycle for every worker. Record task ARNs and timestamps.
6. Require a post-apply Terraform plan with no unexplained drift.

## Rollback

1. Stop traffic changes and preserve the failed deployment/alarm/log evidence.
2. If ECS did not already roll back, point each affected service to its last completed immutable
   task definition. Never reuse or retag an image.
3. If the migration is backward compatible, restore service health and investigate offline. If it
   is not, keep application traffic stopped and execute only its reviewed downgrade or the backup
   restoration runbook; never edit the schema or Terraform state by hand.
4. Re-run readiness and representative flows, then close the incident only when alarms recover.

## Migration failure

- Do not deploy application services and do not run a second migration until the first stopped task,
  logs, database activity, lock state, and Alembic revision are inspected.
- Terminate only a confirmed abandoned migration session. A human reviews the migration's
  idempotency/downgrade before retry.
- Escalate to restoration when correctness cannot be proven within the four-hour RTO.
