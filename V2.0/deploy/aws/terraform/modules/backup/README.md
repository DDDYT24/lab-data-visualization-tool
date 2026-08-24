# Backup module

Creates a KMS-encrypted AWS Backup vault, continuous plus daily RDS/S3 protection with 30-day
retention, a digest-pinned daily `pg_dump` Fargate task, and opt-in quarterly AWS Backup restore
testing into a no-ingress security group. The logical backup task uploads a custom-format dump and
SHA-256 manifest to the versioned data bucket under `backups/logical/`.

Both the logical schedule and quarterly restore testing default off. Enable them only after the
backup image exists, the database secret is populated, SNS alarms are confirmed, and recurring
cost is approved. Automated restore testing supplements, but does not replace, the quarterly
operator drill in `V2.0/runbooks/backup-restore.md`.
