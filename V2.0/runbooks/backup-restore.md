# Backup restoration and quarterly drill

## Targets and evidence

- RPO `<=15 minutes`; RTO `<=4 hours` from declared restore start to validated service readiness.
- Record recovery point, logical dump key/version and SHA-256 manifest, S3 VersionIds, task/image
  digests, schema head, counts, test results, and UTC timing. Never copy production data locally.

## Quarterly isolated drill

1. Create an evidence ID and choose a recent RDS continuous recovery point plus the latest completed
   logical dump. Confirm both are inside retention and their backup alarms are healthy.
2. Restore PITR through AWS Backup restore testing or into drill-named private DB subnets with the
   no-ingress drill security group. Never replace or modify the source database.
3. Separately restore the logical dump with `pg_restore --no-owner --no-acl` into an isolated empty
   database of the same PostgreSQL major version; verify its SHA-256 first.
4. Restore required S3 objects by exact source VersionId into a drill-only prefix/bucket. Never
   overwrite production keys.
5. Run migrations, compare schema head and durable table/object inventories, and exercise
   representative download/export/share operations from an isolated diagnostic task.
6. Measure RPO/RTO. Any mismatch, skipped validation, or missed target fails the drill.
7. Retain evidence first. Delete only resources carrying the exact drill evidence ID and verified
   not referenced by production; AWS Backup recovery points, source versions, state, and logs stay.

## Incident restoration

Stop writes, declare the recovery point and data-loss window, obtain incident approval, restore to a
new isolated target, validate as above, then switch application configuration through reviewed
Terraform/deployment. Never restore over the damaged source until the new target is accepted.
