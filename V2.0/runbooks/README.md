# Phase 6C operations runbooks

These runbooks are executable checklists for the AWS staging/production runtime. Every incident or
drill records UTC start/end times, environment, immutable image digests, Terraform commit/plan,
CloudWatch alarm IDs, operator, observations, and evidence location. Never paste credentials,
secret values, authentication codes, recipient addresses, object contents, state, or saved plans
into evidence.

- [`deployment-rollback-migration.md`](deployment-rollback-migration.md)
- [`dependency-outages.md`](dependency-outages.md)
- [`worker-operations.md`](worker-operations.md)
- [`credential-rotation.md`](credential-rotation.md)
- [`backup-restore.md`](backup-restore.md)
- [`security-incident.md`](security-incident.md)

Production objectives: monthly public availability `>=99.9%`, RPO `<=15 minutes`, RTO `<=4
hours`. A dashboard or alarm definition is not measured proof; Phase 6C acceptance requires real
staging alarm delivery and restore evidence.
