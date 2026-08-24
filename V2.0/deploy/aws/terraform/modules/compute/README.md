# Compute module

This Phase 6C-2 module creates immutable, KMS-encrypted ECR repositories; an ECS/Fargate cluster;
independent API, Web, and worker services; a one-shot Alembic migration task; role-scoped IAM; and
Secrets Manager metadata. It never creates a secret version or stores a secret value in Terraform.

The environment root defaults `activate_services` to `false`. Keep it false until all referenced
image digests exist, the three application secrets have current values, the migration task has
completed successfully, and DNS/ACM/ALB prerequisites are ready. API and Web services use ECS
deployment circuit-breaker rollback plus unhealthy-target alarms. Workers are independent services
and do not use HTTP health checks.

## Secret initialization

Create values outside Terraform with an MFA-backed operator or an approved CI identity. Do not put
values in `.tfvars`, plans, state, shell history, tickets, or logs. The PostgreSQL URL must use
`postgresql+psycopg://` and the RDS-generated master password. Share-token and client-identity keys
must be separate random values of at least 32 bytes.

After the environment metadata is applied, resolve the secret ARNs from the sensitive
`application_secret_arns` output and use `aws secretsmanager put-secret-value` with a temporary
input file that is securely removed. Never pass a secret directly as a command-line argument.

## Release order

1. Push signed API and Web images and record their immutable `sha256:` digests.
2. Populate/rotate all three Secrets Manager values.
3. Register the digest-pinned migration task and run it once in the private application subnets.
4. Require exit code zero and verify the expected Alembic revision.
5. Set `activate_services=true`, deploy API/Web/workers, and wait for ECS stability.
6. Run `/health`, ALB `/api/v1/ready`, application smoke tests, and real AWS evidence collection.

Worker object deletion remains disabled both in environment variables and IAM unless the separate
`enable_worker_delete_permission` acceptance switch is deliberately enabled.
