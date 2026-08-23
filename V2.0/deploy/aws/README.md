# AWS deployment contract

Phase 6A provides deployable images and ECS task-definition templates; it does not create or mutate
an AWS account. Replace every `${...}` placeholder in CI/CD, register immutable image digests, run
the migration task, then update the API, Worker, and Web services.

Phase 6C infrastructure starts in [`terraform/README.md`](terraform/README.md). The `bootstrap`,
`account`, `staging`, and `production` roots have separate state and destruction boundaries. Never
run an application-environment destroy against the bootstrap root.

Build from `V2.0`:

```powershell
docker build -f api/Dockerfile -t labviz-api:phase6a ./api
docker build -f web/Dockerfile -t labviz-web:phase6a .
```

Use the API image for three roles:

```text
API:       python -m uvicorn labviz_api.main:app --host 0.0.0.0 --port 8000
Migration: python -m alembic upgrade head
Worker:    python -m labviz_api.workers.cli <task>
```

Create a separate ECS Worker service per existing task so failures and scaling stay isolated.
The essential Worker process owns its task lifetime; do not attach a PostgreSQL/S3 dependency probe
as its ECS container health check. Use `python -m labviz_api.runtime_health` as a bounded deployment
preflight or operator diagnostic, and monitor Worker heartbeat, backlog, lease, retry, and
quarantine signals separately. Run the migration as a one-shot task before deploying code that
requires the new schema; never run Alembic concurrently from every API replica.

Set `LABVIZ_RUNTIME_ROLE=api` or `worker`. The API role owns HTTPS-origin, SES, cookie, and share
key validation. The Worker role deliberately receives no email-provider or share-token secret; it validates
only its PostgreSQL/S3 production dependencies.

Production authentication email uses the SES v2 API through the ECS task-role credential chain.
The API task supplies `LABVIZ_SES_REGION`, `LABVIZ_SES_FROM`, and
`LABVIZ_SES_CONFIGURATION_SET` as non-secret deployment configuration and receives no SMTP or
static AWS credential. Attach [`api-ses-task-role-policy.example.json`](api-ses-task-role-policy.example.json)
after replacing its identity/from placeholders; it grants only `ses:SendEmail` for that verified
identity and sender. The application sends one recipient, uses only `purpose` and `environment`
tags, and records the returned SES MessageId without the recipient or verification code.

Before public traffic, Phase 6C Terraform must create the configuration-set event destinations for
send, delivery, reject, rendering failure, delivery delay, hard bounce, and complaint events. It
must also enable account-level bounce/complaint suppression and alarms. With the complete staging
production environment and task role loaded, run the probe separately for an accepted mailbox and
the SES bounce/complaint simulator recipients:

```powershell
$env:LABVIZ_SES_PROBE_RECIPIENT = "<mailbox-simulator-or-verified-recipient>"
python -m scripts.probe_ses_delivery
Remove-Item Env:LABVIZ_SES_PROBE_RECIPIENT
```

The probe prints only status, provider, and MessageId. Preserve those IDs with the corresponding
SES event and alarm evidence; never paste recipients, message bodies, credentials, or sign-in URLs
into an acceptance report.

The API container health check calls dependency-free `/health`. The ALB API target group's health
check calls `/api/v1/ready`, so a PostgreSQL or S3 outage removes the task from traffic without
causing ECS restart storms. The ALB routes `/api/v1/*` and `/health` to that target group and all
other paths to Web. TLS terminates at the ALB; the database URL must still include
`sslmode=require` or stronger. API and Workers run in private subnets with an RDS security-group
path and an S3 gateway endpoint or NAT. Only the ALB is public.

Set `LABVIZ_TRUSTED_PROXY_CIDRS` to the private ALB subnet CIDRs and
`LABVIZ_TRUSTED_PROXY_HOPS=1` for the direct append-mode ALB path. Store the independent
`LABVIZ_CLIENT_IDENTITY_KEY` in Secrets Manager. The API ignores `X-Forwarded-For` from an
untrusted immediate peer and stores only the keyed client digest; Phase 6C staging must recheck the
real ALB chain before public traffic.

Authentication request limits use PostgreSQL database-time fixed windows and atomically consume
the client and normalized-email buckets. Keep the initial production values at the migration-
backfilled defaults (`3600`, `30`, and `10`) through the first full window after deploying
migration `0009`; later changes are reviewed configuration releases because changing a window can
intentionally start a new bucket generation. Limiter failure returns 503 and never sends a code.
The client and email values are independently tunable within their bounds. Deployment must run
`0009` before admitting the new API revision and must not keep an old count-based API revision
serving during the transition; this prevents post-backfill requests from bypassing the new buckets.

Secrets are referenced from Secrets Manager in `api-task-definition.example.json`; no secret value
or static AWS access key belongs in a task definition. The task role grants only the selected S3
prefix and the required KMS operations. Web has no database, object-storage, or email credentials.

Required Phase 6C resource controls before launch are RDS/S3 encryption, S3 Block Public Access and
versioning, RDS PITR of at least seven days, 30-day daily logical backups, log retention, alarms,
restore evidence, and a documented region. Do not claim that these templates alone satisfy those
operational controls.
