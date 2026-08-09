# AWS deployment contract

Phase 6A provides deployable images and ECS task-definition templates; it does not create or mutate
an AWS account. Replace every `${...}` placeholder in CI/CD, register immutable image digests, run
the migration task, then update the API, Worker, and Web services.

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
Override its image health check with `python -m labviz_api.runtime_health`. Run the migration as a
one-shot task before deploying code that requires the new schema; never run Alembic concurrently
from every API replica.

Set `LABVIZ_RUNTIME_ROLE=api` or `worker`. The API role owns HTTPS-origin, SMTP, cookie, and share
key validation. The Worker role deliberately receives no SMTP or share-token secret; it validates
only its PostgreSQL/S3 production dependencies.

The ALB routes `/api/v1/*` and `/health` to the API target group and all other paths to Web. TLS
terminates at the ALB; the database URL must still include `sslmode=require` or stronger. API and
Workers run in private subnets with an RDS security-group path and an S3 gateway endpoint or NAT.
Only the ALB is public.

Secrets are referenced from Secrets Manager in `api-task-definition.example.json`; no secret value
or static AWS access key belongs in a task definition. The task role grants only the selected S3
prefix and the required KMS operations. Web has no database, object-storage, or SMTP credentials.

Required Phase 6C resource controls before launch are RDS/S3 encryption, S3 Block Public Access and
versioning, RDS PITR of at least seven days, 30-day daily logical backups, log retention, alarms,
restore evidence, and a documented region. Do not claim that these templates alone satisfy those
operational controls.
