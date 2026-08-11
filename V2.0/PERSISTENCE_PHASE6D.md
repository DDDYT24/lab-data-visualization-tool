# Phase 6D Launch Qualification

**Status:** Pending Phase 6C acceptance and live staging evidence.

Phase 6D is the public-launch gate. It adds bounded product quotas, completes retention/privacy
decisions, measures capacity and security, performs disaster-recovery exercises, and accepts one
exact release candidate. Completing code without live evidence is not Phase 6D completion.

## Admission and release object

- Use the accepted Phase 6C candidate as the exact base.
- Record the release branch, base, candidate commit, container digests, migration head, Terraform
  plan identifiers, staging environment, and intended production Region.
- Every implementation unit is an independent local commit. Final acceptance is read-only and must
  return `PASS`; `PASS WITH NON-BLOCKING BACKLOG` is insufficient for public launch.
- Unexpected skips, unavailable live dependencies, unmeasured RPO/RTO, missing alarm delivery, or
  missing rollback evidence are blocking.

## 6D-1 — quotas and retention

- Configure positive per-user saved-project and total-retained-object-byte limits; no production
  unlimited/default sentinel is allowed.
- Enforce reservations and release atomically in PostgreSQL across upload, Save to Cloud,
  duplicate, export, soft delete, restore, permanent purge, retry, and Worker recovery.
- Concurrent requests cannot over-admit, double-release, or make retained data unreachable.
- Keep the accepted 50 MB upload ceiling unless a separately versioned product decision lowers it.
- Return stable API codes and bilingual UI states for approaching, reaching, and recovering quota.
- Document retention for temporary projects, saved projects, soft deletion, exports, shares,
  authentication records, limiter buckets, logs, backups, and SES feedback.

Acceptance uses real PostgreSQL overlap/restart tests and authoritative object inventory. UI-only
checks or asynchronous best-effort accounting are not sufficient.

## 6D-2 — privacy, security, and claim boundary

Produce a data-flow and data-classification inventory covering uploaded source bytes, derived
Parquet, project metadata, user email, client-identity digests, shares, exports, logs, backups, and
support/incident evidence. Define controller/processor ownership, Region, retention, deletion,
export/access-request handling, subprocessors, and incident contact.

Security qualification includes tenant/owner isolation, share token tamper/replay/scope tests,
CSRF/cookie/origin behavior, upload parser limits, malicious filename/content cases, SSRF/path
boundaries, dependency/container/IaC scans, IAM least privilege, secret exposure scans, and log/PII
redaction.

No Phase 6 result may claim HIPAA, GLP, GxP, GDPR, PIPL, medical-device, clinical, or other regulated
compliance. Such claims require a separately scoped legal/compliance assessment and organizational
controls beyond this repository.

## 6D-3 — load, capacity, and failure recovery

Run representative staging scenarios for small files, the 50 MB boundary, CSV/XLSX parsing,
cleaning, seven chart types, multi-panel analysis, publication exports, history, authentication,
sharing, and concurrent Worker maintenance. Measure Web/API latency, processing duration, memory,
CPU, database connections/locks, S3 throughput, Worker backlog age, SES delivery, autoscaling, and
cost at the documented load profile.

Failure exercises cover API/Web/Worker task loss, concurrent deployment, one-AZ loss, RDS failover,
database connection exhaustion, S3/KMS/SES unavailability, expired credentials, poison work,
stale leases, failed migration, alarm loss, and rollback. Recovery must not duplicate side effects,
change immutable revisions, leak tenant data, or delete reachable objects.

Disaster recovery restores PITR and logical backup plus required object versions into isolation and
proves RPO <= 15 minutes and RTO <= 4 hours with timestamped evidence.

## 6D-4 — release and launch acceptance

The candidate must pass:

- complete V1.1 and V2.0 CI-equivalent local gates;
- real PostgreSQL 17, real MinIO provider contract, and all migrations/schema-drift checks;
- frontend lint, type, unit, production build, Playwright, accessibility, and approved screenshot
  regressions with only explicitly documented opt-in skips;
- API/Web image build, scan, signature verification, non-root/runtime probes, and immutable digests;
- Terraform format/validate/security checks, live staging apply, no unexplained drift, alarms,
  backup/restore, load, failure, rollback, and HTTPS smoke evidence;
- repository hooks, secret scan, Markdown/JSON/YAML checks, `git diff --check`, clean tracked status,
  and prohibited-scope audit.

After independent `PASS`, merge through the protected main workflow, tag the exact release, apply
the approved production plan, run the one-shot migration, deploy by digest, switch/verify DNS,
exercise authentication/upload/processing/export/share smoke flows, and monitor the launch window.
The previous service revision, database recovery point, and rollback runbook remain available until
the launch window closes.

## Production-use conclusion

A Phase 6D `PASS` authorizes ordinary non-regulated public use of the accepted LabViz release under
the documented capacity, Region, privacy, recovery, and support boundaries. It does not add paid
billing, teams, institutional administration, regulated-data certification, or an unlimited SLA.
