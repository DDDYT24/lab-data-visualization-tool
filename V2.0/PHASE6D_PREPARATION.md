# Phase 6D preparation package

**Status:** Draft complete; Phase 6D entry remains blocked by Phase 6C acceptance.

This package makes the next-stage decisions and evidence fields reviewable before DNS and staging
exist. It does not implement product quotas, perform a legal review, measure staging capacity, prove
RPO/RTO, or authorize launch.

## Prepared contracts

- [`phase6d-preparation-v1.json`](contracts/phase6d-preparation-v1.json) owns the admission boundary
  and prohibited claims.
- [`phase6d-quota-retention-v1.json`](contracts/phase6d-quota-retention-v1.json) separates existing
  enforced limits from candidate per-user quotas and lists atomic accounting evidence.
- [`phase6d-data-inventory-v1.json`](contracts/phase6d-data-inventory-v1.json) inventories user,
  derived, identifier, security, log, backup, and incident data without claiming legal compliance.
- [`phase6d-load-recovery-v1.json`](contracts/phase6d-load-recovery-v1.json) defines local preflight
  and live staging profiles, failure exercises, measurements, and recovery evidence.

## Candidate decisions, not implementation

The draft proposes 25 saved projects and 2 GiB of retained object bytes per user. Those positive
values are deliberately labelled `requires-phase6d-product-approval`; no API, database reservation,
or UI enforcement has been added. The accepted 50 MB upload ceiling and existing lifecycle values
remain unchanged.

Local concurrency tests can find functional defects, but they cannot establish AWS capacity,
deployed IAM, SES delivery, one-AZ behavior, RDS failover, alarm delivery, backup restoration, or
cost. The local profile therefore has `acceptanceAuthority=false`.

## Admission checklist

Phase 6D work may begin only after all Phase 6C live evidence passes for one exact staging candidate.
At admission, replace candidate decisions with approved versioned contracts, implement quotas in
PostgreSQL and the bilingual UI, run the staging profiles, execute isolated recovery, and collect
the listed evidence. Until then, [`PERSISTENCE_PHASE6D.md`](PERSISTENCE_PHASE6D.md) remains the
authoritative launch contract and its status stays pending.
