# Phase 6 AWS Dependency Matrix

**Status:** Phase 6B application implementation accepted on 2026-08-12; the stage boundary was
corrected on 2026-08-17 so Phase 6C may create the infrastructure needed for real AWS evidence.

This matrix separates application acceptance from cloud-environment acceptance. It removes the
former circular dependency without weakening the ordered phase gates or presenting local
substitutes as AWS evidence.

The dependency direction and transferred-evidence ownership are also encoded in the versioned
[`contracts/phase6-gates-v1.json`](contracts/phase6-gates-v1.json) contract. The architecture test
rejects cycles, broken unlock relationships, changed evidence ownership, or missing phase-document
references.

## Work that can be completed locally

| Unit | Phase 6B application acceptance | Transferred Phase 6C evidence |
| --- | --- | --- |
| 6B-1 trusted client identity | Production parser, trusted-proxy configuration, keyed digest, API integration, and negative tests | Recheck the configured header chain through the real staging ALB |
| 6B-2 atomic abuse limits | Migration `0009`, PostgreSQL database-time transactions, SQLite parity, cleanup, and real concurrent PostgreSQL tests | None |
| 6B-3 SES delivery | SES v2 adapter, fail-closed settings, task-role contract, message/tag redaction, failure compensation, and deterministic client-boundary tests | Real SES send, delivery, bounce, complaint, suppression, IAM, and alarm evidence in `ap-southeast-1` |
| 6B-4 project descriptions | PostgreSQL/SQLite API contract, immutable revision behavior, bilingual UI, Vitest, and Playwright | None |

The application implementation order was 6B-1, 6B-2, 6B-4, then the SES adapter portion of 6B-3. This
deliberately keeps the three fully local units moving before the AWS-dependent unit. Each unit is
an independent commit and is verified before the next begins.

## Corrected phase boundary

- Phase 6B receives `PASS` for its application implementation and deterministic provider-boundary
  tests. It makes no claim that AWS resources exist or that email has been delivered.
- Phase 6C may start from the accepted Phase 6B application range and owns Terraform, staging
  deployment, and all real SES, deployed-IAM, ALB, and CloudWatch evidence.
- Phase 6C cannot receive `PASS` while any transferred evidence is missing or failed.
- Phase 6D cannot start because its admission contract requires accepted Phase 6C live-staging
  evidence. Although some 6D code could run locally, implementing it before Phase 6C acceptance
  would bypass the ordered architecture and launch gates.

AWS CLI authentication failure therefore blocks Phase 6C planning/apply evidence and Phase 6C
acceptance. It is not permission to use mocks as integration proof, weaken production settings,
add static credentials, or claim Phase 6 complete.
Every completed application unit updates its status in
[`PERSISTENCE_PHASE6B.md`](PERSISTENCE_PHASE6B.md). Because a commit cannot contain its own future
hash, the application acceptance report records the exact immutable commit for every unit. The
transferred AWS evidence will be produced only after the Phase 6C staging environment exists,
without rewriting accepted application commits.
