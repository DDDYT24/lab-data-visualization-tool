# Phase 6 AWS Dependency Matrix

**Status:** Active execution boundary from 2026-08-11.

This matrix allows useful Phase 6 work to continue while AWS CLI authentication is unavailable
without weakening the ordered phase gates or presenting local substitutes as AWS evidence.

## Work that can be completed locally

| Unit | Local completion | AWS evidence still required |
| --- | --- | --- |
| 6B-1 trusted client identity | Production parser, trusted-proxy configuration, keyed digest, API integration, and negative tests | Recheck the configured header chain through the real staging ALB |
| 6B-2 atomic abuse limits | Migration `0009`, PostgreSQL database-time transactions, SQLite parity, cleanup, and real concurrent PostgreSQL tests | None |
| 6B-3 SES delivery | SES v2 adapter, fail-closed settings, task-role contract, message/tag redaction, failure compensation, and deterministic client-boundary tests | Real SES send, delivery, bounce, complaint, suppression, IAM, and alarm evidence in `ap-southeast-1` |
| 6B-4 project descriptions | PostgreSQL/SQLite API contract, immutable revision behavior, bilingual UI, Vitest, and Playwright | None |

The local implementation order is 6B-1, 6B-2, 6B-4, then the offline portion of 6B-3. This
deliberately keeps the three fully local units moving before the AWS-dependent unit. Each unit is
an independent commit and is verified before the next begins.

## Work that must not start yet

- Phase 6B cannot receive a final `PASS` until the real SES and staging ALB evidence exists.
- Phase 6C cannot start because its admission contract requires an accepted Phase 6B range, and
  Phase 6B explicitly prohibits adding Terraform resources.
- Phase 6D cannot start because its admission contract requires accepted Phase 6C live-staging
  evidence. Although some 6D code could run locally, implementing it early would bypass the
  ordered architecture and launch gates.

AWS CLI authentication failure is therefore an evidence blocker, not permission to use mocks as
integration proof, weaken production settings, add static credentials, or claim Phase 6 complete.
Every locally completed unit updates its status in
[`PERSISTENCE_PHASE6B.md`](PERSISTENCE_PHASE6B.md). Because a commit cannot contain its own future
hash, the final local acceptance report records the exact immutable commit for every unit. The
remaining AWS evidence will be resumed from that documented boundary without rewriting accepted
local commits.
