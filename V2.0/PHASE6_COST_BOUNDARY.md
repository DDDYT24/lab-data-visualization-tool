# Phase 6 cost boundary

**Status:** Local-only is the default. No AWS environment apply is cost-approved.

The versioned [`contracts/phase6-cost-model-v1.json`](contracts/phase6-cost-model-v1.json) records a
conservative priced floor for the accepted private AWS topology in `ap-southeast-1`. Prices were
read from the AWS public Price List files and official KMS, Secrets Manager, and Route 53 pricing
pages on 2026-08-25. They are estimates, not a quote or a spending limit.

## Result

| State | Priced monthly minimum | Important omissions |
| --- | ---: | --- |
| Foundation, ECS desired count zero | USD 160.49 | ALB LCU, logs, backups, storage requests, transfer, tax |
| API, Web, and five Workers active | USD 250.45 | Same omissions plus workload-driven use |

The six interface endpoint services placed in two Availability Zones are the largest disabled-
workload floor: 12 endpoint-AZ hours cost about USD 113.88 per 730-hour month. ECS desired count zero
does not remove the ALB, RDS instance, interface endpoints, KMS keys, secrets, or hosted zones.

The USD 30 budget is an alert and cannot enforce a hard ceiling. The accepted private topology is
therefore incompatible with a USD 30 monthly limit and with a permanent zero-cost promise.

## Approved modes

### `local-only` — current default

- Run PostgreSQL, MinIO, API, Web, Workers, and browser tests on the developer machine.
- Do not create Route 53 zones and do not run either Phase 6C apply workflow.
- Keep `COST_APPROVAL_REFERENCE` absent from GitHub environments.
- Local code and evidence preparation do not count as real Phase 6C cloud evidence.

### `aws-private` — blocked

Before any apply, obtain a fresh AWS Pricing Calculator estimate, record an explicit approval
reference in the protected environment variable `COST_APPROVAL_REFERENCE`, and retain that approval
with the release evidence. Values such as `pending`, `local-only`, or an empty string are rejected.

Changing the architecture to a single VM, public tasks, a third-party free tier, or a serverless
database would be a new architecture decision. It must not be presented as the already accepted
private Phase 6C topology without a separately reviewed contract change.

## Price sources

- [AWS KMS pricing](https://aws.amazon.com/kms/pricing/)
- [AWS Secrets Manager pricing](https://aws.amazon.com/secrets-manager/pricing/)
- [Amazon Route 53 pricing](https://aws.amazon.com/route53/pricing/)
- AWS public Price List regional offer files for AmazonVPC, AWSELB, AmazonRDS, and AmazonECS
