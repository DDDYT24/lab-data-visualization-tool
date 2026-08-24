# RDS, S3, SES, KMS, and task-exhaustion outages

## Common triage

1. Declare environment/severity; capture alarm history, AWS Health, recent deploy/config changes,
   dependency metrics, and application/worker logs in UTC.
2. Protect data first: pause risky deployments and destructive workers. Do not bypass encryption,
   public-access blocks, private networking, or IAM boundaries to restore availability.
3. Verify the failure from a role-scoped ECS diagnostic task, not a public workstation.

## RDS

- Check connectivity, free storage, connections, CPU/latency, failover events, locks, and PITR status.
- Scale or fail over only through a reviewed Terraform/managed-service action. If integrity is in
  doubt, stop writes and follow backup restoration.

## S3

- Check request errors, versioning, KMS events, endpoint policy, bucket policy, and prefix-scoped IAM.
- Recover an object by exact VersionId; never delete newer versions during diagnosis.

## SES

- Check account/configuration-set sending state, suppression, identity verification, quota,
  accepted/delivery-delay/reject/bounce/complaint events, and API task-role permission.
- Do not fall back to console/SMTP in production or remove suppression safeguards.

## KMS

- Check key state, grants/policies, rotation, CloudTrail errorCode, and the calling role. Do not swap
  to an unencrypted resource or schedule key deletion. Escalate immediately if unauthorized policy
  or grant changes are observed.

## ECS task exhaustion

- Compare desired/running count, stopped-task reason, CPU/memory, image-pull, secret, log, and ENI
  events. Stop deployment churn, restore the last completed revision, and request quota/capacity
  changes only after the failure class is known.
