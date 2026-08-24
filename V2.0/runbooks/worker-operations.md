# Worker quarantine and controlled deletion

Workers remain independent services. Normal production defaults are destructive maintenance off,
dry-run on, delete off, and no `s3:DeleteObject` permission.

## Quarantine or backlog

1. Identify the exact worker, task revision, event (`worker-item-failed`, quarantine, heartbeat, or
   lease loss), oldest safe timestamp, count, and error type. Do not record object paths/content.
2. Confirm only one valid task lease and review RDS/S3/KMS health before retrying.
3. Repair the cause; release/retry only bounded selected items through the accepted worker API/code.
   Never update lease/fencing columns manually.
4. Require backlog age to return below the documented threshold and verify no new quarantines.

## Enable destructive work

1. Require a current backup, successful staging dry run, reviewed deletion inventory, and two-person
   approval tied to one release window.
2. Set the Terraform IAM switch and runtime flags in a reviewed plan. Do not grant broad bucket
   deletion or change the object prefix.
3. Run one bounded batch, inspect references/version recovery, then continue only if evidence passes.
4. Immediately revert delete permission and flags after the window; confirm a clean post-change plan.
