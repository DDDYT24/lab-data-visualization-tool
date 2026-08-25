# Phase 6 GitHub governance preparation

**Status:** Prepared locally; no GitHub settings were changed.

The desired configuration is versioned in
[`contracts/phase6-github-governance-v1.json`](contracts/phase6-github-governance-v1.json). It keeps
the names and rules reviewable without storing variable values, email addresses, AWS identifiers,
or secret material.

## Current read-only audit — 2026-08-25

- Repository: `DDDYT24/lab-data-visualization-tool`, public, default branch `main`.
- `main` is not protected.
- The only existing deployment environment is `copilot`.
- The Phase 6 workflow commits have not reached remote `main`.

Applying branch protection now would either reference check contexts that have never run or block
the required Phase 6 push. External configuration therefore remains intentionally deferred until
the workflows are present on remote `main`.

## Required protection

After the workflows have run once on remote `main`, protect `main` with the observed check contexts
for all four `CI` jobs and the Phase 6C infrastructure `static` job. Require one current approval,
resolved conversations, linear history, admin enforcement, and prohibit force pushes and branch
deletion.

Create protected environments named exactly `staging` and `production`, restrict deployments to
`main`, require review, and prevent self-review. Production needs a reviewer who is not the release
author; that collaborator is an external prerequisite and is not invented by repository code.

## Safe preparation boundary

- Variable and secret **names** are complete in the contract.
- Values remain absent until bootstrap, DNS, and the cost gate are satisfied.
- `COST_APPROVAL_REFERENCE` must remain absent for local-only operation.
- `NOTIFICATION_EMAIL` must be stored as a GitHub secret, never as a repository variable or file.
- AWS access keys are prohibited; workflows use OIDC roles.

Read-only audit commands:

```powershell
gh api repos/DDDYT24/lab-data-visualization-tool/branches/main/protection
gh api repos/DDDYT24/lab-data-visualization-tool/environments
```

The final external application must compare the resulting GitHub settings with the versioned
contract and capture a redacted audit response. A local contract or successful API request is not
Phase 6C acceptance evidence.
