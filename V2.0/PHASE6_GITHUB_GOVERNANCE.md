# Phase 6 GitHub governance preparation

**Status:** Phase 6 workflow definitions are on remote `main`; GitHub settings and live AWS/Phase
6C acceptance remain pending.

**Last reviewed:** 2026-09-10

**Current backlog and completion status:** [`TODO.md`](TODO.md)

The desired configuration is versioned in
[`contracts/phase6-github-governance-v1.json`](contracts/phase6-github-governance-v1.json). It keeps
the names and rules reviewable without storing variable values, email addresses, AWS identifiers,
or secret material.

## Current read-only audit — 2026-09-10

- Repository: `DDDYT24/lab-data-visualization-tool`, public, default branch `main`.
- The last authenticated settings audit on 2026-08-25 recorded `main` as unprotected and the
  only existing deployment environment as `copilot`; re-run the read-only commands below before
  applying any governance change.
- The Phase 6 workflow definitions are now present on remote `main`.
- The generic CI and Phase 6C infrastructure static workflow have run successfully. The foundation
  apply and release workflows have no runs yet because their external inputs are not configured.

Branch protection, deployment environments, secrets, DNS, and AWS values remain external
configuration. Applying them is intentionally deferred until the account-owned prerequisites and
the current observed check contexts are reviewed.

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
