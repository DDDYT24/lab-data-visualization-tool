# Agent Acceptance Standard

> Purpose: define the repository-wide minimum standard for independent, commit-level acceptance reviews. Stage documents define *what* a phase must deliver; this document defines *how* an agent must verify it.

## 1. Applicability and instruction order

Use this standard whenever an agent is asked to accept a phase, milestone, migration, security change, release candidate, or implementation commit.

Before acting, the acceptance agent must read, in this order:

1. all applicable `AGENTS.md` files and repository instructions;
2. this file in full;
3. the current phase's design, implementation, test, migration, and operations documents;
4. the complete diff for the exact acceptance range;
5. the affected implementation, tests, configuration, and migration history.

The current phase document may add stricter requirements. It may not silently weaken this standard's evidence, safety, integrity, or read-only requirements. If instructions conflict or the acceptance range is ambiguous, return `BLOCKED` and identify the conflict.

## 2. Acceptance is independent verification

Implementation summaries, previous agent reports, screenshots, historical test output, comments, and documentation are claims to verify—not proof.

The acceptance agent must base its conclusion on:

- the repository state and exact commit relationship;
- the complete diff and reachable production code paths;
- independently rerun tests and quality checks;
- real dependencies when the phase claims real integration behavior;
- direct evidence for concurrency, durability, security, migration, and failure-recovery claims.

Do not infer that a requirement passed merely because a test has a matching name or a document says it passed.

## 3. Default permission boundary: read-only

Unless the user explicitly authorizes fixes, acceptance is strictly read-only.

The agent must not:

- edit code, tests, documentation, configuration, lockfiles, or migrations;
- create or amend commits;
- merge, rebase, reset, cherry-pick, clean, push, or alter branches;
- weaken tests, assertions, timeouts, environment checks, security controls, or production defaults;
- begin the next phase;
- inspect or manipulate files explicitly excluded by repository or phase instructions;
- destroy or reset user services, containers, databases, buckets, volumes, worktrees, or local data.

The agent may start isolated services required for verification. It must use collision-resistant test names, avoid existing user data, clean only objects it created, and stop only services it started. Test-generated data, credentials, logs, caches, and build output must not enter Git.

If a required check cannot be performed safely within this boundary, return `BLOCKED`; do not improvise a weaker substitute.

## 4. Establish the exact Git object under review

Before technical review, record and verify:

- current branch;
- `HEAD` commit;
- expected base and/or direct parent;
- ancestry or direct-parent relationship required by the phase;
- initial `git status --short`;
- commit list in the acceptance range;
- complete `--name-status` and `--stat` output for the range.

Review the complete diff, including tests, scripts, configuration, CI, dependencies, generated files, migrations, and deletions. Search for changes that may not be obvious from filenames, including fallback paths, swallowed exceptions, altered defaults, disabled validation, reduced assertions, new skips, or unrelated refactors.

If the commit, parent, branch, range, or worktree state does not match the declared acceptance object, return `BLOCKED` unless the discrepancy is explicitly resolved by the user.

## 5. Scope and invariant audit

Build a checklist from the phase document containing:

- required capabilities;
- prohibited changes;
- compatibility promises;
- API, schema, storage, security, and operational invariants;
- failure and recovery semantics;
- deferred work explicitly assigned to a later phase;
- required commands and test environments.

For each item, trace the real production call path from entry point to persistence/external side effect. Enumerate all relevant create, read, update, delete, retry, deduplication, reconciliation, background-worker, and recovery paths. A safe main path does not compensate for an unsafe secondary path.

Confirm that no unrelated phase work, secrets, `.env` files, local databases, object data, logs, caches, or test artifacts entered the commit.

## 6. Evidence requirements by risk

Use evidence proportional to the claim:

| Claim | Minimum acceptable evidence |
| --- | --- |
| Pure deterministic logic | Focused tests plus code-path inspection |
| API or serialization contract | Contract tests and diff inspection |
| Database behavior | Real target database and schema/migration inspection |
| Object storage, queue, mail, or external service behavior | Real claimed service or an explicitly approved equivalent—not a mock presented as integration |
| Concurrency, lease, lock, or fencing | Genuinely overlapping operations against the real coordination layer and final-state assertions |
| Crash recovery or idempotency | Failure injection at the claimed boundary, restart/retry, and final durable-state verification |
| Security or integrity property | Success, negative, tampering, malformed-input, authorization, and fail-closed cases |
| Performance or bounded-memory claim | Implementation proof plus a representative measurement or regression test |
| Cross-process or restart durability | Separate process/provider instance or actual restart; an in-process reconstruction is insufficient when process state is the risk |

Mocks and fault injection may isolate branches, but they cannot replace a real integration check when acceptance depends on provider, database, protocol, locking, transaction, or SDK semantics.

If a mandatory real dependency is unavailable, report `BLOCKED`. Do not reuse prior output as a substitute and do not silently skip the check.

## 7. Test-validity audit

Review new and modified tests before trusting their results. Confirm that they:

- exercise production code rather than a parallel test-only implementation;
- assert durable database and external-system state, not only status codes or mock calls;
- create real overlap for concurrency claims;
- fail when the claimed guarantee is removed;
- cover success, conflict, malformed input, provider failure, retry, and recovery as applicable;
- contain no broad exception swallowing, vacuous assertions, order dependence, unjustified sleeps, or hidden environment fallback;
- contain no newly introduced skip, `xfail`, retry, exemption, reduced assertion, or relaxed threshold without explicit phase approval;
- identify mock/unit tests honestly and do not label them real integration tests;
- isolate and clean their own data without touching pre-existing resources;
- avoid exposing credentials or sensitive data in output.

Test counts alone are not evidence. Parameterized cases must represent materially distinct assertions if cited separately.

## 8. Required verification areas

Apply every area relevant to the current phase and every area required by its documents.

### 8.1 Functional and contract behavior

- Required user and worker flows work end to end.
- Existing public API, CLI, file-format, and error contracts remain compatible unless the phase explicitly changes them.
- Alternate entry points and background jobs use the same authoritative rules.
- Invalid input and unavailable dependencies fail closed where integrity or security is involved.

### 8.2 Persistence, transactions, and external I/O

- Schema, model, repository, and migration behavior agree.
- Transaction boundaries are explicit and do not hold locks across slow external I/O unless the design specifically proves that requirement safe.
- Partial failures cannot produce invalid reachable state.
- Retries and recovery are idempotent.
- Garbage collection, deletion, reconciliation, and cleanup recheck all authoritative references before destructive action.

### 8.3 Concurrency and recovery

- Ownership, lease, generation, version, and fencing checks protect every state-changing write where applicable.
- A stale worker cannot commit after takeover.
- Crash windows before and after every durable boundary are recoverable.
- Replay does not double-count, duplicate side effects, overwrite immutable data, or skip necessary work.
- Time-based safety claims use an authoritative clock and have tested boundary behavior.

### 8.4 Security and integrity

- Authentication, authorization, signature, token, hash, scope, tenant, provider, and resource bindings are verified wherever relevant.
- Tampered, expired, replayed, malformed, cross-scope, and mismatched inputs are rejected.
- Provider or configuration failure does not silently downgrade to a less secure backend or mode.
- Credentials, signed URLs, secret material, object contents, and sensitive internal paths are not leaked.
- Content integrity does not rely on a provider identifier that is not a cryptographic content hash.

### 8.5 Pagination, cursors, and bounded work

- Pagination is genuinely bounded and does not load or sort the entire collection before slicing.
- Ordering and keyset/continuation behavior are stable at page boundaries.
- Cursors are opaque, integrity-protected when client-visible or persistently resumed, scoped, versioned, expiry-aware, and restart-safe as required.
- Insert, delete, overwrite, malformed cursor, expired cursor, scope mismatch, restart, and replay behavior match the phase contract.

### 8.6 Migrations and compatibility

- Migration lineage is correct and historical migrations outside scope are unchanged.
- Upgrade works from a realistic populated predecessor, not only an empty database.
- Data, constraints, indexes, defaults, nullability, and backfills are correct.
- Downgrade works when promised and fails closed when safe downgrade is impossible.
- Model metadata and live schema have no drift.
- Cross-version compatibility and rollback assumptions match the phase document.

### 8.7 Operational behavior

- Startup, shutdown, health checks, configuration validation, and worker behavior are verified.
- Required observability exists for failures that operators must diagnose.
- Resource cleanup is bounded and safe.
- Production-only responsibilities may be deferred only when the phase document explicitly defers them and the current implementation remains safe without them.

## 9. Verification matrix

Run the exact matrix required by repository and phase documentation. Unless clearly irrelevant, it normally includes:

- focused tests for changed components;
- real integration and end-to-end tests for affected dependencies;
- prior-phase regression suites protecting inherited invariants;
- full backend test suite;
- linters, formatting checks, and static type checks;
- frontend unit, lint, type, production-build, browser, and accessibility checks when frontend behavior may be affected;
- migration upgrade, populated upgrade, downgrade/fail-closed, re-upgrade, and schema-drift checks;
- repository hooks or CI-equivalent checks required by the project;
- `git diff --check`;
- final `git status --short`.

Record the exact command, environment/dependency used, exit result, pass/fail/skip counts, and reason for every skip. A command altered from the documented form must be reported with the reason. An unexpected skip or exemption is a failure unless the phase explicitly permits it.

## 10. Blocking rules

Return `BLOCKED` if any of the following applies:

- the Git object or acceptance range is wrong or ambiguous;
- a required document, dependency, service, credential, dataset, or environment is unavailable;
- a mandatory check fails or cannot be run independently;
- implementation violates a required capability, prohibited-change boundary, or inherited invariant;
- data integrity, immutability, authorization, isolation, concurrency, recovery, deletion safety, migration safety, or transaction-boundary guarantees are unproven or incorrect;
- a mock, emulator, SQLite path, or sequential test is used to stand in for required real provider/database/concurrency behavior without explicit authorization;
- configuration or provider failure silently falls back to a weaker mode;
- tests are weakened, silently skipped, or unable to detect loss of the claimed guarantee;
- the review requires an unapproved code or data mutation;
- cleanup would risk user-owned services or data.

Do not downgrade a correctness or safety defect to backlog merely because it is difficult to reproduce or repair.

## 11. Allowed conclusions

The first line of the final report must be exactly one of:

```text
PASS
PASS WITH NON-BLOCKING BACKLOG
BLOCKED
```

Use `PASS` only when every required capability and check passes with no acceptance-relevant remainder.

Use `PASS WITH NON-BLOCKING BACKLOG` only when all current-phase acceptance conditions pass and remaining items are explicitly optional, operational hardening, or assigned to a later phase. Each backlog item must include its owner/phase and why it cannot affect current correctness, security, data integrity, compatibility, or rollback.

Use `BLOCKED` for both confirmed defects and missing mandatory evidence. Distinguish the two in the report.

## 12. Required final report

Keep the report concise, but include enough evidence to reproduce the verdict:

1. verdict and one-paragraph rationale;
2. branch, `HEAD`, base/parent, ancestry result, and acceptance range;
3. initial and final worktree status;
4. complete changed-file list and prohibited-scope audit;
5. requirement-by-requirement result with production call paths;
6. architecture/invariant findings relevant to the phase;
7. test-validity findings;
8. exact verification commands and pass/fail/skip results;
9. real dependency versions/environment used;
10. migration and schema-drift results when applicable;
11. cleanup of services and test data created by the review;
12. blocking findings or non-blocking backlog;
13. for every failure: file, symbol/function, call path, reproduction evidence, impact, and smallest repair boundary.

Do not claim a check was run if it was inferred, inspected only, skipped, or copied from a prior report.

## 13. Stop condition

After producing the report, stop. Do not fix findings, create a commit, push, or start the next phase unless the user gives a separate explicit instruction.

## 14. Recommended phase document contents

To keep acceptance prompts short, each phase document should state:

- phase name and exact goal;
- expected branch, base/parent, and candidate commit or range;
- required and prohibited changes;
- inherited invariants;
- phase-specific high-risk scenarios and blocking conditions;
- real services and versions required;
- exact test/quality/migration matrix;
- explicitly permitted skips;
- deferred items and their destination phase;
- files or user resources the acceptance agent must not inspect or alter.

If any of these are absent and cannot be determined unambiguously from the repository, the acceptance agent must ask for clarification or return `BLOCKED` rather than guess.
