# LabViz V2.2.0 — Windows local release

Release date: 2026-10-07. Tag: `v2.2.0`. Supported scope: Windows x64, one ordinary
Windows profile. The public installer is unsigned. Cloud collaboration remains deferred.

## Delivered

- Ready-to-run per-user Windows installer bundling CPython 3.13.7, Node.js 24.17.0 and
  the Next.js/FastAPI application. No runtime dependency download or email login.
- Durable SQLite/local-object history for real imports; saved figure previews/downloads,
  processed-data review, explicit deletion, migration/backup and maintenance shortcuts.
- Seven synthetic teaching examples, bilingual Help/About, themes and chart data tables.
- Prediction intervals, Working–Hotelling simultaneous mean bands and linear Huber fitting
  within the documented statistics contract.
- Categorical-X/single-response selection repair for group/distribution data (04/05),
  shared histogram intervals and numeric axis limits, actual PNG/SVG/PDF content checks.
- WebKit navigation repair and reviewed production dependency patches.

## Acceptance and source freeze

The preceding full audit passed 271 API tests, 60 Vitest tests, 65 browser regressions,
ten distinct packaged live workflows, 39 fixture cases and 100 chaos cases. Lifecycle
tests used synthetic data and a test AppId. Original failures and corrections remain in
the P7 report; passing reruns do not erase them.

The owner reported completion of the preceding candidate's clean/offline/restart acceptance
and authorized publication. This is owner testimony, not independent observation or a
verification JSON attached to the final artifact. Real two-Windows-account file/loopback
isolation is untested and excluded from the supported release claim. Authentic V2.1.1
installer upgrade/rollback acceptance was waived by the owner.

The final source aligns API, Web, diagnostics, package contract and installer at 2.2.0.
Publication changes are version/contract metadata, documentation and CI browser provisioning;
scientific processing is unchanged from the preceding audited candidate. The installer is
rebuilt from the frozen source, not a renamed development EXE. Release assets provide
`release-provenance.json`, `release-verification.json`, `VERIFICATION.md` and `SHA256SUMS.txt`
with its exact commit, build ID, artifact identity and fresh checks. Do not attribute the
owner's preceding-package manual tests to a newly measured final EXE hash.

## Downloads and use

[Release and installer](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0).
Install with the default path and open the LabViz shortcut; no sign-in is required.
Updates preserve `%LOCALAPPDATA%\LabViz\data` by default. Stop the application and back
up the whole data directory first. Source checkout history uses a different directory.
See the [English README](../README.md).

## Known limitations and distribution policy

- Unsigned: no certificate or universal SmartScreen/antivirus approval claim. Exact local
  scan outcome is recorded in the release verification asset; use the published SHA-256.
- Single Windows profile only; no verified concurrent multi-account privacy guarantee.
- Arbitrarily long installation paths and all sleep/resume policies are not validated.
- Browser/Python renderers need not be pixel-identical; scientific content and actual
  downloaded geometry are checked. Review publisher requirements before using figures.
- Seven development-tool dependency findings remain documented. The reviewed Arrow
  exception is specific to Python/Parquet, not a blanket vulnerability exemption.
- Native macOS/Linux packages, Unix launcher parity, Cloud sync/team/OAuth and hosted
  deployment are deferred. PostgreSQL/S3 integration source is retained, not publicly hosted.

See [P7 evidence](docs/V2.2_P7_TEST_REPORT.md), [dependency review](docs/DEPENDENCY_SECURITY_REVIEW.md),
[statistics](docs/STATISTICS_CONTRACT_V2.2.md) and [current backlog](TODO.md).
