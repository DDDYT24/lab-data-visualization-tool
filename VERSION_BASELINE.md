# LabViz Version Baseline

- **Last reviewed:** 2026-09-10
- **Current release:** V2.1.1
- **Future backlog and completion status:** [`V2.0/TODO.md`](V2.0/TODO.md)

This file is a short version history, not a second backlog. It records what each
major version actually delivered and the boundary that version did not claim.
For future work, open `V2.0/TODO.md` first.

## Version summary

| Version | Position | Main updates | Explicit boundary |
| --- | --- | --- | --- |
| **V1.0 / early baseline** | Initial Python tool; the repository has no formal `v1.0` tag | Streamlit interactive app and CLI; CSV-oriented cleaning, duplicate/missing-value handling, basic line/scatter/histogram charts, CSV output, tests, and GitHub Actions/Codecov setup | No Next.js website, FastAPI contract, project history workflow, structured 3D contract, or publication export pipeline |
| **V1.1** | Legacy local Streamlit release | Added bounded CSV/TSV/TXT/JSON/XLSX loading, data profiling, explicit cleaning choices, seven 2D/3D chart types, 200-row previews and large-plot sampling, SQLite plot history, CLI compatibility, PNG figure export, and Windows/macOS/Linux usage documentation | Still a single-process Streamlit/CLI application; the current website and FastAPI workflow were not part of V1.1 |
| **V2.0.0** | First supported Next.js + FastAPI local self-hosted release | Added the browser workflow `Import → Inspect → Clean → Chart → Export`, real API-backed upload/progress/preview/quality/cleaning flows, seven chart types, fitting and publication controls, PNG/SVG/PDF export, local history, email-code sessions, descriptions, read-only sharing, bilingual UI/docs, one-command launchers, local SQLite/object storage, and the PostgreSQL/S3-compatible/Worker production route | Ordinary use remains local; AWS, public DNS, paid mail, Docker, external databases, and public hosting are optional maintainer paths, not user prerequisites |
| **V2.1.0** | Structured 3D and research-workflow release | Added structured-grid/surface validation with explicit X/Y/Z roles, grid-aware quality findings and chart recommendations, browser/export parity for 3D surfaces, beginner guidance, ordinary/weighted fitting with Student-t and residual-bootstrap pointwise bands, residual diagnostics and limitations, minimum `Experiment`/`ExperimentRun` lineage, UTF-8 BOM handling, Unicode-safe downloads, migration-name reconciliation, and the local privacy/offline/backup documentation set | Prediction intervals, simultaneous bands, robust regression, multiplicity correction, desktop installers, team features, billing, and AWS Phase 6C/6D live acceptance remain deferred in `TODO.md` |
| **V2.1.1** | Current security patch release | Updated Next.js, Sharp, Nanoid, Vitest and related dependencies; aligned API and website version metadata; preserved V2.1.0 behavior and local data paths | No new scientific workflow or data-format change; it does not claim a deployed cloud service |

## Release references

- V1.0 is represented by the early `Version-0`/initial implementation rather than a
  formal release tag.
- V1.1 is retained under [`V1.1/`](V1.1/) as the legacy Streamlit application.
- V2.0.0, V2.1.0, and V2.1.1 are the repository release tags.
- User-facing installation instructions live in [`README.md`](README.md) and
  [`README.zh-CN.md`](README.zh-CN.md).
- Detailed release notes are [`V2.0/RELEASE_NOTES_V2.0.md`](V2.0/RELEASE_NOTES_V2.0.md),
  [`V2.0/RELEASE_NOTES_V2.1.md`](V2.0/RELEASE_NOTES_V2.1.md), and
  [`V2.0/RELEASE_NOTES_V2.1.1.md`](V2.0/RELEASE_NOTES_V2.1.1.md).

## Maintenance rule

1. Add every future feature, defect, compatibility issue, release gate, or deferred
   decision to [`V2.0/TODO.md`](V2.0/TODO.md) first.
2. Update the checkbox, evidence, and limitation in `TODO.md` when work changes state.
   A task is not complete because an implementation note or an old report says so.
3. Update this file only when a released version changes the version baseline. Update
   `CHANGELOG.md` for the short public release note.
4. `PROJECT_PLAN.md` and `prd.md` describe architecture and product requirements;
   they do not own current completion status.
5. `PERSISTENCE_*`, `PHASE6*`, and runbook files are technical contracts or historical
   evidence. They must link back to `TODO.md` and must not create a competing backlog.
6. Superseded planning drafts should be removed instead of copied into a new status file.

When two documents disagree about whether future work is complete, `TODO.md` is the
current status authority and this file is the release-history authority.
