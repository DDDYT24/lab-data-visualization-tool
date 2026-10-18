# LabViz V2.3.0-rc.1

This Windows preview adds a local research workspace and an optional overview interface. V2.2.0 remains the stable release while the RC is evaluated.

## Changes

- Import research tables with explicit sample, experimental-unit, group, time-point, pair, and unit mappings. Preview preparation and exclusions before saving a revision.
- Create grouped raw-point, paired-change, and violin charts, then arrange up to four independent 2D panels in a saved figure layout.
- Reuse workflow templates and freeze research outputs with source data, calculation details, revision identifiers, and PNG, SVG, and PDF files.
- Switch between classic and overview interfaces while retaining workspace state. Classic remains the default.
- Install V2.3 beside V2.2 with separate data, browser identities, logs, backups, and maintenance commands.
- Preserve the released histogram interval and single-variable defaults. Correct disabled grid exports in the legacy and composed renderers.

## Installation and data

The unsigned installer bundles CPython 3.13.16, Node.js 24.22.0, scientific libraries, fonts, and synthetic examples. Verify its SHA256 before installation. Keep Windows security controls enabled.

V2.3 starts with a separate profile. Importing V2.2 data is an explicit copy into an independent profile. Uninstall keeps data. See [installation, backup, import, and recovery](docs/INSTALL_V2.3.md).

## Verification and remaining limits

The release's verification and provenance attachments identify the exact source commit, Web build, payload, installer hash, and completed checks. Local source results, bundled-runtime results, and installed-UI results are recorded separately.

Actual clean-machine use, a physically disconnected complete workflow, desktop 200% zoom and assistive speech, physical iPhone use, and a researcher pilot with 3 to 5 participants remain pending. Browser emulation and automated accessibility checks do not complete those observations. This RC is available for evaluation; formal stable acceptance is incomplete.

The official npm audit reports five high-severity findings in the development lint chain, rooted in `braces` through `micromatch` and `fast-glob`. The audited production tree has no reported findings. The published `braces` version was still 3.0.3 at review, and the suggested automatic fix downgraded the Next.js lint configuration. The locked versions remain in place pending a compatible fix. See [GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm).

Native macOS/Linux installers and hosted collaboration are outside this RC. Existing scientific limits and the four-panel bound remain in effect.
