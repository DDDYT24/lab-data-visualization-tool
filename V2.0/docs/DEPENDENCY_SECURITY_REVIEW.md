# V2.3 web dependency follow-up — 2026-11-02

The V2.3 Windows RC pipeline stopped at its production npm audit on PR #7. Advisories
published after V2.3.0-rc.1 affect the locked web dependencies. The audits below use the
official npm registry, because the default mirror does not implement the audit endpoint.

## Updates

- Next.js and eslint-config-next: 16.3.8 → 16.4.3. Next.js 16.0.0–16.4.0 is affected by
  unauthenticated remote code execution in the Image Optimization API with SVG files
  ([GHSA-2464-fhr4-w9xw](https://github.com/advisories/GHSA-2464-fhr4-w9xw)) and in `use cache`
  functions returning typed arrays ([GHSA-673q-mjm6-r5xr](https://github.com/advisories/GHSA-673q-mjm6-r5xr)),
  and by a Server Actions denial of service ([GHSA-p3m5-59mp-jv5r](https://github.com/advisories/GHSA-p3m5-59mp-jv5r)).
- zod: 4.4.3 → 4.6.7, for memory exhaustion when validating oversized constrained
  collections ([GHSA-mrhx-c7xc-m3w5](https://github.com/advisories/GHSA-mrhx-c7xc-m3w5)).
- vitest: 4.1.11 → 4.1.12, and vite within its existing range to 8.3.5, for the
  `@vitest/mocker` and `vite` development advisories.

Lock entries rewritten by this update keep the `registry.npmjs.org` source they had before.

## Result

The production audit reports zero findings. The full audit reports 16 moderate findings and
no high or critical ones, all through `minimatch` below 11.0.1
([GHSA-g3hv-75wg-m249](https://github.com/advisories/GHSA-g3hv-75wg-m249)) in the ESLint
plugin chain. No compatible fix is published: ESLint 10.14.0 still depends on
`minimatch ^10.2.6`, and the plugins use the 3.x API, so a forced override would break linting.
The earlier `braces`, `micromatch` and `fast-glob` findings no longer appear.

Disposition, accepted by the owner: the `minimatch` chain runs only while linting, so the
development-audit gate in `packaging/windows-v23/build_rc.py` now allows these 16 packages and
drops the `braces` chain it no longer needs. The gate is narrower than the name list it replaces:
a package passes only while every finding on it stays at moderate or below and traces to the
`minimatch` advisory alone, so a new or more severe advisory on the same package still stops
the build. Reassess when ESLint or its plugins move to minimatch 11.0.1 or later.

V2.2.0 on `main` locks Next.js 16.3.6 and is in the affected range as well.

# V2.2 production dependency review — 2026-10-07

This review covers the local V2.2 candidate's locked production dependencies. It does not
certify the application as free of vulnerabilities or replace installed/offline acceptance.
Raw and follow-up reports are retained under `outputs/v22-full-audit-20261007`.

## Updates

- Next.js and eslint-config-next: 16.3.4 → 16.3.6.
- Next.js Sharp override: 0.35.4 → 0.35.5.
- source-map-js: explicit 1.2.2 override.
- urllib3: 2.7.0 → 2.8.0, constrained in the API requirements and both lock files.

The default npm mirror does not implement its audit endpoint. The follow-up production
audit uses the official npm registry. Its post-update result contains zero known findings.
Development-tool dependencies and public distribution signing are separate review scopes.

The full npm audit also retains seven high findings in development-tool dependency paths
(@next/eslint-plugin-next, eslint-config-next, brace-expansion, braces, fast-glob, micromatch
and undici). Those findings are not counted as production findings and have not been fixed by
this production patch. They remain a tooling follow-up; the full raw report is retained as
`npm-audit-all-after-update.json`. Do not describe this result as a zero-finding audit of every
dependency or as proof that all compiled/transitive code has no vulnerabilities.

References: [Next.js advisory](https://github.com/advisories/GHSA-vcvr-r3jv-pc5j),
[Sharp advisory](https://github.com/advisories/GHSA-wq5f-xc86-pv6w),
[source-map-js advisory](https://github.com/advisories/GHSA-68fv-2mgg-jv7q),
[urllib3 advisory](https://github.com/urllib3/urllib3/security/advisories/GHSA-vxq7-64xx-v4gw).
These package matches do not establish that LabViz exposes each advisory's attack path:
the source does not use next/og ImageResponse and this candidate targets Windows.

## Reviewed Apache Arrow finding

The raw pip-audit report returns PYSEC-2026-113 / CVE-2026-25087 / GHSA-rgxp-2hwp-jwgg
for PyArrow 22.0.0; the service returns duplicate entries for the same advisory.
The [published advisory](https://github.com/advisories/GHSA-rgxp-2hwp-jwgg) explicitly states
that the affected C++ IPC pre-buffering method is not exposed in Python bindings and that
those bindings are not vulnerable. LabViz uses the Python binding for internal Parquet
artifacts and does not ingest Arrow IPC files or call RecordBatchFileReader::PreBufferMetadata.

Disposition: not applicable to the current Python/Parquet implementation. PyArrow remains
within the existing <23 compatibility bound. The filtered follow-up audit excludes only
PYSEC-2026-113; its raw report remains available. Reassess this disposition if a native C++
IPC reader or pre-buffering path is introduced, or if the advisory's scope changes.

## Evidence scope

Functional, integration and packaged-browser checks must be repeated after dependency changes.
Unsigned intermediate packages built before these updates are superseded and are not release
artifacts. The final release source/version freeze and fresh artifact checks are recorded in Release assets.
The owner reports the preceding candidate accepted offline; final-hash manual attestation is
not independently observed. Real account isolation is untested and excluded from release claims.
Unsigned distribution and the actual local scanner result are disclosed.
