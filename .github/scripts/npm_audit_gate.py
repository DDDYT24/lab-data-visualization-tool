"""Fail when `npm audit` of the web package reports a finding that no owner review covers.

Usage: python .github/scripts/npm_audit_gate.py [--web V2.0/web]

It audits the production tree (`--omit=dev`) and the whole tree, and passes only findings that the
reviews below cover. Network access to the npm registry is required. The reviews were moved here
unchanged from the retired V2.3 RC build (`V2.0/packaging/windows-v23/build_rc.py`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = "--registry=https://registry.npmjs.org"

# Reviewed 2026-11-02 (V2.0/docs/DEPENDENCY_SECURITY_REVIEW.md): the ESLint plugin chain
# depends on minimatch below 11.0.1 (GHSA-g3hv-75wg-m249) and no compatible fix is published.
# It runs only while linting. A package is allowed through only while every finding on it
# stays at moderate or below and traces to that one root advisory, so a new or more severe
# advisory on the same package still fails.
REVIEWED_DEVELOPMENT_PACKAGES = frozenset(
    {
        "@eslint-community/eslint-utils",
        "@eslint/config-array",
        "@eslint/eslintrc",
        "@next/eslint-plugin-next",
        "@typescript-eslint/eslint-plugin",
        "@typescript-eslint/parser",
        "@typescript-eslint/type-utils",
        "@typescript-eslint/typescript-estree",
        "@typescript-eslint/utils",
        "eslint",
        "eslint-config-next",
        "eslint-plugin-import",
        "eslint-plugin-jsx-a11y",
        "eslint-plugin-react",
        "minimatch",
        "typescript-eslint",
    }
)
REVIEWED_DEVELOPMENT_ADVISORIES = frozenset({"minimatch"})
REVIEWED_DEVELOPMENT_SEVERITIES = frozenset({"info", "low", "moderate"})

# Reviewed 2026-11-17, owner approved: GHSA-fc5q-39xc-vjpf (picomatch ReDoS through repeated
# extglobs, high, every release up to 4.0.7, no fixed release yet). It reaches the production
# tree only as @parcel/watcher's own picomatch; next-intl loads @parcel/watcher only in its
# message extractor (dist/*/extractor/source/SourceFileWatcher.js), which LabViz does not use,
# so no glob reaches it. A finding passes only on that advisory and inside that one path; the
# same advisory anywhere else, or any other advisory, still fails. Remove this once a fixed
# picomatch is released and installed.
REVIEWED_PRODUCTION_ADVISORY = "https://github.com/advisories/GHSA-fc5q-39xc-vjpf"
REVIEWED_PRODUCTION_PATH = "node_modules/@parcel/watcher"

# Reviewed 2026-11-18, owner approved: the same picomatch advisory reaches the development tree
# through the test and lint tools (vitest, vite, tinyglobby, fdir, micromatch, fast-glob and the
# ESLint plugins). They run only in development and CI, are not shipped, and match only globs
# from the project's own configuration. A development finding passes on this review only when
# every advisory it traces to is that picomatch advisory or the reviewed minimatch one above
# (and then only on the reviewed packages); anything else, or a chain that cannot be traced,
# still fails. Remove with the production exception once picomatch is fixed.
REVIEWED_MINIMATCH_ADVISORY = "https://github.com/advisories/GHSA-g3hv-75wg-m249"


def root_advisories(
    audit: dict, name: str, seen: frozenset[str] = frozenset()
) -> set[str]:
    """The advisory URLs a finding traces to through the vulnerable packages it depends on."""
    findings = audit.get("vulnerabilities", {})
    roots: set[str] = set()
    for source in findings.get(name, {}).get("via", []):
        if isinstance(source, dict):
            roots.add(str(source.get("url")))
        elif source not in findings:
            roots.add("untraced:" + str(source))
        elif source not in seen:
            roots |= root_advisories(audit, source, seen | {name})
    return roots


def reviewed_picomatch_development_finding(audit: dict, name: str) -> bool:
    roots = root_advisories(audit, name)
    return (
        REVIEWED_PRODUCTION_ADVISORY in roots
        and roots <= {REVIEWED_PRODUCTION_ADVISORY, REVIEWED_MINIMATCH_ADVISORY}
        and (
            REVIEWED_MINIMATCH_ADVISORY not in roots
            or name in REVIEWED_DEVELOPMENT_PACKAGES
        )
    )


def reviewed_production_finding(finding: dict) -> bool:
    nodes = finding.get("nodes", [])
    return (
        bool(nodes)
        and all(
            node == REVIEWED_PRODUCTION_PATH
            or node.startswith(REVIEWED_PRODUCTION_PATH + "/")
            for node in nodes
        )
        and all(
            source.get("url") == REVIEWED_PRODUCTION_ADVISORY
            if isinstance(source, dict)
            else source == "picomatch"
            for source in finding.get("via", [])
        )
    )


def unreviewed_production_findings(audit: dict) -> list[str]:
    """Every production audit finding that the review above does not cover."""
    return [
        name
        for name, finding in sorted(audit.get("vulnerabilities", {}).items())
        if not reviewed_production_finding(finding)
    ]


def unreviewed_development_findings(audit: dict) -> list[str]:
    """Every development audit finding that the reviews above do not cover."""
    unreviewed = []
    for name, finding in sorted(audit.get("vulnerabilities", {}).items()):
        if reviewed_production_finding(
            finding
        ) or reviewed_picomatch_development_finding(audit, name):
            continue
        advisories = {
            source.get("name")
            for source in finding.get("via", [])
            if isinstance(source, dict)
        }
        if (
            name not in REVIEWED_DEVELOPMENT_PACKAGES
            or finding.get("severity") not in REVIEWED_DEVELOPMENT_SEVERITIES
            or not advisories <= REVIEWED_DEVELOPMENT_ADVISORIES
        ):
            unreviewed.append(name)
    return unreviewed


def problems(tree: str, exit_code: int, audit: object) -> list[str]:
    """Why one audit run fails the gate; empty when it passes."""
    # npm audit exits 1 when it lists findings; anything else means the audit did not complete.
    if (
        exit_code not in (0, 1)
        or not isinstance(audit, dict)
        or not isinstance(audit.get("metadata"), dict)
    ):
        return [f"the {tree} audit did not complete (exit {exit_code})"]
    check = (
        unreviewed_production_findings
        if tree == "production"
        else unreviewed_development_findings
    )
    return [f"{tree}: unreviewed finding on {name}" for name in check(audit)]


def run_audit(web: Path, *extra: str) -> tuple[int, object]:
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm was not found on PATH")
    result = subprocess.run(
        [npm, "audit", REGISTRY, "--json", *extra],
        cwd=web,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return result.returncode, json.loads(result.stdout)
    except json.JSONDecodeError:
        return result.returncode, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", type=Path, default=ROOT / "V2.0" / "web")
    args = parser.parse_args(argv)
    found = problems("production", *run_audit(args.web, "--omit=dev"))
    found += problems("development", *run_audit(args.web))
    for line in found:
        print(line, file=sys.stderr)
    print(
        "npm audit gate: " + ("failed" if found else "passed (only reviewed findings)")
    )
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
