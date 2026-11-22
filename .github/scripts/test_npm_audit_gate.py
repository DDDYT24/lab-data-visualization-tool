"""The npm audit gate passes only the owner-reviewed findings."""

from __future__ import annotations

import unittest

import npm_audit_gate as gate

PICOMATCH = {
    "name": "picomatch",
    "url": gate.REVIEWED_PRODUCTION_ADVISORY,
    "severity": "high",
}
MINIMATCH = {
    "name": "minimatch",
    "url": gate.REVIEWED_MINIMATCH_ADVISORY,
    "severity": "moderate",
}
OTHER = {
    "name": "lodash",
    "url": "https://github.com/advisories/GHSA-0000-0000-0000",
    "severity": "high",
}


def audit(**vulnerabilities: dict) -> dict:
    return {"metadata": {"vulnerabilities": {}}, "vulnerabilities": vulnerabilities}


def watcher_picomatch(nodes: list[str] | None = None) -> dict:
    return {
        "severity": "high",
        "via": [PICOMATCH],
        "nodes": nodes or ["node_modules/@parcel/watcher/node_modules/picomatch"],
    }


class ProductionTree(unittest.TestCase):
    def test_a_clean_audit_passes(self) -> None:
        self.assertEqual(gate.problems("production", 0, audit()), [])

    def test_picomatch_inside_parcel_watcher_passes(self) -> None:
        self.assertEqual(
            gate.problems("production", 1, audit(picomatch=watcher_picomatch())), []
        )

    def test_the_same_advisory_elsewhere_fails(self) -> None:
        found = audit(picomatch=watcher_picomatch(["node_modules/picomatch"]))
        self.assertEqual(
            gate.problems("production", 1, found),
            ["production: unreviewed finding on picomatch"],
        )

    def test_a_seeded_unreviewed_finding_fails(self) -> None:
        found = audit(
            lodash={
                "severity": "high",
                "via": [OTHER],
                "nodes": ["node_modules/lodash"],
            }
        )
        self.assertEqual(
            gate.problems("production", 1, found),
            ["production: unreviewed finding on lodash"],
        )


class DevelopmentTree(unittest.TestCase):
    def test_moderate_minimatch_on_a_reviewed_package_passes(self) -> None:
        found = audit(
            eslint={
                "severity": "moderate",
                "via": [MINIMATCH],
                "nodes": ["node_modules/eslint"],
            }
        )
        self.assertEqual(gate.problems("development", 1, found), [])

    def test_a_more_severe_minimatch_finding_fails(self) -> None:
        found = audit(
            eslint={
                "severity": "high",
                "via": [MINIMATCH],
                "nodes": ["node_modules/eslint"],
            }
        )
        self.assertEqual(
            gate.problems("development", 1, found),
            ["development: unreviewed finding on eslint"],
        )

    def test_minimatch_on_an_unreviewed_package_fails(self) -> None:
        found = audit(
            glob={
                "severity": "moderate",
                "via": [MINIMATCH],
                "nodes": ["node_modules/glob"],
            }
        )
        self.assertEqual(
            gate.problems("development", 1, found),
            ["development: unreviewed finding on glob"],
        )

    def test_a_tool_chain_that_ends_in_picomatch_passes(self) -> None:
        found = audit(
            picomatch={
                "severity": "high",
                "via": [PICOMATCH],
                "nodes": ["node_modules/picomatch"],
            },
            vitest={
                "severity": "high",
                "via": ["vite"],
                "nodes": ["node_modules/vitest"],
            },
            vite={
                "severity": "high",
                "via": ["picomatch"],
                "nodes": ["node_modules/vite"],
            },
        )
        self.assertEqual(gate.problems("development", 1, found), [])

    def test_a_chain_that_also_reaches_another_advisory_fails(self) -> None:
        found = audit(
            picomatch={
                "severity": "high",
                "via": [PICOMATCH],
                "nodes": ["node_modules/picomatch"],
            },
            vite={
                "severity": "high",
                "via": ["picomatch", OTHER],
                "nodes": ["node_modules/vite"],
            },
        )
        self.assertEqual(
            gate.problems("development", 1, found),
            ["development: unreviewed finding on vite"],
        )

    def test_a_chain_that_cannot_be_traced_fails(self) -> None:
        found = audit(
            picomatch={
                "severity": "high",
                "via": [PICOMATCH],
                "nodes": ["node_modules/picomatch"],
            },
            vite={
                "severity": "high",
                "via": ["picomatch", "absent-package"],
                "nodes": ["node_modules/vite"],
            },
        )
        self.assertEqual(
            gate.problems("development", 1, found),
            ["development: unreviewed finding on vite"],
        )

    def test_a_cycle_in_the_chain_ends(self) -> None:
        found = audit(
            a={"severity": "high", "via": ["b"], "nodes": ["node_modules/a"]},
            b={
                "severity": "high",
                "via": ["a", PICOMATCH],
                "nodes": ["node_modules/b"],
            },
        )
        self.assertEqual(gate.problems("development", 1, found), [])


class IncompleteAudit(unittest.TestCase):
    def test_an_audit_that_did_not_complete_fails(self) -> None:
        for exit_code, value in (
            (2, audit()),
            (1, None),
            (1, {"vulnerabilities": {}}),
            (0, []),
        ):
            with self.subTest(exit_code=exit_code, value=value):
                self.assertEqual(len(gate.problems("production", exit_code, value)), 1)


if __name__ == "__main__":
    unittest.main()
