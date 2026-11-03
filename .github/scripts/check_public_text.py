"""Reject restricted terms in text that is published with the repository.

The term list is kept out of the repository on purpose. CI reads it from the
RESTRICTED_TERMS secret (comma- or newline-separated); a local checkout reads it
from <git common dir>/info/restricted-terms, one term per line. Findings name the
location and the term's number only, so the terms never reach a public log.

Terms of three letters or fewer match whole words, also inside camelCase and
snake_case identifiers; longer terms match anywhere, ignoring case. Lockfile contents
are third-party metadata (package authors, base64 hashes) and are not scanned; their
paths are.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)")
LOCKFILE = re.compile(
    r"(^|/)(package-lock\.json|Cargo\.lock|\.terraform\.lock\.hcl|[^/]*\.lock\.txt)$"
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def load_terms() -> list[str]:
    raw = os.environ.get("RESTRICTED_TERMS", "")
    if not raw.strip():
        path = (
            Path(git("rev-parse", "--git-common-dir").strip())
            / "info"
            / "restricted-terms"
        )
        raw = path.read_text(encoding="utf-8") if path.is_file() else ""
    return [t.strip().lower() for t in re.split(r"[,\n]", raw) if t.strip()]


def hits(text: str, terms: list[str]) -> list[int]:
    lowered = text.lower()
    words = {w.lower() for w in WORD.findall(text)}
    return [
        number
        for number, term in enumerate(terms, 1)
        if (term in words if len(term) <= 3 else term in lowered)
    ]


def added_lines(diff: str) -> Iterator[tuple[str, str]]:
    """Yield (location, text) for every added path and line of a unified diff."""
    path, line = "", 0
    for row in diff.splitlines():
        if row.startswith("+++ "):
            path = row[6:] if row.startswith("+++ b/") else ""
            if path:
                yield f"{path} (path)", path
        elif match := HUNK.match(row):
            line = int(match.group(1))
        elif row.startswith("+") and path and not LOCKFILE.search(path):
            yield f"{path}:{line}", row[1:]
            line += 1


def commits(base: str, head: str) -> Iterator[tuple[str, str]]:
    for sha in git("rev-list", f"{base}..{head}").split():
        short = sha[:9]
        yield f"commit {short} message", git("log", "-1", "--format=%B", sha)
        yield (
            f"commit {short} author/committer",
            git("log", "-1", "--format=%an %ae %cn %ce", sha),
        )


def scan(items: Iterable[tuple[str, str]], terms: list[str]) -> int:
    found = 0
    for location, text in items:
        for number in hits(text, terms):
            print(f"{location}: restricted term #{number}")
            found += 1
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged", action="store_true", help="staged paths and added lines"
    )
    parser.add_argument(
        "--message", type=Path, help="commit message file, plus the current branch"
    )
    parser.add_argument(
        "--range",
        nargs=2,
        metavar=("BASE", "HEAD"),
        help="commits, identities and changes",
    )
    parser.add_argument(
        "--env", action="append", default=[], help="environment variable holding text"
    )
    parser.add_argument(
        "--require-terms", action="store_true", help="fail when no term list is set"
    )
    args = parser.parse_args()

    terms = load_terms()
    if not terms:
        print("No restricted term list is configured.", file=sys.stderr)
        return 2 if args.require_terms else 0

    items: list[tuple[str, str]] = []
    if args.staged:
        items += added_lines(
            git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff")
        )
    if args.message:
        items.append(("commit message", args.message.read_text(encoding="utf-8")))
        items.append(("branch name", git("rev-parse", "--abbrev-ref", "HEAD")))
    if args.range:
        base, head = args.range
        items += added_lines(
            git("diff", "-U0", "--no-color", "--no-ext-diff", f"{base}...{head}")
        )
        items += commits(base, head)
    items += [(name, os.environ.get(name, "")) for name in args.env]

    found = scan(items, terms)
    if found:
        print(
            f"{found} restricted term(s) found; see CONTRIBUTING.md.", file=sys.stderr
        )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
