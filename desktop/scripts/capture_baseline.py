"""Capture source boundaries and deterministic synthetic desktop fixtures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

BASE = "b587db848ba5f1930657d5a316e0160c71864ead"


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def capture(root: Path) -> dict:
    names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    files = {}
    for name in sorted(set(filter(None, names))):
        path = root / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "root": str(root),
        "head": git(root, "rev-parse", "HEAD").decode().strip(),
        "branch": git(root, "branch", "--show-current").decode().strip(),
        "status": git(root, "status", "--porcelain=v1").decode(),
        "indexSha256": hashlib.sha256(git(root, "ls-files", "--stage", "-z")).hexdigest(),
        "sourceSha256": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protect", type=Path, action="append", default=[])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    roots = [capture(root.resolve()) for root in args.protect]
    (out / "protected-sources.json").write_text(json.dumps(roots, indent=2), encoding="utf-8")
    fixture = out / "fixtures"
    fixture.mkdir()
    fields = ["sample", "value", "unit", "time", "pair", "group"]
    rows = []
    for n in range(1, 7):
        rows.extend([[f"s{n}", n, "mV", "before", f"p{n}", "A"], [f"s{n}", n + 2, "mV", "after", f"p{n}", "A"]])
    for suffix, delimiter in [("csv", ","), ("tsv", "\t")]:
        with (fixture / f"desktop-small.{suffix}").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
            writer.writerow(fields)
            writer.writerows(rows)
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Research"
    sheet.append(fields)
    for row in rows:
        sheet.append(row)
    workbook.save(fixture / "desktop-small.xlsx")
    chinese = fixture / "中文 路径"
    chinese.mkdir()
    (chinese / "科研数据.csv").write_bytes((fixture / "desktop-small.csv").read_bytes())
    with (fixture / "desktop-100k.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        for n in range(50000):
            group = "A" if n % 2 == 0 else "B"
            writer.writerow([f"s{n}", n % 1000, "mV", "before", f"p{n}", group])
            writer.writerow([f"s{n}", n % 1000 + 2, "mV", "after", f"p{n}", group])
    record = {
        "format": "labviz-desktop-baseline-v1",
        "sourceCommit": BASE,
        "releaseTag": "v2.3.0-rc.1",
        "installerBytes": 132271120,
        "installerSha256": "57f0e5e2c88c4e9f82ab3b56e3f84ecc3e577d3ce16a77f46f1e942da3e95cbd",
        "smallExpected": {"observations": 12, "experimentalUnits": 6, "mean": 4.5, "pairedDeltas": [2] * 6},
        "largeExpected": {"observations": 100000, "experimentalUnits": 50000, "mean": 500.5, "pairedDelta": 2},
        "fixtureSha256": {p.relative_to(fixture).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture.rglob("*") if p.is_file()},
        "protectedRootCount": len(roots),
        "realUserDataCopied": False,
    }
    (out / "baseline.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "protectedRoots": len(roots), "fixtures": len(record["fixtureSha256"])}))


if __name__ == "__main__":
    main()
