"""Check actual trial exports and scientific answers without changing the profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    trial = args.trial.resolve()
    report = read(trial / "workflow-full.json")
    assert report["passed"] and len(report["checks"]) == 15
    assert not report["pageErrors"] and not report["externalRequests"]
    results = []
    for record in report["exports"]:
        target = Path(record["file"])
        assert hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"]
    for family in ("Raw", "Paired", "Violin"):
        frozen = read(Path(next(record['file'] for record in report['exports'] if record['label'] == f'D1-{family}' and record['format'] == 'json')))["result"]
        values = [point["y"] for point in frozen["points"]]
        assert len(values) == 12 and abs(sum(values) / len(values) - 4.5) < 1e-12
        assert len({point["observation"]["experimentalUnit"] for point in frozen["points"]}) == 6
        result = {"family": family, "values": values, "unit": frozen["unit"], "experimentalUnits": 6}
        if family == "Paired":
            result["deltas"] = [pair["delta"] for pair in frozen["pairs"]]
            assert result["deltas"] == [2] * 6
        if family == "Violin":
            result["densities"] = [{key: value for key, value in item.items() if key in ("grid", "density", "bandwidth", "group")} for item in frozen["densities"]]
        results.append(result)
    frozen_zip = next(item for item in report["exports"] if item["format"] == "zip")
    with zipfile.ZipFile(frozen_zip["file"]) as archive:
        assert archive.testzip() is None
        assert "panels/A-source.csv" in archive.namelist()
        assert any(name.endswith(".pdf") for name in archive.namelist())
        assert any(name.endswith(".svg") for name in archive.namelist())
        zip_members = archive.namelist()
    native = []
    if args.native:
        assert len(list((trial / "native").iterdir())) == 6
        for format_name in ("png", "svg", "pdf", "json", "csv", "zip"):
            expected = next(item for item in report["exports"] if item["format"] == format_name)
            actual = trial / f"native/actual.{format_name}"
            assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected["sha256"]
            native.append({"format": format_name, "bytes": actual.stat().st_size, "sha256": expected["sha256"]})
    output = {"passed": True, "scope": "Scientific content and files; native UI actions are recorded separately", "scientificAnswers": results, "frozenZipMembers": zip_members, "nativeFiles": native}
    (trial / "content-verification.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "scientificFamilies": len(results), "nativeFormats": len(native)}))


if __name__ == "__main__":
    main()
