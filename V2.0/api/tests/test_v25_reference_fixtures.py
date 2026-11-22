from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter

from scripts.build_v25_reference_fixtures import ENTRIES, R12_TIMES, R16_FOLLOW_UP, ROOT, build


def read(files: dict[str, str], entry_id: str) -> list[dict[str, str]]:
    (name,) = [name for name in files if name.startswith(f"{entry_id}_")]
    return list(csv.DictReader(io.StringIO(files[name])))


def test_committed_fixtures_match_the_generator_byte_for_byte() -> None:
    files = build()
    assert sorted(path.name for path in ROOT.iterdir()) == sorted(files)
    for name, content in files.items():
        assert (ROOT / name).read_bytes() == content.encode(), name


def test_manifest_describes_every_reference_entry() -> None:
    files = build()
    entries = json.loads(files["manifest.json"])["entries"]
    ids = [entry["id"] for entry in entries]
    assert ids == [f"R{number:02d}" for number in range(1, 21)] == [e.id for e in ENTRIES]
    with_data = {entry["id"] for entry in entries if entry["file"] is not None}
    for entry in entries:
        assert entry["design"] and entry["analysis"] and entry["chart"] and entry["milestones"]
        if entry["file"] is None:
            assert entry["panels"] and set(entry["panels"]) <= with_data
            continue
        content = files[entry["file"]]
        assert hashlib.sha256(content.encode()).hexdigest() == entry["sha256"]
        rows = list(csv.DictReader(io.StringIO(content)))
        assert list(rows[0]) == entry["columns"]
        assert len(rows) == entry["rows"]
        assert len({tuple(row.values()) for row in rows}) == len(rows)


def test_each_design_has_the_structure_its_analysis_needs() -> None:
    files = build()

    for entry_id, unit in (("R03", "slice"), ("R04", "mouse"), ("R19", "blot")):
        rows = read(files, entry_id)
        conditions = {row["condition"] for row in rows}
        assert len(conditions) == 2
        for value in {row[unit] for row in rows}:
            assert sorted(r["condition"] for r in rows if r[unit] == value) == sorted(conditions)

    cells = Counter((row["genotype"], row["treatment"]) for row in read(files, "R09"))
    assert len(cells) == 4 and sorted(cells.values()) == [5, 6, 6, 6]

    repeats = read(files, "R10")
    assert set(Counter(row["animal"] for row in repeats).values()) == {4}
    group_of: dict[str, str] = {}
    for row in repeats:
        assert group_of.setdefault(row["animal"], row["group"]) == row["group"]

    course = read(files, "R12")
    animals = {row["animal"] for row in course}
    assert len(animals) == 24 and len(course) == 24 * len(R12_TIMES)
    for animal in animals:
        assert [int(r["time_min"]) for r in course if r["animal"] == animal] == list(R12_TIMES)

    doses = read(files, "R13")
    assert all(float(row["concentration_uM"]) > 0 for row in doses)
    compound_of: dict[str, str] = {}
    for row in doses:
        assert compound_of.setdefault(row["replicate"], row["compound"]) == row["compound"]

    survival = read(files, "R16")
    assert all(1 <= int(row["time_days"]) <= R16_FOLLOW_UP for row in survival)
    for arm in {row["arm"] for row in survival}:
        assert {r["event"] for r in survival if r["arm"] == arm} <= {"0", "1"}
        assert any(r["event"] == "1" for r in survival if r["arm"] == arm)

    table = read(files, "R17")
    outcomes = {row["outcome"] for row in table}
    assert Counter(row["condition"] for row in table) == Counter(
        {condition: len(outcomes) for condition in {row["condition"] for row in table}}
    )
