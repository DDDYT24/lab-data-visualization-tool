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


def test_generator_repeats_from_its_seeds() -> None:
    assert build() == build()


def test_manifest_describes_every_reference_entry() -> None:
    files = build()
    entries = json.loads(files["manifest.json"])["entries"]
    ids = [entry["id"] for entry in entries]
    assert ids == [f"R{number:02d}" for number in range(1, 21)] == [e.id for e in ENTRIES]
    for entry in entries:
        assert entry["design"] and entry["analysis"] and entry["chart"] and entry["milestones"]
        if entry["file"] is None:
            assert entry["panels"] and set(entry["panels"]) < set(ids)
            continue
        content = files[entry["file"]]
        assert hashlib.sha256(content.encode()).hexdigest() == entry["sha256"]
        rows = list(csv.DictReader(io.StringIO(content)))
        assert list(rows[0]) == entry["columns"]
        assert len(rows) == entry["rows"]
        assert len({tuple(row.values()) for row in rows}) == len(rows)


def test_each_design_has_the_structure_its_analysis_needs() -> None:
    files = build()

    for entry_id, unit in (("R03", "slice"), ("R04", "mouse")):
        rows = read(files, entry_id)
        assert set(Counter(row[unit] for row in rows).values()) == {2}
        assert len({row["condition"] for row in rows}) == 2

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

    for row in read(files, "R13"):
        assert float(row["concentration_uM"]) > 0

    survival = read(files, "R16")
    assert {row["event"] for row in survival} == {"0", "1"}
    assert all(1 <= int(row["time_days"]) <= R16_FOLLOW_UP for row in survival)

    assert all(int(row["count"]) >= 0 for row in read(files, "R17"))

    blots = read(files, "R19")
    assert Counter(row["blot"] for row in blots) == Counter({f"b{n}": 2 for n in range(1, 5)})
    assert all(float(row["target_signal"]) > 0 < float(row["loading_signal"]) for row in blots)
