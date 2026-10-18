from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean
from typing import Any

import pytest
from pydantic import ValidationError

from labviz_api.research_contract import ResearchContractV1
from scripts.generate_v23_fixtures import build

ROOT = Path(__file__).resolve().parents[2] / "contracts"
FIXTURES = ROOT / "fixtures" / "v23"


def read(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_generated_schema_and_fixture_bytes_have_not_drifted() -> None:
    assert json.loads((ROOT / "research-v1.schema.json").read_text()) == (
        ResearchContractV1.model_json_schema()
    )
    for name, content in build().items():
        assert (FIXTURES / name).read_bytes() == content.encode()


def test_synthetic_answer_keys_and_source_cell_lineage() -> None:
    data = read("research.valid.json")
    parsed = ResearchContractV1.model_validate(data)
    assert parsed.model_dump(mode="json", by_alias=True) == data
    source_rows = {}
    for source in parsed.sources:
        payload = (FIXTURES / source.name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source.sha256
        source_rows[source.id] = list(csv.DictReader(io.StringIO(payload.decode())))
    for observation in parsed.observations:
        for cell in observation.lineage:
            row = source_rows[cell.source_id][cell.row - 2]
            assert row["id"] == observation.id
            if observation.value is not None:
                assert float(row[cell.column]) == observation.value
    primary = parsed.observations[:54]
    for group, expected in enumerate([3.5, 13.5, 23.5]):
        items = [
            item for item in primary if item.sample_id and item.sample_id.startswith(f"s{group}")
        ]
        assert len(items) == 18
        assert len({item.sample_id for item in items}) == 6
        assert mean(item.value for item in items if item.value is not None) == expected
    for number in range(1, 7):
        pair = [item for item in parsed.observations if item.pair_key == f"pair{number}"]
        assert len(pair) == 2
        assert pair[0].value is not None and pair[1].value is not None
        assert pair[1].value - pair[0].value == 2
    assert parsed.observations[66].value == 0
    assert all(item.state != "observed" for item in parsed.observations[67:])


@pytest.mark.parametrize("fixture", read("research.invalid.json"), ids=lambda item: item["case"])
def test_shared_invalid_contracts(fixture: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ResearchContractV1.model_validate(fixture["value"])


def test_unknown_legacy_identity_is_not_inferred() -> None:
    result = ResearchContractV1.model_validate(read("research.unknown-identity.json"))
    assert result.analyses[0].independent_unit_count is None
    assert result.analyses[0].sample_count == 0
    assert result.analyses[0].observation_count == 54
    assert all(item.sample_id is None for item in result.observations)
