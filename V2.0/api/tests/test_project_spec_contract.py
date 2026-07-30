from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from labviz_api.project_spec import ProjectSpecV1

V2_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = V2_ROOT / "contracts" / "project-spec-v1.schema.json"
VALID_PATH = V2_ROOT / "contracts" / "fixtures" / "project-spec-v1.valid.json"
INVALID_PATH = V2_ROOT / "contracts" / "fixtures" / "project-spec-v1.invalid.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_json_schema_is_generated_from_pydantic_source() -> None:
    assert load_json(SCHEMA_PATH) == ProjectSpecV1.model_json_schema()


def test_pydantic_accepts_the_shared_valid_fixture() -> None:
    spec = ProjectSpecV1.model_validate(load_json(VALID_PATH))
    assert spec.schema_version == 1
    assert spec.cleaning is None
    assert spec.source.dataset_version_id.version == 4


@pytest.mark.parametrize("fixture", load_json(INVALID_PATH))
def test_pydantic_rejects_every_shared_invalid_fixture(fixture: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ProjectSpecV1.model_validate(fixture["value"])
