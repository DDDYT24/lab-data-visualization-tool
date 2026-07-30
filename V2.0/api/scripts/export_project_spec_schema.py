"""Regenerate the committed ProjectSpec v1 JSON Schema from Pydantic."""

from __future__ import annotations

import json
from pathlib import Path

from labviz_api.project_spec import ProjectSpecV1


def main() -> None:
    destination = Path(__file__).resolve().parents[2] / "contracts" / "project-spec-v1.schema.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(ProjectSpecV1.model_json_schema(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
