"""Export/check the executable Phase 1 preparation plan contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from labviz_api.research_preparation import PreparationPlan

PATH = Path(__file__).resolve().parents[2] / "contracts/preparation-plan-v1.schema.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    schema = PreparationPlan.model_json_schema()
    if arguments.check:
        if json.loads(PATH.read_text(encoding="utf-8")) != schema:
            raise SystemExit("Preparation plan schema drift")
        print("Preparation plan schema matches.")
    else:
        PATH.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
