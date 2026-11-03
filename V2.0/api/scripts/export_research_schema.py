"""Generate or check the prospective research contract shared with the frontend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from labviz_api.research_contract import ResearchContractV1

DESTINATION = Path(__file__).resolve().parents[2] / "contracts" / "research-v1.schema.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    schema = ResearchContractV1.model_json_schema()
    if args.check:
        if json.loads(DESTINATION.read_text(encoding="utf-8")) != schema:
            raise SystemExit("research-v1.schema.json differs from Python contract")
        print("Research schema matches.")
    else:
        DESTINATION.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
