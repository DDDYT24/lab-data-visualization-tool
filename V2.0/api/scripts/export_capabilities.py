"""Generate or check the contracts that follow from the registries.

capabilities-v1.json, which the web pickers and messages read, holds the research chart kinds, the
summary methods, the limits and every refusal code. The chart and template schemas list the kinds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, get_args

from pydantic.alias_generators import to_camel

from labviz_api.engine.layers import LAYERS
from labviz_api.engine.stats import SUMMARIES
from labviz_api.refusals import DEFAULT_STATUS, PARAMS, STATUS, RefusalCode
from labviz_api.render_contract import GEOMETRY
from labviz_api.render_plan import CONTRACT
from labviz_api.research_charts import MAX_GROUPS, ScientificChartSpec
from labviz_api.research_reuse import WorkflowTemplate

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


def capabilities() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "contracts": {"renderPlan": CONTRACT, "chartRender": GEOMETRY["contractVersion"]},
        "limits": {"groupsPerChart": MAX_GROUPS},
        "layers": {
            kind: {
                "title": layer.title._asdict(),
                "short": layer.short._asdict(),
                "options": [to_camel(option) for option in layer.options],
                "roles": list(layer.roles),
                "xAxis": layer.x_axis,
                "limits": dict(layer.limits),
            }
            for kind, layer in LAYERS.items()
        },
        "summaries": {
            key: {"label": summary.label._asdict()} for key, summary in SUMMARIES.items()
        },
        "refusals": {
            code: {
                "status": STATUS.get(code, DEFAULT_STATUS),
                "params": list(PARAMS.get(code, ())),
            }
            for code in sorted(get_args(RefusalCode))
        },
    }


def build() -> dict[str, Any]:
    return {
        "capabilities-v1.json": capabilities(),
        "scientific-chart-v1.schema.json": ScientificChartSpec.model_json_schema(),
        "workflow-template-v1.schema.json": WorkflowTemplate.model_json_schema(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, built in build().items():
        path = CONTRACTS / name
        if not args.check:
            text = json.dumps(built, ensure_ascii=False, indent=2) + "\n"
            path.write_text(text, encoding="utf-8", newline="\n")
        elif json.loads(path.read_text(encoding="utf-8")) != built:
            raise SystemExit(f"{name} differs from the registries")
    if args.check:
        print("Contracts match the registries.")


if __name__ == "__main__":
    main()
