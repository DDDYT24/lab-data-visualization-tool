"""The generated capabilities file, the refusal list, and the one place a chart kind lives in."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest

from labviz_api.engine.layers import LAYERS
from labviz_api.engine.stats import SUMMARIES
from labviz_api.refusals import PARAMS, STATUS, PreparationError, RefusalCode
from labviz_api.render_plan import RenderPlan
from labviz_api.research_charts import ScientificChartSpec
from scripts.export_capabilities import CONTRACTS, build

API = Path(__file__).resolve().parents[1] / "labviz_api"
WEB = API.parents[1] / "web/src"
CODES = set(get_args(RefusalCode))


@pytest.mark.parametrize("name", build())
def test_the_generated_contracts_are_what_the_registries_state(name: str) -> None:
    assert json.loads((CONTRACTS / name).read_text(encoding="utf-8")) == build()[name]


def raised() -> list[tuple[str, str, set[str]]]:
    """Every refusal raised in the API: where, its code, and the names of the values it passes."""
    found = []
    for path in API.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") in {
                "PreparationError",
                "fail",
                "reject",
            }:
                code = node.args[0]
                if isinstance(code, ast.Name) and path.name in {
                    "research_preparation.py",
                    "research_charts.py",
                }:
                    continue  # the bodies of fail() and reject()
                assert isinstance(code, ast.Constant), f"{path.name}:{node.lineno}"
                where = f"{path.name}:{node.lineno}"
                found.append((where, str(code.value), {str(key.arg) for key in node.keywords}))
    return found


def test_every_refusal_is_listed_and_passes_the_values_its_text_needs() -> None:
    calls = raised()
    assert {code for _, code, _ in calls} == CODES
    assert PARAMS.keys() | STATUS.keys() <= CODES
    for where, code, params in calls:
        assert params == set(PARAMS.get(code, ())), where  # type: ignore[call-overload]


@pytest.mark.parametrize("locale", ["zh", "en"])
def test_every_refusal_has_a_text_that_uses_only_its_values(locale: str) -> None:
    texts: dict[str, str] = json.loads(
        (WEB / f"messages/{locale}.json").read_text(encoding="utf-8")
    )["refusals"]
    assert texts.keys() == CODES
    for code, text in texts.items():
        assert text.strip()
        used = set(re.findall(r"{(\w+)}", text))
        assert used == set(PARAMS.get(code, ())), code  # type: ignore[call-overload]


def test_a_refusal_states_its_status_and_values() -> None:
    error = PreparationError("row-width", "a.csv, row 3: column count differs.", file="a", row=3)
    assert (error.status, error.params) == (422, {"file": "a", "row": 3})
    assert PreparationError("research-busy", "Busy.").status == 429


def literal(schema: dict[str, Any], *path: str) -> list[str]:
    for key in path:
        schema = schema[key]
    return list(schema["enum"])


def test_the_contracts_take_their_kinds_and_summaries_from_the_registries() -> None:
    spec = ScientificChartSpec.model_json_schema()
    assert literal(spec, "properties", "kind") == list(LAYERS)
    assert literal(spec, "properties", "summary") == ["none", *SUMMARIES]
    plan = RenderPlan.model_json_schema()["$defs"]
    assert literal(plan, "PlanSource", "properties", "kind")[: len(LAYERS)] == list(LAYERS)
    assert literal(plan, "SummaryMark", "properties", "statistic") == list(SUMMARIES)


# Files that name one of these words for something other than a research chart kind.
OTHER_MEANINGS = {
    "render_plan.py": "the violin mark",
    "render_plan_draw.py": "the violin mark",
    "plan_axes.py": "the violin mark",
    "figure_spec.py": "the layer kinds of FigureSpec v2, a contract of its own",
    "research_contract.py": "the prospective research-v1 contract",
    "base.py": "the first word of a stored axis signature",
    "render-plan.ts": "the violin mark",
    "render-plan-option.ts": "the violin mark",
    "figure-spec.ts": "the layer kinds of FigureSpec v2",
}


def test_no_other_file_names_a_chart_kind() -> None:
    named = re.compile("|".join(f"[\"'`]{kind}[\"'`]" for kind in LAYERS))
    entries = {f"{kind}.py" for kind in LAYERS}
    sources = [*API.rglob("*.py"), *WEB.rglob("*.ts"), *WEB.rglob("*.tsx")]
    naming = {
        path.name
        for path in sources
        if ".test." not in path.name
        and path.name not in entries
        and named.search(path.read_text(encoding="utf-8"))
    }
    # The spec's default kind is the one word outside the registry.
    assert naming - OTHER_MEANINGS.keys() == {"research_charts.py", "research_strip.py"}
    charts = (API / "research_charts.py").read_text(encoding="utf-8")
    assert named.findall(charts) == ['"grouped_strip"']
