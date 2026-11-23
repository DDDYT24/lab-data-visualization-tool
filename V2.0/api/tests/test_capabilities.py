"""The generated capabilities file, the refusal list, and the one place a chart kind lives in."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest

from labviz_api.engine.layers import LAYERS, violin
from labviz_api.engine.stats import SUMMARIES
from labviz_api.refusals import PARAMS, STATUS, PreparationError, RefusalCode
from labviz_api.render_plan import RenderPlan
from labviz_api.research_charts import MAX_GROUPS, ScientificChartSpec
from labviz_api.research_figure_board import refusal, refusal_text
from labviz_api.research_preparation import LIMITS
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


def test_a_refusal_stored_in_a_job_keeps_its_values_and_2_4_rows_still_read() -> None:
    error = PreparationError("figure-shared-axis-control", "A shared x axis …", axis="x")
    assert refusal(refusal_text(error)) == {
        "refusal": "figure-shared-axis-control",
        "message": "A shared x axis …",
        "params": {"axis": "x"},
    }
    written_by_2_4 = '{"refusal": "axis-log-bars", "message": "Bars start at zero."}'
    assert refusal(written_by_2_4) == {
        "refusal": "axis-log-bars",
        "message": "Bars start at zero.",
        "params": {},
    }


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


# Where one of these words means something other than a research chart kind, and how often.
OTHER_MEANINGS = {
    "api/labviz_api/render_plan.py": 1,  # the violin mark
    "api/labviz_api/render_plan_draw.py": 1,
    "api/labviz_api/plan_axes.py": 2,
    "api/labviz_api/figure_spec.py": 2,  # the layer kinds of FigureSpec v2, a contract of its own
    "web/src/domain/figure-spec.ts": 2,
    "api/labviz_api/research_contract.py": 2,  # the prospective research-v1 contract
    "api/labviz_api/engine/layers/base.py": 1,  # the first word of a stored axis signature
    "web/src/domain/render-plan.ts": 1,  # the violin mark
    "web/src/components/common/render-plan-option.ts": 2,
    "api/labviz_api/research_charts.py": 1,  # the default kind of a spec
    "api/labviz_api/research_strip.py": 2,  # the spec of the point geometry, not of a chart
}


def test_no_other_file_names_a_chart_kind() -> None:
    root = API.parents[1]
    quoted = "|".join(f"[\"'`]{kind}[\"'`]" for kind in LAYERS)
    keys = "|".join(rf"(?m:^|[{{,])\s*{kind}\s*:" for kind in LAYERS)
    entries = {f"api/labviz_api/engine/layers/{kind}.py" for kind in LAYERS}
    naming: dict[str, int] = {}
    for path in [*API.rglob("*.py"), *WEB.rglob("*.ts"), *WEB.rglob("*.tsx")]:
        name = path.relative_to(root).as_posix()
        if ".test." in name or name in entries:
            continue
        pattern = quoted if path.suffix == ".py" else f"{quoted}|{keys}"
        found = len(re.findall(pattern, path.read_text(encoding="utf-8")))
        if found:
            naming[name] = found
    assert naming == OTHER_MEANINGS


def test_the_texts_state_the_limits_the_code_applies() -> None:
    limits = {
        "aggregate-byte-limit": LIMITS["aggregateBytes"] // 2**20,
        "chart-density-small": violin.MIN_OBSERVATIONS,
        "chart-many-groups": MAX_GROUPS,
        "file-count-limit": LIMITS["filesPerImport"],
        "row-limit": LIMITS["aggregateRows"],
        "single-file-limit": LIMITS["singleFileBytes"] // 2**20,
    }
    words = {3: "three", 5: "five"}
    for locale in ("zh", "en"):
        texts = json.loads((WEB / f"messages/{locale}.json").read_text(encoding="utf-8"))
        for code, limit in limits.items():
            text = texts["refusals"][code]
            assert f"{limit:,}" in text or words.get(limit, "") in text.split(), (locale, code)
