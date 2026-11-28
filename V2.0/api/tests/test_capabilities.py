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


# Errors whose English message is shown as it is: the classic steps (with every ProcessingError,
# found below) and the request validation, whose message is the model's own.
CLASSIC = {
    "duplicate-quality-rule",
    "experiment-title-required",
    "invalid-chart",
    "invalid-experiment-metadata",
    "unknown-quality-finding",
    "validation-error",
}
# Errors of the cloud deployment, which no screen of the desktop package meets: accounts, e-mail
# codes, shares, the readiness probe of the hosted service.
CLOUD = {
    "auth-rate-limit-unavailable",
    "authentication-required",
    "challenge-locked",
    "database-unavailable",
    "email-delivery-failed",
    "email-rate-limited",
    "feature-not-migrated",
    "incorrect-code",
    "invalid-challenge",
    "invalid-client-identity",
    "object-storage-unavailable",
    "resend-too-soon",
    "share-download-disabled",
    "share-not-found",
}
CODE = re.compile(r"[a-z]+(?:-[a-z][a-z0-9]*)+")  # not "utf-8"
ERROR = re.compile(r"(?:Error|Problem|Conflict|NotFound|Unavailable|Pending|Migrated)$")


def stated_codes() -> tuple[set[str], set[str]]:
    """Every error code the API states, and those of ProcessingError among them."""

    def codes(node: ast.AST | None) -> set[str]:
        values = ast.walk(node) if node else []
        return {
            str(v.value)
            for v in values
            if isinstance(v, ast.Constant) and CODE.fullmatch(str(v.value))
        }

    stated: set[str] = set()
    classic: set[str] = set()
    for path in API.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                if ERROR.search(name):
                    found = set().union(*map(codes, node.args)) if node.args else set()
                    stated |= found
                    classic |= found if name == "ProcessingError" else set()
                stated |= codes(
                    next((k.value for k in node.keywords if k.arg in {"code", "error_code"}), None)
                )
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(
                    getattr(t, "id", getattr(t, "attr", "")) in {"code", "error_code"}
                    for t in targets
                ):
                    stated |= codes(node.value)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if getattr(key, "value", "") in {"code", "error_code"}:
                        stated |= codes(value)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                stated |= set(re.findall(r"error_code\s*=\s*'([a-z0-9-]+)'", node.value))
    return stated, classic


def test_every_service_error_is_worded_or_listed_as_not_worded() -> None:
    stated, processing = stated_codes()
    texts = json.loads((WEB / "messages/zh.json").read_text(encoding="utf-8"))["serviceErrors"]
    client = {"api-error", "invalid-contract-response", "network-unavailable"}
    assert texts.keys() >= client
    # A worded code must not also be a ProcessingError, whose own message is shown.
    assert not texts.keys() & processing
    assert stated - CODES - processing == (texts.keys() - client) | CLASSIC | CLOUD


def test_the_collector_finds_each_way_a_code_is_stated() -> None:
    stated, processing = stated_codes()
    assert {
        "invalid-chart",  # held in a variable first
        "processing-interrupted",  # in SQL
        "incorrect-code",  # AuthError(message, code)
        "auth-rate-limit-unavailable",  # a class attribute
        "desktop-draining",  # a response body
        "project-revision-conflict",  # a persistence exception
    } <= stated
    assert {"fit-not-suitable", "unreadable-file"} <= processing


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
