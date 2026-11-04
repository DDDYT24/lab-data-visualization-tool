from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from labviz_api.render_plan import RenderPlan, build_research_plan, grayscale, nice_axis
from labviz_api.research_chart_render import draw_plan
from labviz_api.research_charts import ScientificChartSpec, compute_chart
from labviz_api.research_preparation import PreparationPlan, parse_sources, prepare

CONTRACT = Path(__file__).resolve().parents[2] / "contracts/render-plan-v1.schema.json"


def chart(body: str, spec: dict[str, Any], **mapping: str) -> dict[str, Any]:
    prepared = prepare(
        parse_sources("plan-test", [("test.csv", body.encode())]),
        PreparationPlan.model_validate({"mapping": {"value": "v", **mapping}}),
    )
    return compute_chart(prepared, ScientificChartSpec.model_validate(spec))


GROUPED = "s,g,v\na,A,1.5\nb,A,2\nc,A,4.25\nd,B,7\ne,B,8.5\nf,B,6\n"


def grouped(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    return chart(GROUPED, spec or {}, sample="s", experimentalUnit="s", group="g")


def paired() -> dict[str, Any]:
    return chart(
        "s,p,t,v,g\na,p1,before,1,A\na,p1,after,3,A\nb,p2,before,4,B\nb,p2,after,6,B\n",
        {"kind": "paired", "pairTimes": ["before", "after"]},
        sample="s",
        experimentalUnit="s",
        pair="p",
        time="t",
        group="g",
    )


def marks(plan: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [m for m in plan["panels"][0]["marks"] if m["kind"] == kind]


def assert_axis_frames(axis: dict[str, Any], values: list[float]) -> None:
    low, high = axis["domain"]
    margin = (max(values) - min(values)) * 0.05
    assert (low, high) == pytest.approx((min(values) - margin, max(values) + margin))
    ticks = [t["value"] for t in axis["ticks"]]
    assert low <= ticks[0] and ticks[-1] <= high and ticks == sorted(ticks)
    assert 2 <= len(ticks) <= 9
    step = ticks[1] - ticks[0]
    mantissa = step / 10 ** math.floor(math.log10(step))
    assert any(math.isclose(mantissa, m) for m in (1, 2, 5))


def test_schema_matches_checked_in_contract() -> None:
    assert json.loads(CONTRACT.read_text(encoding="utf-8")) == RenderPlan.model_json_schema(
        by_alias=True
    )


def test_grouped_plan_covers_every_observation_and_is_deterministic() -> None:
    result = grouped({"summary": "mean", "summaryUnit": "observation"})
    plan = build_research_plan(result)
    RenderPlan.model_validate_json(json.dumps(plan))
    assert build_research_plan(result) == plan
    points = marks(plan, "points")
    refs = [ref for mark in points for ref in mark["sourceRefs"]]
    assert sorted(refs) == sorted(p["observation"]["id"] for p in result["points"])
    assert all(len(m["x"]) == len(m["y"]) == len(m["sourceRefs"]) for m in points)
    # Summaries draw last, above the points.
    assert [m["kind"] for m in plan["panels"][0]["marks"]][-2:] == ["summary", "summary"]
    assert sorted(m["y"] for m in marks(plan, "summary")) == [2.5833333333333335, 7.166666666666667]
    axes = plan["panels"][0]["axes"]
    assert_axis_frames(axes["y"], [y for m in points for y in m["y"]])
    assert [t["label"].split("\n")[0] for t in axes["x"]["ticks"]] == ["A", "B"]
    assert axes["x"]["domain"] == [-0.65, 1.65]


def test_plan_hash_binds_presentation_and_source() -> None:
    result = grouped()
    plan = build_research_plan(result)
    gray = build_research_plan(result, use_grayscale=True)
    assert gray["planHash"] != plan["planHash"]
    assert {m["color"] for m in marks(gray, "points")} <= {
        grayscale(m["color"]) for m in marks(plan, "points")
    }
    assert plan["source"]["resultSha256"] == result["resultSha256"]


def test_paired_plan_keeps_each_pair_and_its_times() -> None:
    result = paired()
    plan = build_research_plan(result)
    RenderPlan.model_validate_json(json.dumps(plan))
    segments = marks(plan, "segments")
    assert sorted(ref for m in segments for ref in map(tuple, m["sourceRefs"])) == sorted(
        (p["firstId"], p["secondId"]) for p in result["pairs"]
    )
    axes = plan["panels"][0]["axes"]
    assert [t["label"] for t in axes["x"]["ticks"]] == ["before", "after"]
    assert [entry["label"] for entry in plan["panels"][0]["legend"]] == ["A", "B"]
    assert_axis_frames(axes["y"], [1, 3, 4, 6])


def test_violin_plan_frames_the_full_density_support() -> None:
    result = grouped({"kind": "violin"})
    plan = build_research_plan(result)
    RenderPlan.model_validate_json(json.dumps(plan))
    violins = marks(plan, "violin")
    assert [m["n"] for m in violins] == [3, 3]
    assert all(len(m["grid"]) == len(m["halfWidths"]) for m in violins)
    support = [value for d in result["densities"] for value in d["support"]]
    assert_axis_frames(plan["panels"][0]["axes"]["y"], support)
    assert "Gaussian KDE" in plan["disclosures"][1]


@pytest.mark.parametrize(
    ("low", "high", "domain", "labels"),
    [
        (0.12, 9.7, (-0.359, 10.179), ["0", "2", "4", "6", "8", "10"]),
        (1.0, 1.0, (0.89, 1.11), ["0.90", "0.95", "1.00", "1.05", "1.10"]),
        (
            -250.0,
            -20.0,
            (-261.5, -8.5),
            ["\u2212250", "\u2212200", "\u2212150", "\u2212100", "\u221250"],
        ),
    ],
)
def test_nice_axis_steps_through_one_two_five(
    low: float, high: float, domain: tuple[float, float], labels: list[str]
) -> None:
    found, ticks = nice_axis(low, high)
    assert found == pytest.approx(domain)
    assert [t["label"] for t in ticks] == labels


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(lambda: grouped({"summary": "mean", "summaryUnit": "observation"}), id="dots"),
        pytest.param(paired, id="paired"),
        pytest.param(lambda: grouped({"kind": "violin"}), id="violin"),
    ],
)
def test_every_export_format_draws_the_axes_the_plan_states(
    monkeypatch: pytest.MonkeyPatch, result: Any
) -> None:
    from matplotlib.figure import Figure

    plan = build_research_plan(result())
    original = Figure.savefig
    drawn = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        ax = figure.axes[0]
        drawn.append(
            (
                ax.get_ylim(),
                ax.get_xlim(),
                [float(tick) for tick in ax.get_yticks()],
                [label.get_text() for label in ax.get_yticklabels()],
            )
        )
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    artifacts = draw_plan(plan)
    assert set(artifacts) == {"png", "svg", "pdf"} and len(drawn) == 3
    axes = plan["panels"][0]["axes"]
    expected = (
        tuple(axes["y"]["domain"]),
        tuple(axes["x"]["domain"]),
        [tick["value"] for tick in axes["y"]["ticks"]],
        [tick["label"] for tick in axes["y"]["ticks"]],
    )
    assert all(frame == expected for frame in drawn)


def test_board_panel_keeps_axes_it_can_share() -> None:
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator

    figure = Figure()
    axis = figure.subplots()
    assert draw_plan(build_research_plan(grouped()), target_figure=figure, target_axis=axis) == {}
    assert not isinstance(axis.yaxis.get_major_locator(), FixedLocator)


def test_grayscale_matches_matplotlib() -> None:
    from matplotlib.colors import to_hex, to_rgb

    for color in ("#1768ac", "#b66d13", "#ffffff", "#000000"):
        gray = sum(c * w for c, w in zip(to_rgb(color), (0.2126, 0.7152, 0.0722), strict=True))
        assert grayscale(color) == to_hex((gray, gray, gray))
