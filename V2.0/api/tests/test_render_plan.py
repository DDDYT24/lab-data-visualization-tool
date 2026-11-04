from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from matplotlib.figure import Figure

from labviz_api.render_plan import RenderPlan, build_research_plan, grayscale, nice_axis
from labviz_api.research_chart_render import draw_plan
from labviz_api.research_charts import ScientificChartSpec, compute_chart
from labviz_api.research_preparation import (
    PreparationError,
    PreparationPlan,
    parse_sources,
    prepare,
)

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


def test_plan_cites_the_geometry_rules_it_used() -> None:
    assert build_research_plan(grouped())["geometry"] == {
        "contract": "chart-render-v2",
        "axis": "nice-1-2-5-v1",
        "jitter": "sha256-uniform-v1",
        "density": None,
    }
    assert build_research_plan(paired())["geometry"]["jitter"] is None
    assert build_research_plan(grouped({"kind": "violin"}))["geometry"]["density"] == (
        "gaussian-kde-v1"
    )


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
        (1.0, 1.0, (0.945, 1.055), ["0.96", "0.98", "1.00", "1.02", "1.04"]),
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
    ("low", "high"),
    [
        (5.0, 5.0),
        (0.0, 0.0),
        (-3.0, -3.0),
        (1.0, 1.0 + 2.3e-16),
        (1e-9, 4.5e-9),
        (1.2e9, 4.5e9),
        (-250.0, -20.0),
    ],
)
def test_domain_equals_matplotlib_autoscale(low: float, high: float) -> None:
    axis = Figure().subplots()
    axis.plot([0, 1], [low, high])
    assert nice_axis(low, high)[0] == pytest.approx(axis.get_ylim())


@pytest.mark.parametrize(
    ("low", "high", "labels"),
    [
        (1e-9, 3e-9, ["1.0e\u22129", "1.5e\u22129", "2.0e\u22129", "2.5e\u22129", "3.0e\u22129"]),
        (
            -1e-9,
            1e-9,
            ["\u22121.0e\u22129", "\u22125.0e\u221210", "0", "5.0e\u221210", "1.0e\u22129"],
        ),
        (1e9, 3e9, ["1.0e9", "1.5e9", "2.0e9", "2.5e9", "3.0e9"]),
        (1e5, 4e5, ["100000", "150000", "200000", "250000", "300000", "350000", "400000"]),
        (1e6, 2e6, ["1.0e6", "1.2e6", "1.4e6", "1.6e6", "1.8e6", "2.0e6"]),
        (1e-4, 3e-4, ["0.00010", "0.00015", "0.00020", "0.00025", "0.00030"]),
        (
            123456789012.0,
            123456790999.0,
            [
                "1.234567890e11",
                "1.234567895e11",
                "1.234567900e11",
                "1.234567905e11",
                "1.234567910e11",
            ],
        ),
    ],
)
def test_extreme_values_keep_short_labels_that_stand_alone(
    low: float, high: float, labels: list[str]
) -> None:
    assert [t["label"] for t in nice_axis(low, high)[1]] == labels


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (0.1 + i * 1e-3, 0.1 + i * 1e-3 + 10.0**-exponent)
        for i in range(5)
        for exponent in range(16)
    ],
)
def test_ticks_stay_distinct_and_inside_the_domain(low: float, high: float) -> None:
    (start, end), ticks = nice_axis(low, high)
    values = [t["value"] for t in ticks]
    assert 2 <= len(ticks) <= 9
    assert start <= values[0] and values[-1] <= end
    assert len({t["label"] for t in ticks}) == len(ticks) == len(set(values))


@pytest.mark.parametrize(("low", "high"), [(-1.7e308, 1.7e308), (5e-324, 1e-323)])
def test_values_no_axis_can_hold_are_refused(low: float, high: float) -> None:
    with pytest.raises(PreparationError) as error:
        nice_axis(low, high)
    assert error.value.code == "chart-axis-range"


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


def test_long_tick_labels_stay_on_the_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Figure.savefig
    lefts = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        lefts.append(figure.axes[0].get_tightbbox().x0)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    result = chart(
        "s,g,v\na,A,123456789012\nb,A,123456789999\nc,B,123456790123\nd,B,123456790999\n",
        {},
        sample="s",
        experimentalUnit="s",
        group="g",
    )
    draw_plan(build_research_plan(result))
    assert len(lefts) == 3 and min(lefts) >= 0


def test_board_panel_draws_the_axes_its_plan_states() -> None:
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator

    figure = Figure()
    axis = figure.subplots()
    plan = build_research_plan(grouped())
    assert draw_plan(plan, target_figure=figure, target_axis=axis) == {}
    assert isinstance(axis.yaxis.get_major_locator(), FixedLocator)
    assert list(axis.get_ylim()) == plan["panels"][0]["axes"]["y"]["domain"]


def test_grayscale_matches_matplotlib() -> None:
    from matplotlib.colors import to_hex, to_rgb

    for color in ("#1768ac", "#b66d13", "#ffffff", "#000000"):
        gray = sum(c * w for c, w in zip(to_rgb(color), (0.2126, 0.7152, 0.0722), strict=True))
        assert grayscale(color) == to_hex((gray, gray, gray))
