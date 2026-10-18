from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from matplotlib.figure import Figure
from test_research_api import project, settings, unlock
from test_research_charts import prepared_csv
from test_research_figure_board import charts

from labviz_api.main import create_app
from labviz_api.research_charts import ScientificChartSpec, compute_chart
from labviz_api.research_figure_board import FigureBoardSpec, interaction, scene_row_references
from labviz_api.research_figure_render import render_board
from labviz_api.research_figure_sources import scientific_source


@pytest.mark.parametrize("grid_visible", [False, True])
@pytest.mark.parametrize("panel_count", [1, 4])
def test_composed_grid_visibility_applies_to_all_formats_without_moving_marks(
    monkeypatch: pytest.MonkeyPatch, grid_visible: bool, panel_count: int
) -> None:
    result = compute_chart(prepared_csv("v\n0\n1\n2\n"), ScientificChartSpec())
    sources = [scientific_source(result, f"panel-{i}") for i in range(panel_count)]
    spec = FigureBoardSpec(panel_chart_ids=["a"] * panel_count, grid_visible=grid_visible)
    original = Figure.savefig
    observed: list[list[bool]] = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        observed.append(
            [
                line.get_visible()
                for axis in figure.axes
                for line in axis.get_xgridlines() + axis.get_ygridlines()
            ]
        )
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        artifacts, scene = render_board(sources, spec)
    assert artifacts["png"].startswith(b"\x89PNG")
    assert artifacts["pdf"].startswith(b"%PDF")
    assert b"<svg" in artifacts["svg"]
    assert len(observed) == 3 and all(states for states in observed)
    assert all(visible == grid_visible for states in observed for visible in states)
    assert not any("First parameter to grid() is false" in str(item.message) for item in captured)
    _, opposite_scene = render_board(
        sources, spec.model_copy(update={"grid_visible": not grid_visible})
    )
    assert scene["marks"] == opposite_scene["marks"]
    assert scene["rowSets"] == opposite_scene["rowSets"]


def legacy(client: TestClient, kind: str) -> tuple[str, dict[str, Any], str]:
    uploaded = client.post(
        "/api/v1/projects",
        files={"file": ("original.csv", b"x,y,z\n1,2,4\n2,4,3\n3,6,2\n4,8,1\n", "text/csv")},
    )
    identifier = uploaded.json()["projectId"]
    workspace = client.get(f"/api/v1/projects/{identifier}/workspace").json()
    chart = workspace["chart"]
    chart["type"] = kind
    chart["title"] = f"Frozen {kind}"
    chart["panelCount"] = 1
    chart["xAxis"].update(field="x", title="Time", unit="s")
    chart["yAxis"].update(field="y", title="Signal", unit="mV")
    chart["series"] = [
        {
            "field": field,
            "label": field,
            "color": "#1768ac",
            "panel": 1,
            "yAxis": "primary",
            "lineStyle": "solid",
        }
        for field in (["y", "z"] if kind == "heatmap" else ["y"])
    ]
    assert (
        client.put(f"/api/v1/projects/{identifier}/chart", json={"chart": chart}).status_code == 200
    )
    response = client.post(f"/api/v1/projects/{identifier}/research/figure-sources?panel=1")
    assert response.status_code == 200, response.text
    return identifier, chart, response.json()["id"]


@pytest.mark.parametrize("kind", ["line", "scatter", "box", "heatmap"])
def test_legacy_mixed_export_frozen_copy_and_access(tmp_path: Path, kind: str) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier, chart, captured = legacy(client, kind)
        scientific = charts(client, identifier)[0]["id"]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        response = client.post(
            base,
            json={
                "title": "组合图",
                "panelChartIds": [captured, scientific],
                "palette": "grayscale",
                "fontSize": 12,
            },
        )
        assert response.status_code == 200, response.text
        board = response.json()
        assert board["composed"] is True
        assert board["panels"][0]["kind"] == kind
        url = base + "/" + board["id"]
        artifacts = {f: client.get(url + "/artifact/" + f).content for f in ["svg", "png", "pdf"]}
        assert b"<svg" in artifacts["svg"]
        assert artifacts["png"][1:4] == b"PNG"
        assert artifacts["pdf"][:4] == b"%PDF"
        view = client.get(url + "/interaction").json()
        a_marks = [m for m in view["marks"] if m["panel"] == "A"]
        assert a_marks
        selected = client.get(url + "/interaction", params={"mark": a_marks[0]["id"]}).json()
        assert selected["selectedInputs"] == (1 if kind in {"line", "scatter"} else 4)
        assert all(row["panel"] == "A" for row in selected["rows"])
        # Main chart edits do not rewrite captured inputs or artifacts.
        chart["title"] = "Later main chart"
        client.put(f"/api/v1/projects/{identifier}/chart", json={"chart": chart})
        assert client.get(url + "/artifact/svg").content == artifacts["svg"]
        duplicate = client.post(f"/api/v1/projects/{identifier}/duplicate").json()["projectId"]
        copied_base = f"/api/v1/projects/{duplicate}/research/figure-boards"
        copied_id = client.get(copied_base).json()["boards"][0]["id"]
        assert client.get(copied_base + "/" + board["id"] + "/interaction").status_code == 404
        assert (
            client.get(f"/api/v1/projects/{duplicate}/research/figure-sources").status_code == 200
        )
        client.delete(f"/api/v1/projects/{identifier}")
        for f, payload in artifacts.items():
            assert client.get(copied_base + "/" + copied_id + "/artifact/" + f).content == payload
        assert (
            client.get(copied_base + "/" + copied_id + "/interaction").json()["rows"]
            == view["rows"]
        )


def test_shared_axes_geometry_and_formats_use_one_frozen_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = compute_chart(prepared_csv("v\n0\n1\n2\n"), ScientificChartSpec())
    second = compute_chart(prepared_csv("v\n10\n20\n30\n"), ScientificChartSpec(kind="violin"))
    panels = [scientific_source(first, "one"), scientific_source(second, "two")]
    original = Figure.savefig
    recorded = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        recorded.append([(a.get_xlim(), a.get_ylim(), a.get_title()) for a in figure.axes])
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    artifacts, scene = render_board(
        panels, FigureBoardSpec(panel_chart_ids=["a", "b"], axis_mode="shared_xy")
    )
    assert set(artifacts) == {"svg", "png", "pdf"}
    assert recorded[0] == recorded[1] == recorded[2]
    assert scene["panels"][0]["xLimits"] == scene["panels"][1]["xLimits"]
    assert scene["panels"][0]["yLimits"] == scene["panels"][1]["yLimits"]
    assert all(0 < m["cx"] < 1 and 0 < m["cy"] < 1 for m in scene["marks"])
    density = next(m for m in scene["marks"] if m["kind"] == "density")
    selected = interaction(scene, density["id"], None, 0, "", False)
    assert selected["selectedInputs"] == 3
    assert selected["selectedKind"] == "density"


def test_rejected_shared_axis_is_atomic_and_selection_never_changes_counts(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier, _, captured = legacy(client, "line")
        saved = charts(client, identifier)
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        rejected = client.post(
            base, json={"panelChartIds": [captured, saved[0]["id"]], "axisMode": "shared_x"}
        )
        assert rejected.status_code == 422
        assert rejected.json()["code"] == "figure-incompatible-axis"
        assert client.get(base).json()["boards"] == []
        board = client.post(
            base, json={"panelChartIds": [c["id"] for c in saved], "axisMode": "shared_xy"}
        ).json()
        url = base + "/" + board["id"]
        before = client.get(url).json()
        scene = client.get(url + "/interaction").json()
        mark = next(m for m in scene["marks"] if m["panel"] == "A" and m["kind"] == "point")
        selected = client.get(url + "/interaction", params={"mark": mark["id"]}).json()
        assert {r["panel"] for r in selected["rows"]} == {"A", "B"}
        assert len({r["sampleId"] for r in selected["rows"]}) == 1
        row_id = selected["rows"][0]["id"]
        reversed_view = client.get(
            url + "/interaction", params={"row": row_id, "descending": True}
        ).json()
        assert {r["id"] for r in reversed_view["rows"]} == {r["id"] for r in selected["rows"]}
        filtered = client.get(
            url + "/interaction", params={"mark": mark["id"], "query": "no-match-at-all"}
        ).json()
        assert filtered["totalRows"] == 0
        assert filtered["selectedInputs"] == 1
        assert client.get(url).json() == before
        assert client.get(url + "/interaction", params={"mark": "Z:wrong"}).status_code == 422


def test_old_layout_reopens_without_fabricated_export(tmp_path: Path) -> None:
    import sqlite3

    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        saved = charts(client, identifier)
        with sqlite3.connect(config.database_path) as connection:
            connection.execute(
                "INSERT INTO research_figure_boards VALUES (?,?,?,?)",
                (
                    "old",
                    identifier,
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "title": "Old",
                            "axisMode": "independent",
                            "panelChartIds": [saved[0]["id"]],
                        }
                    ),
                    "2026-09-28",
                ),
            )
        url = f"/api/v1/projects/{identifier}/research/figure-boards/old"
        assert client.get(url).json()["composed"] is False
        assert client.get(url + "/artifact/svg").status_code == 404


def test_capture_selected_legacy_panel_preserves_assignments_and_categories(tmp_path: Path) -> None:
    import sqlite3

    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier, chart, _ = legacy(client, "line")
        chart["panelCount"] = 2
        chart["series"].append({**chart["series"][0], "field": "z", "label": "z", "panel": 2})
        url = f"/api/v1/projects/{identifier}"
        assert client.put(url + "/chart", json={"chart": chart}).status_code == 200
        before = client.get(url + "/workspace").json()["chart"]
        captured = client.post(url + "/research/figure-sources?panel=2")
        assert captured.status_code == 200, captured.text
        assert client.get(url + "/workspace").json()["chart"] == before
        with sqlite3.connect(config.database_path) as connection:
            result = json.loads(
                connection.execute(
                    "SELECT result_json FROM research_figure_sources WHERE id=?",
                    (captured.json()["id"],),
                ).fetchone()[0]
            )
        assert result["legacySpec"]["series"][0]["field"] == "z"
        assert result["legacySpec"]["series"][0]["panel"] == 1
        assert [m["y"] for m in result["marks"]] == [4, 3, 2, 1]


def test_categorical_legacy_hit_targets_follow_duplicate_rows(tmp_path: Path) -> None:
    import pandas as pd

    from labviz_api.models import ChartSpec
    from labviz_api.processing import default_chart_spec
    from labviz_api.research_figure_sources import legacy_snapshot

    frame = pd.DataFrame({"name": ["b", "a", "b"], "v": [1.0, 2.0, 3.0]})
    chart = default_chart_spec(frame)
    chart["type"] = "scatter"
    chart["xAxis"]["field"] = "name"

    class Repository:
        def get_workspace(self, _: str) -> dict[str, Any]:
            return {"chart": chart, "decisions": [], "quality": {}, "preview": {}}

        def load_chart_dataframe(self, _: str) -> Any:
            return frame

        def load_quality_dataframe(self, _: str) -> Any:
            return frame

    result = legacy_snapshot(Repository(), "id", 1, "name", "research-namespace")
    assert result["rows"][0]["sampleId"] == result["rows"][2]["sampleId"]
    assert result["rows"][0]["sampleId"] != result["rows"][1]["sampleId"]
    assert result["xSignature"][-1] == ["b", "a"]
    _, scene = render_board([result], FigureBoardSpec(panel_chart_ids=["id"]))
    assert scene["marks"][0]["cx"] == scene["marks"][2]["cx"]
    assert scene["marks"][0]["rowIds"] != scene["marks"][2]["rowIds"]
    assert scene["marks"][0]["cy"] != scene["marks"][2]["cy"]
    assert ChartSpec.model_validate(chart).type == "scatter"


def test_sampled_hit_targets_keep_complete_paginated_inputs() -> None:
    result = compute_chart(
        prepared_csv("v\n" + "\n".join(str(i) for i in range(2005)) + "\n"), ScientificChartSpec()
    )
    _, scene = render_board(
        [scientific_source(result, "revision")], FigureBoardSpec(panel_chart_ids=["chart"])
    )
    assert scene["sampled"] is True
    assert len(scene["marks"]) <= 2000
    assert len(scene_row_references(scene)) == 2005
    first = interaction(scene, None, None, 0, "", False)
    second = interaction(scene, None, None, 100, "", False)
    assert len(first["rows"]) == len(second["rows"]) == 100
    assert not ({r["id"] for r in first["rows"]} & {r["id"] for r in second["rows"]})
    omitted = next(
        r
        for r, _ in scene_row_references(scene)
        if all(r["id"] not in m["rowIds"] for m in scene["marks"])
    )
    selected = interaction(scene, None, omitted["id"], 0, "", False)
    assert selected["rows"][0]["value"] == omitted["value"]
    assert len(scene_row_references(scene)) == 2005


def test_cached_svg_backend_matches_standard_marker_geometry_and_alpha() -> None:
    import io

    import matplotlib as mpl

    with mpl.rc_context({"svg.hashsalt": "research-style-test"}):
        figure = Figure(figsize=(4, 3), dpi=300)
        axis = figure.subplots()
        axis.plot([1, 1, 2], [2, 2, 3], linestyle="None", marker="D", color="#1768ac", alpha=0.7)
        normal, cached = io.BytesIO(), io.BytesIO()
        figure.savefig(normal, format="svg", metadata={"Date": None})
        figure.savefig(
            cached,
            format="svg",
            metadata={"Date": None},
            backend="module://labviz_api.research_svg_backend",
        )
        assert normal.getvalue() == cached.getvalue()


def test_compact_scene_retains_old_flat_scene_interaction() -> None:
    result = compute_chart(prepared_csv("v\n0\n1\n2\n"), ScientificChartSpec())
    panel = scientific_source(result, "same-revision")
    _, compact = render_board([panel, panel], FigureBoardSpec(panel_chart_ids=["a", "a"]))
    assert len(compact["rowSets"]) == 1
    flattened = {
        **compact,
        "rows": [{**row, "panel": label} for row, label in scene_row_references(compact)],
    }
    del flattened["rowSets"]
    del flattened["panelRowSets"]
    for mark in [None, compact["marks"][0]["id"]]:
        assert interaction(compact, mark, None, 0, "", False) == interaction(
            flattened, mark, None, 0, "", False
        )


def test_indexed_scene_matches_uncached_linking_filter_and_sort() -> None:
    from labviz_api.research_figure_board import SceneIndex

    first = compute_chart(prepared_csv("s,v\na,1\na,2\nb,3\n", sample="s"), ScientificChartSpec())
    sources = [scientific_source(first, "one")] * 2
    _, scene = render_board(sources, FigureBoardSpec(panel_chart_ids=["a", "a"]))
    index = SceneIndex.build(scene)
    for mark, row, query, descending in [
        (None, None, "", False),
        ("A:0", None, "", False),
        (None, first["points"][0]["observation"]["id"], "a", True),
        (None, None, "no-match", True),
    ]:
        assert interaction(scene, mark, row, 0, query, descending, index) == interaction(
            scene, mark, row, 0, query, descending
        )


def test_geometry_only_draw_matches_full_draw_in_hybrid_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    result = compute_chart(prepared_csv("v\n0\n1\n2\n"), ScientificChartSpec())
    source = scientific_source(result, "revision")
    spec = FigureBoardSpec(panel_chart_ids=["a", "a"], point_rendering="raster")
    artifacts, scene = render_board([source, source], spec)

    def full_draw(figure: Figure) -> None:
        cast(Callable[[], None], FigureCanvasAgg(figure).draw)()

    monkeypatch.setattr(Figure, "draw_without_rendering", full_draw)
    full_artifacts, full_scene = render_board([source, source], spec)
    assert scene == full_scene
    assert artifacts["png"] == full_artifacts["png"]
    assert scene["exportRendering"]["rasterPanels"] == ["A", "B"]
    assert len(scene["rowSets"][scene["panelRowSets"]["A"]]) == 3


def test_bounded_hit_targets_preserve_group_extrema_and_all_source_inputs() -> None:
    prepared = prepared_csv(
        "g,v\n"
        + "".join(
            f"{i % 2},{-1000000 if i == 177 else 1000000 if i == 977 else i}\n" for i in range(3000)
        ),
        group="g",
    )
    result = compute_chart(prepared, ScientificChartSpec())
    frozen_hash = result["resultSha256"]
    snapshot = scientific_source(result, "revision")
    points = [mark for mark in snapshot["marks"] if mark["kind"] == "point"]
    assert len(points) <= 500
    assert snapshot["sampled"] is True
    assert len(snapshot["rows"]) == len(result["points"]) == 3000
    selected = {mark["rowIds"][0] for mark in points}
    for group in ("0", "1"):
        values = [point for point in result["points"] if point["observation"]["group"] == group]
        for extreme in (min(values, key=lambda p: p["y"]), max(values, key=lambda p: p["y"])):
            assert extreme["observation"]["id"] in selected
    assert result["resultSha256"] == frozen_hash


def test_capture_sample_linkage_reads_only_revision_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_research_chart_api import applied

    from labviz_api.research_store import ResearchStore

    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier, _, _ = legacy(client, "line")
        applied(
            client,
            identifier,
            b"sample,v\na,1\nb,2\n",
            {"mapping": {"sample": "sample", "value": "v"}},
        )
        with ResearchStore(config.database_path).connect() as connection:
            namespace = connection.execute(
                "SELECT json_extract(raw_json, '$.namespace') FROM research_revisions "
                "WHERE project_id=?",
                (identifier,),
            ).fetchone()[0]

        def forbid_full_snapshot(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Capture must not load complete research snapshots")

        monkeypatch.setattr(ResearchStore, "current", forbid_full_snapshot)
        response = client.post(
            f"/api/v1/projects/{identifier}/research/figure-sources?sampleField=x"
        )
        assert response.status_code == 200, response.text
        with ResearchStore(config.database_path).connect() as connection:
            frozen = json.loads(
                connection.execute(
                    "SELECT result_json FROM research_figure_sources WHERE id=?",
                    (response.json()["id"],),
                ).fetchone()[0]
            )
        from labviz_api.research_preparation import stable_id

        assert frozen["rows"][0]["sampleId"] == stable_id(namespace, "sample", "1")
        assert len(frozen["rows"]) == 4
