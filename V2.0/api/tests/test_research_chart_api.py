from __future__ import annotations

import gc
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from matplotlib.figure import Figure
from test_research_api import PLAN, SOURCE, project, settings, unlock
from test_research_charts import paired_data, prepared_csv

from labviz_api.main import create_app
from labviz_api.research_chart_render import render_chart
from labviz_api.research_charts import ScientificChartSpec, compute_chart


def applied(
    client: TestClient, identifier: str, source: bytes = SOURCE, plan: dict[str, Any] = PLAN
) -> str:
    url = f"/api/v1/projects/{identifier}/research"
    current = client.get(url).json()["revisionId"]
    data = {"plan": json.dumps(plan)}
    if current:
        data["expectedRevision"] = current
    preview = client.post(
        url + "/preview", data=data, files={"files": ("chart.csv", source, "text/csv")}
    )
    assert preview.status_code == 200, preview.text
    saved = client.post(
        url + "/apply", json={"token": preview.json()["token"], "expectedRevision": current}
    )
    assert saved.status_code == 200
    return str(saved.json()["revisionId"])


@pytest.mark.parametrize("kind", ["grouped_strip", "paired", "violin"])
def test_all_render_formats_share_geometry_and_do_not_recompute(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = paired_data() if kind == "paired" else prepared_csv("v\n0\n1\n2\n")
    spec = ScientificChartSpec.model_validate(
        {"kind": kind, "pairTimes": ["before", "after"] if kind == "paired" else []}
    )
    result = compute_chart(prepared, spec)
    original = Figure.savefig
    captured = []
    point_geometry = []

    def capture(figure: Any, *args: Any, **kwargs: Any) -> Any:
        captured.append(
            [
                (c.get_offsets().tolist(), [p.vertices.tolist() for p in c.get_paths()])
                for c in figure.axes[0].collections
            ]
        )
        point_geometry.append(
            [
                list(zip(line.get_xdata(), line.get_ydata(), strict=True))
                for line in figure.axes[0].lines
                if line.get_linestyle() == "None"
            ]
        )
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    artifacts = render_chart(result)
    assert captured[0] == captured[1] == captured[2]
    assert artifacts["png"].startswith(b"\x89PNG")
    assert artifacts["pdf"].startswith(b"%PDF")
    assert b"<svg" in artifacts["svg"]
    if kind != "paired":
        assert point_geometry[0] == point_geometry[1] == point_geometry[2]
        observed = [xy for group in point_geometry[0] for xy in group]
        assert sorted(observed) == sorted((p["x"], p["y"]) for p in result["points"])
    else:
        segments = captured[0][0][1]
        assert segments == [[[0, p["first"]], [1, p["second"]]] for p in result["pairs"]]


def test_saved_chart_exports_restart_copy_delete_and_stale_revision(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        revision = applied(client, identifier)
        url = f"/api/v1/projects/{identifier}/research/charts"
        ids = []
        hashes = []
        for kind in ("grouped_strip", "violin"):
            response = client.post(url, json={"revisionId": revision, "spec": {"kind": kind}})
            assert response.status_code == 200, response.text
            chart = response.json()
            ids.append(chart["id"])
            hashes.append(chart["resultSha256"])
            assert chart["observations"] == 4 and chart["groups"][0]["independentUnits"] == 2
            for format in ("svg", "png", "pdf", "json"):
                exported = client.get(f"{url}/{chart['id']}/artifact/{format}?download=true")
                assert exported.status_code == 200
                assert exported.headers["x-labviz-result-sha256"] == chart["resultSha256"]
                assert "attachment" in exported.headers["content-disposition"]
            inline = client.get(f"{url}/{chart['id']}/artifact/svg")
            download = client.get(f"{url}/{chart['id']}/artifact/svg?download=true")
            assert inline.content == download.content
        next_revision = applied(client, identifier, SOURCE.replace(b",7,", b",9,"))
        assert next_revision != revision
        assert client.post(url, json={"revisionId": revision, "spec": {}}).status_code == 409
        assert len(client.get(url).json()["charts"]) == 2
        old = client.get(f"{url}/{ids[0]}").json()
        assert old["revisionId"] == revision and old["resultSha256"] == hashes[0]
        copied = client.post(f"/api/v1/projects/{identifier}/duplicate").json()["projectId"]
        copied_url = f"/api/v1/projects/{copied}/research/charts"
        copied_chart = client.get(copied_url).json()["charts"][0]
        assert copied_chart["id"] not in ids and copied_chart["revisionId"] != revision
        assert client.get(f"{copied_url}/{ids[0]}").status_code == 404
    with TestClient(create_app(config)) as client:
        assert client.get(url).status_code == 403
        assert client.get(f"{url}/{ids[0]}/artifact/svg").status_code == 403
        unlock(client)
        assert client.get(f"{url}/{ids[0]}").json()["resultSha256"] == hashes[0]
        assert client.delete(f"/api/v1/projects/{identifier}").status_code == 204
        assert client.get(copied_url).status_code == 200
        assert client.get(f"{copied_url}/{copied_chart['id']}/artifact/pdf").status_code == 200
    with sqlite3.connect(config.database_path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM research_charts WHERE project_id=?", (identifier,)
        ).fetchone() == (0,)


def test_paired_api_and_invalid_data_do_not_save_chart(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        source = b"s,p,t,v\na,p1,before,1\na,p1,after,3\nb,p2,before,4\nb,p2,after,6\n"
        plan = {"mapping": {"value": "v", "sample": "s", "pair": "p", "time": "t"}}
        revision = applied(client, identifier, source, plan)
        url = f"/api/v1/projects/{identifier}/research/charts"
        bad = client.post(
            url, json={"revisionId": revision, "spec": {"kind": "paired", "pairTimes": ["x", "y"]}}
        )
        assert bad.status_code == 422
        assert client.get(url).json()["charts"] == []
        response = client.post(
            url,
            json={
                "revisionId": revision,
                "spec": {"kind": "paired", "pairTimes": ["before", "after"]},
            },
        )
        assert response.status_code == 200, response.text
        assert [p["delta"] for p in response.json()["pairs"]] == [2, 2]
        assert response.json()["groups"][0]["independentUnits"] is None


def test_saved_chart_data_pages_use_frozen_full_data(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        source = ("v\n" + "\n".join(str(i) for i in range(205)) + "\n").encode()
        revision = applied(client, identifier, source, {"mapping": {"value": "v"}})
        url = f"/api/v1/projects/{identifier}/research/charts"
        response = client.post(url, json={"revisionId": revision, "spec": {}})
        assert response.status_code == 200, response.text
        chart = response.json()
        assert len(chart["rows"]) == 100 and chart["totalRows"] == 205
        last = client.get(f"{url}/{chart['id']}?offset=200").json()
        assert len(last["rows"]) == 5
        assert last["resultSha256"] == chart["resultSha256"]


def test_chart_owner_boundary_covers_data_and_artifacts(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        revision = applied(client, identifier)
        url = f"/api/v1/projects/{identifier}/research/charts"
        chart = client.post(url, json={"revisionId": revision, "spec": {}}).json()
        with sqlite3.connect(config.database_path) as connection:
            connection.execute(
                "UPDATE projects SET owner_user_id='other' WHERE id=?", (identifier,)
            )
        assert client.get(url).status_code == 403
        assert client.post(url, json={"revisionId": revision, "spec": {}}).status_code == 403
        assert client.get(f"{url}/{chart['id']}").status_code == 403
        for format in ("svg", "png", "pdf", "json"):
            assert client.get(f"{url}/{chart['id']}/artifact/{format}").status_code == 403


def test_chart_installation_rolls_back_on_failure(tmp_path: Path) -> None:
    from labviz_api.repository import ProjectRepository
    from labviz_api.research_chart_store import install_charts
    from labviz_api.research_store import ResearchStore

    database = tmp_path / "migration.db"
    ProjectRepository(database, 7200)
    store = ResearchStore(database)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE research_charts_project (blocker TEXT)")
        before = "\n".join(connection.iterdump())
    with pytest.raises(sqlite3.OperationalError):
        install_charts(store)
    with sqlite3.connect(database) as connection:
        assert "\n".join(connection.iterdump()) == before


def test_reopening_single_value_source_provides_valid_quick_plot(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        response = client.post(
            "/api/v1/projects",
            files={"file": ("research.csv", b"sample,value\ns1,1\ns2,2\ns3,3\n", "text/csv")},
        )
        identifier = response.json()["projectId"]
        workspace = client.get(f"/api/v1/projects/{identifier}/workspace").json()
        assert workspace["chart"]["type"] == "histogram"
        analysis = client.post(
            f"/api/v1/projects/{identifier}/chart-analysis", json={"chart": workspace["chart"]}
        )
        assert analysis.status_code == 200, analysis.text


def test_grayscale_converts_explicit_group_color_without_changing_marker_or_values() -> None:
    from matplotlib.colors import to_rgb
    from matplotlib.figure import Figure
    from test_research_charts import prepared_csv

    from labviz_api.research_chart_render import render_chart
    from labviz_api.research_charts import ScientificChartSpec, compute_chart

    result = compute_chart(prepared_csv("v\n1\n2\n3\n"), ScientificChartSpec())
    result["spec"]["categoryStyles"] = [{"group": None, "color": "#008577", "marker": "s"}]
    figure = Figure()
    axis = figure.subplots()
    render_chart(result, target_figure=figure, target_axis=axis, grayscale=True)
    line = next(line for line in axis.lines if line.get_linestyle() == "None")
    r, g, b = to_rgb(line.get_color())
    assert r == g == b
    assert line.get_marker() == "s"
    assert np.asarray(line.get_ydata()).tolist() == [point["y"] for point in result["points"]]
    assert result["spec"]["categoryStyles"][0]["color"] == "#008577"


def test_hybrid_retains_full_geometry_vector_axes_and_300dpi_image() -> None:
    import base64
    import io
    import xml.etree.ElementTree as ET

    from matplotlib.figure import Figure
    from PIL import Image
    from test_research_charts import prepared_csv

    from labviz_api.research_chart_render import rasterize_points

    assert not rasterize_points("auto", 20_000)
    assert rasterize_points("auto", 20_001)
    assert not rasterize_points("vector", 100_000)
    assert rasterize_points("raster", 1)
    data = prepared_csv("v\n1\n2\n3\n4\n")
    vector = compute_chart(data, ScientificChartSpec(point_rendering="vector"))
    hybrid = compute_chart(data, ScientificChartSpec(point_rendering="raster"))
    assert vector["points"] == hybrid["points"]
    assert vector["groups"] == hybrid["groups"]
    figure = Figure(figsize=(8, 5.5), dpi=300)
    axis = figure.subplots()
    render_chart(hybrid, target_figure=figure, target_axis=axis)
    points = [line for line in axis.lines if line.get_linestyle() == "None"]
    assert sum(np.asarray(line.get_xdata()).size for line in points) == len(hybrid["points"])
    assert all(bool(line.get_rasterized()) for line in points)
    artifacts = render_chart(hybrid)
    normal = render_chart(vector)
    # Agg publication PNG is byte-identical, regardless of vector container policy.
    assert artifacts["png"] == normal["png"]
    svg = ET.fromstring(artifacts["svg"])
    image = svg.find(".//{http://www.w3.org/2000/svg}image")
    assert image is not None
    pixels = Image.open(
        io.BytesIO(
            base64.b64decode(image.attrib["{http://www.w3.org/1999/xlink}href"].split(",")[1])
        )
    )
    assert abs(pixels.width / float(image.attrib["width"]) * 72 - 300) < 2
    assert abs(pixels.height / float(image.attrib["height"]) * 72 - 300) < 2
    assert b"<image" not in normal["svg"]
    assert b"Grouped raw points" in artifacts["svg"]
    assert b"/Subtype /Image" in artifacts["pdf"]


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_computation_batch_restores_gc_state_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, fail: bool
) -> None:
    from labviz_api import research_chart_routes
    from labviz_api.research_preparation import PreparationError

    original_compute = compute_chart
    original_render = render_chart
    observed = []

    def checked(prepared: dict[str, Any], spec: ScientificChartSpec) -> dict[str, Any]:
        observed.append(gc.isenabled())
        if fail:
            raise PreparationError("chart-test-failure", "Injected batch failure")
        return original_compute(prepared, spec)

    def checked_render(result: dict[str, Any]) -> dict[str, bytes]:
        assert not gc.isenabled()
        return original_render(result)

    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        revision = applied(client, identifier)
        monkeypatch.setattr(research_chart_routes, "compute_chart", checked)
        monkeypatch.setattr(research_chart_routes, "render_chart", checked_render)
        previous = gc.isenabled()
        (gc.enable if enabled else gc.disable)()
        try:
            response = client.post(
                f"/api/v1/projects/{identifier}/research/charts",
                json={"revisionId": revision, "spec": {}},
            )
            assert response.status_code == (422 if fail else 200), response.text
            assert observed == [False]
            assert gc.isenabled() == enabled
            assert (
                client.get(f"/api/v1/projects/{identifier}/research").json()["revisionId"]
                == revision
            )
        finally:
            (gc.enable if previous else gc.disable)()


def test_compact_catalogs_and_artifact_hash_preserve_older_header_compatibility(
    tmp_path: Path,
) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        revision = applied(client, identifier)
        base = f"/api/v1/projects/{identifier}/research"
        chart = client.post(
            base + "/charts",
            json={
                "revisionId": revision,
                "spec": {"kind": "violin", "title": "Immutable full spec"},
            },
        ).json()
        chart_url = base + "/charts/" + chart["id"]
        expected_charts = client.get(base + "/charts").json()
        expected_sources = client.get(base + "/figure-sources").json()
        original_png = client.get(chart_url + "/artifact/png").content
        with sqlite3.connect(config.database_path) as connection:
            frozen = connection.execute(
                "SELECT result_json FROM research_charts WHERE id=?", (chart["id"],)
            ).fetchone()[0]
            # The fast path must not evaluate the large JSON fallback when a complete
            # immutable header exists. Deliberate corruption is confined to this test DB.
            connection.execute(
                "UPDATE research_charts SET result_json='invalid-json' WHERE id=?", (chart["id"],)
            )
        assert client.get(chart_url).json() == chart
        assert client.get(base + "/charts").json() == expected_charts
        assert client.get(base + "/figure-sources").json() == expected_sources
        exported = client.get(chart_url + "/artifact/png")
        assert exported.content == original_png
        assert exported.headers["x-labviz-result-sha256"] == chart["resultSha256"]
        with sqlite3.connect(config.database_path) as connection:
            connection.execute(
                "UPDATE research_charts SET result_json=? WHERE id=?", (frozen, chart["id"])
            )
            # Headers written before this optimization have no full spec.
            connection.execute(
                "UPDATE research_panel_headers "
                "SET header_json=json_remove(header_json,'$.spec','$.firstPage') "
                "WHERE id=?",
                (chart["id"],),
            )
        assert client.get(chart_url).json() == chart
        assert client.get(base + "/charts").json() == expected_charts
        assert client.get(base + "/figure-sources").json() == expected_sources
        with sqlite3.connect(config.database_path) as connection:
            connection.execute("DELETE FROM research_panel_headers WHERE id=?", (chart["id"],))
        assert client.get(chart_url).json() == chart
        assert client.get(base + "/charts").json() == expected_charts
        assert client.get(base + "/figure-sources").json() == expected_sources
        exported = client.get(chart_url + "/artifact/png")
        assert exported.content == original_png
        assert exported.headers["x-labviz-result-sha256"] == chart["resultSha256"]
