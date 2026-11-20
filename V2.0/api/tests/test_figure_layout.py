"""FigureSpec v2 saved as a board: layouts up to six panels, axis controls and annotations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_research_api import project, settings, unlock
from test_research_figure_board import charts

from labviz_api.figure_spec import FigureSpec, panel_cells
from labviz_api.main import create_app
from labviz_api.plan_axes import axis_position
from labviz_api.render_plan import RenderPlan
from labviz_api.research_figure_render import check_figure
from labviz_api.research_preparation import PreparationError

MOSAIC = ["AAB", "CDE", "CDF"]


def figure(ids: list[str], labels: str, **extra: Any) -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "panels": [
            {
                "id": f"p{index + 1}",
                "label": label,
                "layers": [{"kind": "chart", "chartId": ids[index % len(ids)]}],
            }
            for index, label in enumerate(labels)
        ],
        **extra,
    }


@pytest.fixture
def saved(tmp_path: Path) -> Any:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        ids = [item["id"] for item in charts(client, identifier)]
        yield client, f"/api/v1/projects/{identifier}/research/figure-boards", ids


def test_cells_of_grid_and_mosaic() -> None:
    grid = FigureSpec.model_validate(figure(["c"], "ABCDE", layout={"rows": 2, "cols": 3}))
    assert panel_cells(grid) == [
        (0, 0, 1, 1),
        (0, 1, 1, 1),
        (0, 2, 1, 1),
        (1, 0, 1, 1),
        (1, 1, 1, 1),
    ]
    mosaic = FigureSpec.model_validate(
        figure(["c"], "ABCDEF", layout={"kind": "mosaic", "rows": 3, "cols": 3, "mosaic": MOSAIC})
    )
    assert panel_cells(mosaic) == [
        (0, 0, 1, 2),
        (0, 2, 1, 1),
        (1, 0, 2, 1),
        (1, 1, 2, 1),
        (1, 2, 1, 1),
        (2, 2, 1, 1),
    ]


@pytest.mark.parametrize(
    ("labels", "layout"),
    [
        ("ABC", {"rows": 1, "cols": 3}),
        ("ABCDE", {"rows": 2, "cols": 3}),
        ("ABCDEF", {"kind": "mosaic", "rows": 3, "cols": 3, "mosaic": MOSAIC}),
    ],
)
def test_three_five_and_six_panels_save_and_reopen(
    saved: Any, labels: str, layout: dict[str, Any]
) -> None:
    client, base, ids = saved
    response = client.post(base, json=figure(ids, labels, layout=layout, title="Layout"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["spec"]["schemaVersion"] == 2 and body["composed"] is True
    assert [panel["label"] for panel in body["panels"]] == list(labels)
    assert [panel["chartId"] for panel in body["panels"]] == [
        ids[index % len(ids)] for index in range(len(labels))
    ]
    url = base + "/" + body["id"]
    assert client.get(url).json() == body
    for label in labels:
        plan = RenderPlan.model_validate_json(client.get(f"{url}/plans/{label}").content)
        assert plan.panels[0].id == label
    assert client.get(url + "/artifact/png").content.startswith(b"\x89PNG")
    listed = client.get(base).json()["boards"]
    assert listed[0]["spec"]["schemaVersion"] == 2


def test_a_v1_board_is_still_stored_as_v1_and_headed_by_its_label(saved: Any) -> None:
    client, base, ids = saved
    plain = client.post(base, json={"panelChartIds": ids}).json()
    assert plain["spec"]["schemaVersion"] == 1
    plan = json.loads(client.get(f"{base}/{plain['id']}/plans/B").content)
    assert plan["panels"][0]["title"].startswith("B · ")


def test_a_log_axis_and_a_bracket_reach_the_stored_plan(saved: Any) -> None:
    client, base, ids = saved

    def stored(spec: dict[str, Any]) -> dict[str, Any]:
        response = client.post(base, json=spec)
        assert response.status_code == 200, response.text
        return dict(json.loads(client.get(f"{base}/{response.json()['id']}/plans/A").content))

    plain = stored(figure(ids, "A"))
    logged_spec = figure(ids, "A")
    logged_spec["panels"][0]["axes"] = {"y": {"scale": "log10"}}
    logged = stored(logged_spec)["panels"][0]
    low, high = logged["axes"]["y"]["domain"]
    middle = (low * high) ** 0.5
    left, right = logged["axes"]["x"]["domain"]
    first, second = left + (right - left) / 4, right - (right - left) / 4
    noted_spec = figure(ids, "A")
    noted_spec["panels"][0]["axes"] = {"y": {"scale": "log10"}}
    noted_spec["panels"][0]["annotations"] = [
        {"kind": "bracket", "x": [first, second], "y": middle, "label": "p < 0.01"},
        {"kind": "hline", "value": middle, "label": "limit"},
    ]
    plan = stored(noted_spec)
    RenderPlan.model_validate_json(json.dumps(plan))
    panel = plan["panels"][0]
    assert panel["axes"] == logged["axes"] and panel["axes"]["y"]["scale"] == "log10"
    assert panel["axes"]["x"] == plain["panels"][0]["axes"]["x"]
    bracket, line = panel["notes"]
    assert bracket["kind"] == "bracket"
    assert line == {"kind": "hline", "value": middle, "label": "limit"}
    assert axis_position("log10", bracket["foot"]) < axis_position("log10", middle)
    assert panel["marks"] == plain["panels"][0]["marks"]
    assert "notes" not in plain["panels"][0] and "notes" not in logged
    assert len({plan["planHash"], plain["planHash"]}) == 2


def test_label_style_and_canvas_reach_the_export(saved: Any) -> None:
    client, base, ids = saved
    spec = figure(
        ids,
        "AB",
        layout={"rows": 1, "cols": 2, "labelStyle": "lower"},
        canvas={"sizePreset": "custom", "width": 180, "height": 80, "unit": "mm", "dpi": 150},
    )
    body = client.post(base, json=spec).json()
    plan = json.loads(client.get(f"{base}/{body['id']}/plans/B").content)
    assert plan["panels"][0]["title"].startswith("b · ") and plan["panels"][0]["id"] == "B"
    png = client.get(f"{base}/{body['id']}/artifact/png").content
    width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    # Matplotlib truncates the pixel size: 1062.99 px wide is 1062.
    assert (width, height) == (int(180 / 25.4 * 150), int(80 / 25.4 * 150))


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (
            {"layers": [{"kind": "scatter", "encoding": {"x": "a", "y": "b"}}]},
            "figure-layer-unsupported",
        ),
        ({"annotations": [{"kind": "hline", "value": 1e12}]}, "annotation-outside"),
        ({"axes": {"x": {"scale": "log10"}}}, "axis-control-category"),
    ],
)
def test_refused_figures(saved: Any, change: dict[str, Any], code: str) -> None:
    client, base, ids = saved
    spec = figure(ids, "A")
    spec["panels"][0] |= change
    response = client.post(base, json=spec)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == code
    assert client.get(base).json()["boards"] == []


def test_a_shared_axis_refuses_a_scale_on_it_and_an_unknown_palette_is_refused(saved: Any) -> None:
    client, base, ids = saved
    spec = figure([ids[0]], "AB", layout={"rows": 1, "cols": 2}, axisMode="shared_y")
    spec["panels"][0]["axes"] = {"y": {"scale": "log10"}}
    assert client.post(base, json=spec).json()["code"] == "figure-shared-axis-control"
    other = figure(ids, "A", palette={"categorical": "neon"})
    assert client.post(base, json=other).json()["code"] == "figure-palette-unknown"


def test_the_deferred_path_builds_a_six_panel_figure(saved: Any) -> None:
    client, base, ids = saved
    layout = {"kind": "mosaic", "rows": 3, "cols": 3, "mosaic": MOSAIC}
    response = client.post(
        base, params={"deferArtifacts": True}, json=figure(ids, "ABCDEF", layout=layout)
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert [panel["label"] for panel in body["panels"]] == list("ABCDEF")
    deadline = perf_counter() + 120
    while perf_counter() < deadline:
        view = client.get(base + "/" + body["id"]).json()
        if view["artifactState"] in ("ready", "failed"):
            break
    assert view["artifactState"] == "ready", view
    assert client.get(f"{base}/{body['id']}/plans/F").status_code == 200


@pytest.mark.parametrize("body", [[], "x", 3, {"schemaVersion": "2"}, {"schemaVersion": 3}])
def test_a_body_that_is_no_figure_is_a_422(saved: Any, body: Any) -> None:
    client, base, _ = saved
    assert client.post(base, json=body).status_code == 422


@pytest.mark.parametrize(("style", "first"), [("lower", "a: "), ("none", "Panel 1: ")])
def test_output_notes_name_panels_as_the_figure_draws_them(
    saved: Any, style: str, first: str
) -> None:
    client, base, ids = saved
    spec = figure(ids, "AB", layout={"rows": 1, "cols": 2, "labelStyle": style})
    board = client.post(base, json=spec).json()["id"]
    project_url = base.rsplit("/", 1)[0]
    preview = client.post(project_url + "/outputs/preview", json={"boardId": board})
    assert preview.status_code == 200, preview.text
    notes = preview.json()["notes"]
    assert notes["legendEn"].startswith(first) and notes["methodsEn"].startswith(first)


def test_the_deferred_save_refuses_what_the_figure_alone_decides(saved: Any) -> None:
    client, base, ids = saved
    spec = figure([ids[0]], "AB", layout={"rows": 1, "cols": 2}, axisMode="shared_y")
    spec["panels"][0]["axes"] = {"y": {"scale": "log10"}}
    response = client.post(base, params={"deferArtifacts": True}, json=spec)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "figure-shared-axis-control"
    assert client.get(base).json()["boards"] == []


def test_a_deferred_build_the_plan_refuses_keeps_its_reason_and_is_not_retried(saved: Any) -> None:
    client, base, ids = saved
    spec = figure(ids, "A")
    spec["panels"][0]["annotations"] = [{"kind": "hline", "value": 1e12, "label": "far"}]
    response = client.post(base, params={"deferArtifacts": True}, json=spec)
    assert response.status_code == 202, response.text
    url = base + "/" + response.json()["id"]
    deadline = perf_counter() + 120
    while perf_counter() < deadline:
        view = client.get(url).json()
        if view["artifactState"] in ("ready", "failed"):
            break
    assert view["artifactState"] == "failed", view
    assert view["artifactRefusal"] == "annotation-outside"
    assert "Retry" not in view["artifactError"]
    retried = client.post(url + "/retry")
    assert retried.status_code == 422, retried.text
    assert retried.json()["code"] == "figure-build-refused"


def test_duplicating_a_project_copies_a_v2_figure_onto_the_copied_charts(saved: Any) -> None:
    client, base, ids = saved
    client.post(base, json=figure(ids, "ABC", layout={"rows": 1, "cols": 3}))
    identifier = base.split("/")[4]
    response = client.post(f"/api/v1/projects/{identifier}/duplicate")
    assert response.status_code == 200, response.text
    copy = response.json()["projectId"]
    copied_base = base.replace(identifier, copy)
    (copied,) = client.get(copied_base).json()["boards"]
    copied_charts = {
        chart["id"]
        for chart in client.get(f"/api/v1/projects/{copy}/research/charts").json()["charts"]
    }
    layers = [panel["layers"][0]["chartId"] for panel in copied["spec"]["panels"]]
    assert copied["spec"]["schemaVersion"] == 2 and len(layers) == 3
    assert set(layers) <= copied_charts and not set(layers) & set(ids)
    png = client.get(f"{copied_base}/{copied['id']}/artifact/png")
    assert png.content.startswith(b"\x89PNG")


def test_a_v2_figure_that_v1_can_express_makes_a_template(saved: Any) -> None:
    client, base, ids = saved
    body = client.post(base, json=figure(ids, "AB", layout={"rows": 1, "cols": 2})).json()
    assert body["spec"]["schemaVersion"] == 2
    project_url = base.rsplit("/", 1)[0]
    response = client.post(project_url + "/templates", json={"boardId": body["id"], "name": "T"})
    assert response.status_code == 200, response.text


def test_a_template_is_refused_for_a_v2_figure(saved: Any) -> None:
    client, base, ids = saved
    body = client.post(base, json=figure(ids, "ABC", layout={"rows": 1, "cols": 3})).json()
    project_url = base.rsplit("/", 1)[0]
    response = client.post(project_url + "/templates", json={"boardId": body["id"], "name": "T"})
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "reuse-figure-unsupported"


@pytest.mark.parametrize(
    ("extra", "refused"),
    [
        ({"layout": {"kind": "grid", "rows": 1, "cols": 6}, "canvas": {"dpi": 300}}, False),
        ({"layout": {"kind": "grid", "rows": 1, "cols": 6}, "canvas": {"dpi": 1200}}, True),
        ({"canvas": {"sizePreset": "custom", "width": 180, "height": 120, "dpi": 1200}}, False),
        ({"canvas": {"sizePreset": "custom", "width": 1000, "height": 1000, "dpi": 1200}}, True),
    ],
)
def test_a_figure_too_large_to_draw_is_refused(extra: dict[str, Any], refused: bool) -> None:
    count = extra.get("layout", {}).get("cols", 1)
    spec = FigureSpec.model_validate(figure(["c"], "ABCDEF"[:count], **extra))
    if refused:
        with pytest.raises(PreparationError) as caught:
            check_figure(spec)
        assert caught.value.code == "figure-canvas-too-large"
    else:
        check_figure(spec)


def artifact(client: TestClient, url: str, kind: str) -> bytes:
    response = client.get(f"{url}/artifact/{kind}")
    assert response.status_code == 200, response.text
    return bytes(response.content)


def assert_text_is_text(pdf: bytes, svg: bytes) -> None:
    """Journals refuse Type 3 fonts, and outlined SVG text cannot be edited."""
    assert b"/Type3" not in pdf and b"/FontFile2" in pdf
    assert b"<text" in svg


@pytest.mark.parametrize("version", [1, 2])
def test_exports_keep_text_and_leave_the_provenance_footer_off(saved: Any, version: int) -> None:
    client, base, ids = saved
    body = {"panelChartIds": ids} if version == 1 else figure(ids, "AB", layout={"cols": 2})
    board = client.post(base, json=body).json()
    assert board["spec"]["schemaVersion"] == version
    url = f"{base}/{board['id']}"
    svg = artifact(client, url, "svg")
    assert_text_is_text(artifact(client, url, "pdf"), svg)
    assert b"Input rows=" not in svg
    # The footer's text stays in the panel plan and the output notes.
    plan = json.loads(client.get(f"{url}/plans/A").content)
    assert plan["disclosures"][0].startswith("Input rows=")


def test_the_provenance_footer_can_be_drawn(saved: Any) -> None:
    client, base, ids = saved
    board = client.post(base, json=figure(ids, "A", provenanceOnFigure=True)).json()
    assert b"Input rows=" in artifact(client, f"{base}/{board['id']}", "svg")


def test_research_chart_exports_keep_text(saved: Any) -> None:
    client, base, ids = saved
    url = base.replace("figure-boards", "charts") + f"/{ids[0]}"
    assert_text_is_text(artifact(client, url, "pdf"), artifact(client, url, "svg"))


def test_a_chosen_font_reaches_every_text_of_the_figure(saved: Any) -> None:
    from matplotlib import font_manager

    client, base, ids = saved
    board = client.post(
        base, json=figure(ids, "AB", layout={"cols": 2}, title="Dose", fontFamily="Times New Roman")
    ).json()
    url = f"{base}/{board['id']}"
    svg = artifact(client, url, "svg").decode()
    families = re.findall(r"font-family: ([^;\"]+)", svg)
    assert families and all(family.startswith("'Times New Roman'") for family in families)
    pdf = artifact(client, url, "pdf")
    assert b"/Type3" not in pdf
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError:
        return  # The CI runner has no Times New Roman; Windows checks the embedded font.
    assert set(re.findall(rb"/BaseFont /[A-Z]{6}\+(\w+)", pdf)) == {b"TimesNewRomanPSMT"}


def test_chinese_labels_embed_their_glyphs_in_the_svg(saved: Any) -> None:
    client, base, ids = saved
    board = client.post(base, json=figure(ids, "A", title="剂量反应")).json()
    url = f"{base}/{board['id']}"
    svg = artifact(client, url, "svg")
    assert_text_is_text(artifact(client, url, "pdf"), svg)
    assert b"@font-face" in svg and "剂量反应".encode() in svg


def test_the_count_line_can_be_left_out(saved: Any) -> None:
    client, base, ids = saved

    def ticks(spec: dict[str, Any]) -> list[str]:
        board = client.post(base, json=spec).json()
        plan = json.loads(client.get(f"{base}/{board['id']}/plans/A").content)
        return [tick["label"] for tick in plan["panels"][0]["axes"]["x"]["ticks"]]

    counted, plain = ticks(figure(ids, "A")), ticks(figure(ids, "A", categoryCounts=False))
    assert all("\nobs=" in label for label in counted)
    assert plain == [label.split("\n")[0] for label in counted]


@pytest.mark.parametrize(
    "extra", [{"fontFamily": "Arial"}, {"categoryCounts": False}, {"provenanceOnFigure": True}]
)
def test_a_template_is_refused_for_what_v1_cannot_draw(saved: Any, extra: dict[str, Any]) -> None:
    client, base, ids = saved
    body = client.post(base, json=figure(ids, "AB", layout={"cols": 2}, **extra)).json()
    project_url = base.rsplit("/", 1)[0]
    response = client.post(project_url + "/templates", json={"boardId": body["id"], "name": "T"})
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "reuse-figure-unsupported"
    assert "font choice" in response.json()["message"]
