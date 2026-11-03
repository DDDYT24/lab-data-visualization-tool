from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from test_research_api import SOURCE, project, settings, unlock
from test_research_chart_api import applied
from test_research_figure_composition import legacy

from labviz_api.main import create_app
from labviz_api.research_outputs import csv_bytes


def setup(client: TestClient) -> tuple[str, str, str, str]:
    unlock(client)
    identifier = project(client)
    revision = applied(client, identifier)
    base = f"/api/v1/projects/{identifier}/research"
    chart = client.post(base + "/charts", json={"revisionId": revision, "spec": {}})
    assert chart.status_code == 200, chart.text
    board = client.post(base + "/figure-boards", json={"panelChartIds": [chart.json()["id"]]})
    assert board.status_code == 200, board.text
    template = client.post(
        base + "/templates", json={"boardId": board.json()["id"], "name": "Recipe"}
    )
    assert template.status_code == 200, template.text
    return base, revision, board.json()["id"], template.json()["id"]


def preview(
    client: TestClient,
    base: str,
    template: str,
    revision: str,
    source: bytes = SOURCE,
    mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = client.post(
        base + f"/templates/{template}/preview",
        data={"mapping": json.dumps(mapping or {}), "expectedRevision": revision},
        files={"file": ("replacement.csv", source, "text/csv")},
    )
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], response.json())


def test_renamed_replacement_is_previewed_applied_once_and_data_free(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        base, revision, board, template = setup(client)
        recipe = client.get(base + f"/templates/{template}").json()
        assert recipe["figure"]["panelChartIds"] == ["panel-0"]
        assert recipe["panels"][0]["plan"]["exclusions"] == []
        text = json.dumps(recipe)
        assert revision not in text and board not in text
        assert "observations" not in recipe and "raw" not in recipe
        replacement = SOURCE.replace(b",v,", b",voltage,").replace(b",7,", b",17,")
        check = preview(client, base, template, revision, replacement, {"v": "voltage"})
        assert check["readyCount"] == 1 and check["figureReady"]
        assert client.get(base).json()["revisionId"] == revision
        response = client.post(
            base + "/template-apply", json={"token": check["token"], "expectedRevision": revision}
        )
        assert response.status_code == 200, response.text
        assert response.json()["boardId"] != board
        assert client.get(base).json()["revisionId"] != revision
        assert (
            client.post(
                base + "/template-apply",
                json={"token": check["token"], "expectedRevision": revision},
            ).status_code
            == 409
        )
        result = client.get(base + "/charts/" + response.json()["chartIds"][0]).json()
        assert any(p["value"] == 17 for p in result["rows"])


def test_partial_panel_unit_block_and_stale_preview(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        base, revision, _, template = setup(client)
        recipe = client.get(base + f"/templates/{template}").json()
        recipe["panels"].append({**recipe["panels"][0], "expectedUnit": "V"})
        recipe["figure"]["panelChartIds"] = ["panel-0", "panel-1"]
        imported = client.post(base + "/templates/import", json=recipe)
        assert imported.status_code == 200, imported.text
        check = preview(client, base, imported.json()["id"], revision)
        assert [p["status"] for p in check["panels"]] == ["ready", "blocked"]
        assert not check["figureReady"]
        result = client.post(
            base + "/template-apply", json={"token": check["token"], "expectedRevision": revision}
        )
        assert result.status_code == 200, result.text
        assert result.json()["boardId"] is None and len(result.json()["chartIds"]) == 1
        revision = result.json()["revisionId"]
        check = preview(client, base, template, revision)
        applied(client, base.split("/")[4])
        assert (
            client.post(
                base + "/template-apply",
                json={"token": check["token"], "expectedRevision": revision},
            ).status_code
            == 409
        )
        recipe["script"] = "untrusted"
        assert client.post(base + "/templates/import", json=recipe).status_code == 422


def test_frozen_output_files_hashes_notes_copy_and_ownership(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        base, revision, board, template = setup(client)
        response = client.post(base + "/outputs/preview", json={"boardId": board})
        assert response.status_code == 200, response.text
        check = response.json()
        assert not check["manifest"]["originalUploadsIncluded"]
        assert check["manifest"]["panels"][0]["sourceCsvRows"] == 4
        check["notes"]["legendZh"] = "研究员编辑的图注"
        body = {"token": check["token"], "notes": check["notes"], "reviewedIncludedData": False}
        assert client.post(base + "/outputs", json=body).status_code == 422
        body["reviewedIncludedData"] = True
        response = client.post(base + "/outputs", json=body)
        assert response.status_code == 200, response.text
        saved = response.json()
        download = client.get(base + f"/outputs/{saved['id']}/download")
        assert hashlib.sha256(download.content).hexdigest() == saved["sha256"]
        assert "no-store" in download.headers["cache-control"]
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            for f in manifest["files"]:
                content = archive.read(f["name"])
                assert len(content) == f["bytes"]
                assert hashlib.sha256(content).hexdigest() == f["sha256"]
            rows = list(
                csv.DictReader(io.StringIO(archive.read("panels/A-source.csv").decode("utf-8-sig")))
            )
            assert {float(r["derivedValue"]) for r in rows} == {1, 3, 5, 7}
            assert all(float(r["reportedText"]) == float(r["derivedValue"]) for r in rows)
            assert "研究员编辑的图注" in archive.read("notes.zh.txt").decode()
        identifier = base.split("/")[4]
        applied(client, identifier, SOURCE.replace(b",7,", b",27,"))
        assert client.get(base + f"/outputs/{saved['id']}/download").content == download.content
        other = project(client)
        assert (
            client.get(
                f"/api/v1/projects/{other}/research/outputs/{saved['id']}/download"
            ).status_code
            == 404
        )
        duplicate = client.post(f"/api/v1/projects/{identifier}/duplicate").json()["projectId"]
        copied_base = f"/api/v1/projects/{duplicate}/research"
        copied = client.get(copied_base + "/outputs").json()["outputs"][0]
        assert len(client.get(copied_base + "/templates").json()["templates"]) == 1
        assert client.delete(f"/api/v1/projects/{identifier}").status_code == 204
        assert (
            client.get(copied_base + f"/outputs/{copied['id']}/download").content
            == download.content
        )


def test_csv_escapes_formula_text_but_preserves_negative_numbers() -> None:
    rows = list(
        csv.DictReader(
            io.StringIO(
                csv_bytes([{"name": "=SUM(1)", "value": -2}], ["name", "value"]).decode("utf-8-sig")
            )
        )
    )
    assert rows[0] == {"name": "'=SUM(1)", "value": "-2"}


@pytest.mark.parametrize("kind", ["line", "scatter", "box", "heatmap"])
def test_legacy_recipe_replacement_and_output(tmp_path: Path, kind: str) -> None:
    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier, _, source_id = legacy(client, kind)
        base = f"/api/v1/projects/{identifier}/research"
        board = client.post(base + "/figure-boards", json={"panelChartIds": [source_id]}).json()
        created = client.post(base + "/templates", json={"boardId": board["id"], "name": kind})
        assert created.status_code == 200, created.text
        check = client.post(
            base + f"/templates/{created.json()['id']}/preview",
            data={"mapping": json.dumps({"y": "signal"})},
            files={
                "file": ("new.csv", b"x,signal,z\n1,20,4\n2,40,3\n3,60,2\n4,80,1\n", "text/csv")
            },
        )
        assert check.status_code == 200, check.text
        assert check.json()["readyCount"] == 1, check.text
        saved = client.post(base + "/template-apply", json={"token": check.json()["token"]})
        assert saved.status_code == 200, saved.text
        output = client.post(base + "/outputs/preview", json={"boardId": saved.json()["boardId"]})
        assert output.status_code == 200, output.text
        assert output.json()["manifest"]["panels"][0]["sourceCsvRows"] == 4


def test_schema_files_match_models() -> None:
    from labviz_api.research_charts import ScientificChartSpec
    from labviz_api.research_outputs import OutputNotes
    from labviz_api.research_reuse import WorkflowTemplate

    models: list[tuple[str, type[BaseModel]]] = [
        ("scientific-chart-v1", ScientificChartSpec),
        ("workflow-template-v1", WorkflowTemplate),
        ("research-output-notes-v1", OutputNotes),
    ]
    for name, model in models:
        path = Path(__file__).parents[2] / "contracts" / (name + ".schema.json")
        assert json.loads(path.read_text()) == model.model_json_schema()


def test_recipe_replays_conversion_drops_exclusions_and_keeps_group_styles(tmp_path: Path) -> None:
    from test_research_api import PLAN

    with TestClient(create_app(settings(tmp_path))) as client:
        unlock(client)
        identifier = project(client)
        base = f"/api/v1/projects/{identifier}/research"
        source = (
            b"s,repeat,v,unit,group\na,1,1000,mV,A\na,2,3000,mV,A\nb,1,5000,mV,B\nb,2,7000,mV,B\n"
        )
        plan = {
            **cast(dict[str, Any], PLAN),
            "mapping": {**cast(dict[str, str], PLAN["mapping"]), "group": "group"},
            "steps": [
                {"kind": "convert_unit", "valueField": "v", "unitField": "unit", "toUnit": "V"}
            ],
        }
        revision = applied(client, identifier, source, plan)
        view = client.get(base).json()
        excluded = view["data"]["observations"][0]["id"]
        plan["exclusions"] = [{"observationId": excluded, "reason": "OLD_ROW_REASON"}]
        revision = applied(client, identifier, source, plan)
        chart = client.post(
            base + "/charts",
            json={
                "revisionId": revision,
                "spec": {"categoryStyles": [{"group": "B", "color": "#aabbcc", "marker": "D"}]},
            },
        ).json()
        board = client.post(base + "/figure-boards", json={"panelChartIds": [chart["id"]]}).json()
        template = client.post(
            base + "/templates", json={"boardId": board["id"], "name": "Conversion"}
        ).json()
        assert "OLD_ROW_REASON" not in json.dumps(template)
        assert excluded not in json.dumps(template)
        replacement = source.replace(b",A", b",C").replace(b",v,", b",voltage,")
        check = preview(client, base, template["id"], revision, replacement, {"v": "voltage"})
        assert check["readyCount"] == 1 and check["panels"][0]["observations"] == 4
        styles = {s["group"]: s for s in check["panels"][0]["styles"]}
        assert styles["B"] == {"group": "B", "color": "#aabbcc", "marker": "D"}
        assert "C" in styles
        saved = client.post(
            base + "/template-apply", json={"token": check["token"], "expectedRevision": revision}
        ).json()
        result = client.get(base + "/charts/" + saved["chartIds"][0]).json()
        assert {r["value"] for r in result["rows"]} == {1, 3, 5, 7}
        assert {r["unit"] for r in result["rows"]} == {"V"}
        assert {r["reportedUnit"] for r in result["rows"]} == {"mV"}
        svg = client.get(base + "/charts/" + saved["chartIds"][0] + "/artifact/svg").text
        assert "#aabbcc" in svg
