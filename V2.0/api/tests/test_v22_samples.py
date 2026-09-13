from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from labviz_api.auth import AuthService, MemoryEmailSender
from labviz_api.config import Settings
from labviz_api.main import create_app
from labviz_api.models import ChartSpec, SampleExample
from labviz_api.processing import (
    ProcessingError,
    analyze_chart,
    build_quality_report,
    load_dataframe,
)
from labviz_api.repository import ProjectRepository

SAMPLES_ROOT = Path(__file__).resolve().parents[1] / "samples" / "v22"
MANIFEST_PATH = SAMPLES_ROOT / "manifest.json"


@pytest.fixture
def api_client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        database_path=tmp_path / "labviz-v22.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
    )
    repository = ProjectRepository(settings.database_path, settings.project_ttl_seconds)
    auth = AuthService(MemoryEmailSender(), settings.session_ttl_seconds, repository)
    with TestClient(create_app(settings, repository, auth)) as client:
        yield client


def _field(example: dict[str, Any], role: str) -> dict[str, Any] | None:
    return next((item for item in example["fields"] if item["role"] == role), None)


def _numeric_fields(example: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in example["fields"] if item["kind"] == "number"]


def _series(field: dict[str, Any], index: int = 0) -> dict[str, Any]:
    palette = ["#2563EB", "#0F766E", "#D97706", "#7C3AED", "#DB2777"]
    return {
        "field": field["name"],
        "label": field.get("titleEn", field["name"]),
        "color": palette[index % len(palette)],
    }


def _chart_for(example: dict[str, Any], export_format: str = "png") -> dict[str, Any]:
    chart_type = example["recommendedChart"]
    numeric = _numeric_fields(example)
    x_field = _field(example, "x") or (numeric[0] if numeric else example["fields"][0])
    y_field = _field(example, "y") or (numeric[1] if len(numeric) > 1 else numeric[0])
    series: list[dict[str, Any]]
    group_field: str | None = None

    if chart_type == "surface3d":
        y_surface = _field(example, "y") or numeric[1]
        z_surface = _field(example, "z") or numeric[2]
        series = [_series(y_surface), _series(z_surface, 1)]
        y_field = z_surface
    elif chart_type == "heatmap":
        series = [_series(item, index) for index, item in enumerate(numeric)]
        x_field, y_field = numeric[0], numeric[1]
    else:
        series = [_series(y_field)]
        grouping = _field(example, "group")
        if chart_type in {"line", "scatter"} and grouping:
            group_field = grouping["name"]
        if chart_type in {"histogram", "box"}:
            x_field = _field(example, "label") or grouping or x_field

    return {
        "schemaVersion": 1,
        "type": chart_type,
        "title": example["title"]["en"],
        "xAxis": {"field": x_field["name"], "title": x_field["name"], "unit": ""},
        "yAxis": {"field": y_field["name"], "title": y_field["name"], "unit": ""},
        "series": series,
        "groupField": group_field,
        "panelCount": 1,
        "export": {
            "format": export_format,
            "dpi": 300,
            "sizePreset": "single-column",
            "grayscalePreview": False,
        },
    }


def _ready_project(client: TestClient, slug: str) -> dict[str, Any]:
    response = client.post(f"/api/v1/samples/{slug}/projects")
    assert response.status_code == 202, response.text
    session = cast(dict[str, Any], response.json())
    job = client.get(f"/api/v1/jobs/{session['job']['id']}")
    assert job.status_code == 200, job.text
    assert job.json()["stage"] == "ready", job.text
    return session


def test_v22_manifest_is_versioned_synthetic_and_complete() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    examples = manifest["examples"]
    public = [item for item in examples if item["visibility"] == "public"]
    edges = [item for item in examples if item["visibility"] == "edge"]

    assert manifest["catalogVersion"] == "v1"
    assert manifest["release"] == "V2.2"
    assert manifest["syntheticOnly"] is True
    assert len(public) == 7
    assert len(edges) == 9
    assert {item["format"] for item in public} == {"csv", "tsv", "txt", "json", "xlsx"}
    assert {item["recommendedChart"] for item in public} == {
        "line",
        "scatter",
        "box",
        "histogram",
        "heatmap",
        "surface3d",
    }
    assert len({item["slug"] for item in examples}) == len(examples)

    for item in examples:
        payload_path = SAMPLES_ROOT / item["filename"]
        assert payload_path.is_file()
        assert item["synthetic"] is True
        assert item["byteSize"] == payload_path.stat().st_size
        if item["visibility"] == "public":
            SampleExample.model_validate(item)


def test_v22_public_samples_run_import_preview_quality_analysis_and_exports(
    api_client: TestClient,
) -> None:
    client = api_client
    catalog_response = client.get("/api/v1/samples")
    assert catalog_response.status_code == 200, catalog_response.text
    catalog = catalog_response.json()
    assert catalog["catalogVersion"] == "v1"
    assert catalog["syntheticOnly"] is True
    assert len(catalog["examples"]) == 7

    for example in catalog["examples"]:
        session = _ready_project(client, example["slug"])
        project_id = session["projectId"]
        preview = client.get(f"/api/v1/projects/{project_id}/preview")
        quality = client.get(f"/api/v1/projects/{project_id}/quality")
        workspace = client.get(f"/api/v1/projects/{project_id}/workspace")
        assert preview.status_code == 200, preview.text
        assert quality.status_code == 200, quality.text
        assert workspace.status_code == 200, workspace.text
        assert preview.json()["totalRows"] == example["rowCount"]
        quality_kinds = {item["kind"] for item in quality.json()["findings"]}
        assert set(example["expectedQualityKinds"]).issubset(quality_kinds)
        cleaning = client.patch(
            f"/api/v1/projects/{project_id}/cleaning-decisions",
            json={
                "decisions": [
                    {"findingId": finding["id"], "action": "ignore"}
                    for finding in quality.json()["findings"]
                ]
            },
        )
        assert cleaning.status_code == 200, cleaning.text

        chart = _chart_for(example)
        saved = client.put(f"/api/v1/projects/{project_id}/chart", json={"chart": chart})
        analysis = client.post(
            f"/api/v1/projects/{project_id}/chart-analysis", json={"chart": chart}
        )
        cleaned = client.get(f"/api/v1/projects/{project_id}/exports/cleaned-data.csv")
        assert saved.status_code == 200, saved.text
        assert analysis.status_code == 200, analysis.text
        assert cleaned.status_code == 200, cleaned.text
        assert cleaned.content.startswith(b"\xef\xbb\xbf")
        assert analysis.json()["preview"] is not None
        assert "recommendations" in analysis.json()

        for export_format, signature in (
            ("png", b"\x89PNG\r\n\x1a\n"),
            ("svg", b"<?xml"),
            ("pdf", b"%PDF"),
        ):
            export = client.post(
                f"/api/v1/projects/{project_id}/exports",
                json={"chart": _chart_for(example, export_format)},
            )
            assert export.status_code == 200, export.text
            assert export.json()["status"] == "ready"
            download = client.get(export.json()["downloadUrl"])
            assert download.status_code == 200, download.text
            assert download.content.startswith(signature)


def test_v22_edge_fixtures_cover_quality_grid_encoding_and_unicode_filename(
    api_client: TestClient,
) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    edge_entries = [item for item in manifest["examples"] if item["visibility"] == "edge"]

    for entry in edge_entries:
        payload = (SAMPLES_ROOT / entry["filename"]).read_bytes()
        if entry["slug"] == "duplicate-columns":
            with pytest.raises(ProcessingError, match="unique") as error:
                load_dataframe(payload, entry["filename"])
            assert error.value.code == "duplicate-columns"
            continue

        frame, _sheet, _sheets, _header = load_dataframe(payload, entry["filename"])
        quality = build_quality_report("fixture", frame)
        quality_kinds = {item["kind"] for item in quality["findings"]}
        if entry["slug"] == "bom-header":
            assert "\ufeff" not in str(frame.columns[0])
        elif entry["slug"] == "unicode-filename":
            assert "时间" in str(frame.columns[0])
        elif entry["slug"].startswith("invalid-surface"):
            chart = {
                "schemaVersion": 1,
                "type": "surface3d",
                "title": "Invalid surface",
                "xAxis": {"field": "x", "title": "X", "unit": ""},
                "yAxis": {"field": "z", "title": "Z", "unit": ""},
                "series": [
                    {"field": "y", "label": "Y", "color": "#2563EB"},
                    {"field": "z", "label": "Z", "color": "#0F766E"},
                ],
                "panelCount": 1,
                "export": {
                    "format": "png",
                    "dpi": 300,
                    "sizePreset": "single-column",
                    "grayscalePreview": False,
                },
            }
            preview = analyze_chart(frame, ChartSpec.model_validate(chart))["preview"]
            status = preview["surfaceDiagnostics"][0]["status"]
            assert f"surface-{status}" in entry["expectedErrors"]
        else:
            assert set(entry["expectedErrors"]).intersection(quality_kinds)

    unicode_payload = (SAMPLES_ROOT / "edge_中文文件名.csv").read_bytes()
    uploaded = api_client.post(
        "/api/v1/projects",
        files={"file": ("实验数据-中文.csv", unicode_payload, "text/csv")},
    )
    assert uploaded.status_code == 202, uploaded.text
    project_id = uploaded.json()["projectId"]
    job = api_client.get(f"/api/v1/jobs/{uploaded.json()['job']['id']}")
    assert job.json()["stage"] == "ready", job.text
    cleaned = api_client.get(f"/api/v1/projects/{project_id}/exports/cleaned-data.csv")
    assert cleaned.status_code == 200
    assert "filename*=UTF-8''" in cleaned.headers["content-disposition"]
