from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook

from labviz_api.auth import AuthService, MemoryEmailSender
from labviz_api.config import Settings
from labviz_api.main import create_app
from labviz_api.models import ChartSpec
from labviz_api.processing import (
    ProcessingError,
    analyze_chart,
    build_preview,
    build_quality_report,
    render_chart,
)
from labviz_api.repository import ProjectRepository


@pytest.fixture
def api_client(tmp_path: Path) -> Iterator[tuple[TestClient, MemoryEmailSender]]:
    settings = Settings(
        database_path=tmp_path / "labviz-test.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
    )
    repository = ProjectRepository(settings.database_path, settings.project_ttl_seconds)
    sender = MemoryEmailSender()
    auth = AuthService(sender, settings.session_ttl_seconds, repository)
    with TestClient(create_app(settings, repository, auth)) as client:
        yield client, sender


def _sample_project(client: TestClient) -> dict[str, Any]:
    response = client.post("/api/v1/samples/thermal-response/projects")
    assert response.status_code == 202
    session = response.json()
    assert session["apiVersion"] == "v1"
    assert session["storageMode"] == "temporary-cloud"

    job = client.get(f"/api/v1/jobs/{session['job']['id']}")
    assert job.status_code == 200
    assert job.json()["stage"] == "ready"
    return cast(dict[str, Any], session)


def _chart(format_name: str = "png") -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "type": "line",
        "title": "Thermal response",
        "xAxis": {"field": "Time (min)", "title": "Time", "unit": "min"},
        "yAxis": {
            "field": "Response (mV)",
            "title": "Response",
            "unit": "mV",
        },
        "series": [
            {
                "field": "Response (mV)",
                "label": "Sample A",
                "color": "#2563EB",
            }
        ],
        "panelCount": 1,
        "export": {
            "format": format_name,
            "dpi": 300,
            "sizePreset": "double-column",
            "grayscalePreview": False,
        },
    }


def _sign_in(client: TestClient, sender: MemoryEmailSender) -> str:
    requested = client.post("/api/v1/auth/email-code", json={"email": "researcher@example.com"})
    assert requested.status_code == 200
    assert requested.json()["deliveryMode"] == "email"
    verified = client.post(
        "/api/v1/auth/email-code/verify",
        json={
            "challengeId": requested.json()["challengeId"],
            "code": sender.messages[-1][1],
        },
    )
    assert verified.status_code == 200
    token = client.cookies.get("labviz_session")
    assert token
    return token


def test_processing_quality_cleaning_and_publication_exports(
    api_client: tuple[TestClient, MemoryEmailSender],
) -> None:
    client, _sender = api_client
    session = _sample_project(client)
    project_id = session["projectId"]

    preview = client.get(f"/api/v1/projects/{project_id}/preview")
    assert preview.status_code == 200
    assert preview.json()["totalRows"] == 120
    assert preview.json()["sampled"] is False
    assert preview.json()["columns"][0]["unit"] == "min"

    quality = client.get(f"/api/v1/projects/{project_id}/quality")
    assert quality.status_code == 200
    quality_body = quality.json()
    assert quality_body["missingValues"] == 1
    assert quality_body["suspiciousPoints"] >= 1

    finding_id = quality_body["findings"][0]["id"]
    decisions = client.patch(
        f"/api/v1/projects/{project_id}/cleaning-decisions",
        json={"decisions": [{"findingId": finding_id, "action": "exclude"}]},
    )
    assert decisions.status_code == 200
    assert decisions.json()["decisions"][0]["findingId"] == finding_id

    saved = client.put(f"/api/v1/projects/{project_id}/chart", json={"chart": _chart()})
    assert saved.status_code == 200
    assert saved.json()["chart"]["title"] == "Thermal response"

    expected_files = {
        "png": ("image/png", b"\x89PNG\r\n\x1a\n"),
        "svg": ("image/svg+xml", b"<?xml"),
        "pdf": ("application/pdf", b"%PDF"),
    }
    for format_name, (media_type, signature) in expected_files.items():
        export = client.post(
            f"/api/v1/projects/{project_id}/exports",
            json={"chart": _chart(format_name)},
        )
        assert export.status_code == 200
        assert export.json()["status"] == "ready"
        download = client.get(export.json()["downloadUrl"])
        assert download.status_code == 200
        assert download.headers["content-type"].startswith(media_type)
        assert download.content.startswith(signature)


def test_email_sign_in_history_and_shared_download(
    api_client: tuple[TestClient, MemoryEmailSender],
) -> None:
    client, sender = api_client
    session = _sample_project(client)
    project_id = session["projectId"]
    export = client.post(f"/api/v1/projects/{project_id}/exports", json={"chart": _chart()})
    assert export.status_code == 200

    requested = client.post("/api/v1/auth/email-code", json={"email": "Researcher@Example.com"})
    assert requested.status_code == 200
    assert "code" not in requested.json()
    assert sender.messages[-1][0] == "researcher@example.com"

    verified = client.post(
        "/api/v1/auth/email-code/verify",
        json={
            "challengeId": requested.json()["challengeId"],
            "code": sender.messages[-1][1],
        },
    )
    assert verified.status_code == 200
    assert verified.json()["authenticated"] is True
    assert client.cookies.get("labviz_session")

    shared = client.post(
        f"/api/v1/projects/{project_id}/shares",
        json={"downloadsEnabled": True},
    )
    assert shared.status_code == 200
    token = shared.json()["token"]

    history = client.get("/api/v1/projects")
    assert history.status_code == 200
    assert history.json()["projects"][0]["id"] == project_id

    public_chart = client.get(f"/api/v1/shares/{token}")
    assert public_chart.status_code == 200
    assert public_chart.json()["downloads"]["png"] is not None


def test_real_csv_and_xlsx_uploads_and_stable_errors(
    api_client: tuple[TestClient, MemoryEmailSender],
) -> None:
    client, _sender = api_client
    csv_payload = b"elapsed_s,signal_v\n0,1.2\n1,1.5\n2,\n"
    response = client.post(
        "/api/v1/projects",
        files={"file": ("experiment.csv", csv_payload, "text/csv")},
    )
    assert response.status_code == 202
    project_id = response.json()["projectId"]
    preview = client.get(f"/api/v1/projects/{project_id}/preview")
    assert preview.status_code == 200
    assert preview.json()["totalRows"] == 3

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Measurements"
    worksheet.append(["time_s", "temperature_c"])
    worksheet.append([0, 21.2])
    worksheet.append([1, 21.8])
    summary = workbook.create_sheet("Summary")
    summary.append(["instrument export"])
    summary.append(["elapsed_s", "mean_c"])
    summary.append([0, 21.5])
    summary.append([1, 22.0])
    xlsx_payload = io.BytesIO()
    workbook.save(xlsx_payload)
    xlsx = client.post(
        "/api/v1/projects",
        files={
            "file": (
                "measurements.xlsx",
                xlsx_payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"sheetName": "Summary", "headerRow": "2"},
    )
    assert xlsx.status_code == 202
    xlsx_project = client.get(f"/api/v1/projects/{xlsx.json()['projectId']}")
    assert xlsx_project.status_code == 200
    assert xlsx_project.json()["source"]["sheetName"] == "Summary"
    assert xlsx_project.json()["source"]["availableSheets"] == ["Measurements", "Summary"]
    assert xlsx_project.json()["source"]["headerRow"] == 2
    xlsx_preview = client.get(f"/api/v1/projects/{xlsx.json()['projectId']}/preview").json()
    assert [column["field"] for column in xlsx_preview["columns"]] == [
        "elapsed_s",
        "mean_c",
    ]

    unsupported = client.post(
        "/api/v1/projects",
        files={"file": ("notes.docx", b"not a table", "application/octet-stream")},
    )
    assert unsupported.status_code == 422
    assert unsupported.json() == {
        "code": "unsupported-format",
        "message": "Choose an XLSX, CSV, TSV, TXT, or JSON file.",
    }

    anonymous = TestClient(client.app)
    denied = anonymous.post(
        f"/api/v1/projects/{project_id}/shares",
        json={"downloadsEnabled": False},
    )
    assert denied.status_code == 401
    assert denied.json()["code"] == "authentication-required"


def test_temporary_projects_are_scoped_to_the_creating_browser(
    api_client: tuple[TestClient, MemoryEmailSender],
) -> None:
    client, _sender = api_client
    session = _sample_project(client)
    project_id = session["projectId"]
    job_id = session["job"]["id"]

    with TestClient(client.app) as another_browser:
        for path in (
            f"/api/v1/projects/{project_id}",
            f"/api/v1/jobs/{job_id}",
            f"/api/v1/projects/{project_id}/preview",
            f"/api/v1/projects/{project_id}/quality",
        ):
            denied = another_browser.get(path)
            assert denied.status_code == 403
            assert denied.json()["code"] == "project-access-denied"

    assert client.get(f"/api/v1/projects/{project_id}/preview").status_code == 200


def test_production_configuration_rejects_console_codes_and_insecure_cookies(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="SMTP"):
        Settings(
            database_path=tmp_path / "production.db",
            allowed_origins=("https://labviz.example",),
            public_web_url="https://labviz.example",
            environment="production",
            auth_mode="console",
            cookie_secure=True,
        )
    with pytest.raises(ValueError, match="secure cookies"):
        Settings(
            database_path=tmp_path / "production.db",
            allowed_origins=("https://labviz.example",),
            public_web_url="https://labviz.example",
            environment="production",
            auth_mode="smtp",
            cookie_secure=False,
        )


def test_wide_tables_return_bounded_preview_and_representative_findings() -> None:
    row_count = 2_260
    column_count = 224
    values = np.arange(row_count, dtype=float)
    frame = pd.DataFrame({f"signal_{index}": values.copy() for index in range(column_count)})
    frame.loc[:499, :] = np.nan

    preview = build_preview("wide-project", frame)
    quality = build_quality_report("wide-project", frame)
    assert len(preview["rows"]) <= 20_000 // column_count
    assert all(len(finding["rowIds"]) <= 100 for finding in quality["findings"])
    missing = next(item for item in quality["findings"] if item["kind"] == "missing")
    assert missing["affectedCount"] == 500
    assert missing["rowIdsTruncated"] is True
    assert len(json.dumps(preview)) < 400_000
    assert len(json.dumps(quality)) < 300_000


def test_valid_ranges_and_trend_flags_remain_user_controlled(
    api_client: tuple[TestClient, MemoryEmailSender],
) -> None:
    client, _sender = api_client
    csv_rows = ["step,response,label"]
    csv_rows.extend(
        f"{index},{70 if index == 12 else 2 * index + 1},sample-{index}" for index in range(30)
    )
    created = client.post(
        "/api/v1/projects",
        files={"file": ("range.csv", "\n".join(csv_rows).encode(), "text/csv")},
    )
    project_id = created.json()["projectId"]

    ruled = client.put(
        f"/api/v1/projects/{project_id}/quality-rules",
        json={"ranges": [{"field": "response", "minimum": 0, "maximum": 60}]},
    )
    assert ruled.status_code == 200
    findings = ruled.json()["findings"]
    outside = next(item for item in findings if item["kind"] == "outside-range")
    assert outside["affectedCount"] == 1
    assert outside["rowIds"] == [13]
    assert any(item["kind"] == "trend-inconsistent" for item in findings)

    removed = client.patch(
        f"/api/v1/projects/{project_id}/cleaning-decisions",
        json={"decisions": [{"findingId": outside["id"], "action": "remove"}]},
    )
    assert removed.status_code == 200
    cleaned = client.get(f"/api/v1/projects/{project_id}/exports/cleaned-data.csv").text
    assert ",70," not in cleaned

    invalid = client.put(
        f"/api/v1/projects/{project_id}/quality-rules",
        json={"ranges": [{"field": "label", "minimum": 0}]},
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "invalid-quality-rule"


def test_scientific_analysis_and_all_chart_renderers() -> None:
    rows = []
    for x_value in range(8):
        for replicate in range(4):
            y_value = 2 * x_value + 1 + (replicate - 1.5) * 0.1
            rows.append(
                {
                    "x": float(x_value),
                    "y": y_value,
                    "z": float(x_value**2 + replicate),
                    "error": 0.2,
                }
            )
    frame = pd.DataFrame(rows)
    advanced = _chart()
    advanced.update(
        {
            "xAxis": {"field": "x", "title": "X", "unit": "s"},
            "yAxis": {"field": "y", "title": "Y", "unit": "V"},
            "series": [{"field": "y", "label": "Y", "color": "#2563EB"}],
            "fitting": {
                "model": "linear",
                "polynomialOrder": 1,
                "showEquation": True,
                "showRSquared": True,
                "confidenceBand": True,
                "confidenceLevel": 95,
            },
            "uncertainty": {
                "mode": "confidence-interval",
                "errorField": None,
                "confidenceLevel": 95,
            },
            "export": {
                **advanced["export"],
                "sizePreset": "custom",
                "width": 180,
                "height": 100,
                "unit": "mm",
                "fontFamily": "Arial",
                "fontSize": 9,
                "lineWidth": 1.25,
                "markerSize": 3,
                "legendPosition": "top",
                "transparentBackground": False,
            },
        }
    )
    spec = ChartSpec.model_validate(advanced)
    analysis = analyze_chart(frame, spec)
    fit = analysis["series"][0]["fit"]
    assert fit["rSquared"] > 0.999
    assert fit["equation"].startswith("y =")
    assert all(point["lower"] < point["upper"] for point in fit["points"])
    uncertainty = analysis["series"][0]["uncertainty"]
    assert len(uncertainty["points"]) == 8
    assert render_chart(frame, spec).startswith(b"\x89PNG")

    base = {
        **advanced,
        "fitting": {**advanced["fitting"], "model": "none", "confidenceBand": False},
        "uncertainty": {**advanced["uncertainty"], "mode": "none"},
        "export": {**advanced["export"], "sizePreset": "double-column"},
    }
    for chart_type in ("scatter", "bar", "histogram", "box", "heatmap"):
        payload = {**base, "type": chart_type}
        if chart_type == "heatmap":
            payload["series"] = [
                {"field": "y", "label": "Y", "color": "#2563EB"},
                {"field": "z", "label": "Z", "color": "#DC6B2F"},
            ]
        rendered = render_chart(frame, ChartSpec.model_validate(payload))
        assert rendered.startswith(b"\x89PNG")

    surface_rows = [
        {"x": float(x), "y": float(y), "z": float(x**2 + y**2)} for x in range(5) for y in range(5)
    ]
    surface = {
        **base,
        "type": "surface3d",
        "series": [
            {"field": "y", "label": "Y", "color": "#2563EB"},
            {"field": "z", "label": "Z", "color": "#DC6B2F"},
        ],
    }
    assert render_chart(pd.DataFrame(surface_rows), ChartSpec.model_validate(surface)).startswith(
        b"\x89PNG"
    )


def test_long_format_grouping_creates_independent_series_and_fits() -> None:
    frame = pd.DataFrame(
        [
            {"time": float(time), "response": slope * time + 1, "sample": sample}
            for sample, slope in (("Control", 1.0), ("Treatment", 2.0))
            for time in range(6)
        ]
    )
    payload = _chart()
    payload.update(
        {
            "xAxis": {"field": "time", "title": "Time", "unit": "min"},
            "yAxis": {"field": "response", "title": "Response", "unit": "mV"},
            "series": [{"field": "response", "label": "Response", "color": "#2563EB"}],
            "groupField": "sample",
            "fitting": {
                "model": "linear",
                "polynomialOrder": 1,
                "showEquation": True,
                "showRSquared": True,
                "confidenceBand": False,
                "confidenceLevel": 95,
            },
        }
    )
    spec = ChartSpec.model_validate(payload)
    analysis = analyze_chart(frame, spec)
    assert [series["group"] for series in analysis["series"]] == ["Control", "Treatment"]
    assert [series["label"] for series in analysis["series"]] == [
        "Response · Control",
        "Response · Treatment",
    ]
    assert all(series["fit"]["rSquared"] == pytest.approx(1.0) for series in analysis["series"])
    assert render_chart(frame, spec).startswith(b"\x89PNG")

    too_many = pd.DataFrame(
        {"time": range(13), "response": range(13), "sample": [f"S{i}" for i in range(13)]}
    )
    with pytest.raises(ProcessingError, match="more than 12 groups"):
        analyze_chart(too_many, spec)


def test_saved_project_security_restore_sharing_and_cleaned_export(
    api_client: tuple[TestClient, MemoryEmailSender],
) -> None:
    client, sender = api_client
    csv_payload = b"x,y,secret_note\n0,1,private-a\n1,,private-b\n2,5,private-c\n"
    created = client.post(
        "/api/v1/projects",
        files={"file": ("sensitive.csv", csv_payload, "text/csv")},
    )
    project_id = created.json()["projectId"]
    quality = client.get(f"/api/v1/projects/{project_id}/quality").json()
    finding_id = next(item["id"] for item in quality["findings"] if item["kind"] == "missing")

    _sign_in(client, sender)
    saved = client.post(f"/api/v1/projects/{project_id}/save")
    assert saved.status_code == 200
    assert saved.json()["storageMode"] == "saved-cloud"
    me = client.get("/api/v1/auth/me")
    assert me.json()["authenticated"] is True

    decision = client.patch(
        f"/api/v1/projects/{project_id}/cleaning-decisions",
        json={"decisions": [{"findingId": finding_id, "action": "exclude"}]},
    )
    assert decision.status_code == 200
    excluded_csv = client.get(f"/api/v1/projects/{project_id}/exports/cleaned-data.csv").text
    assert "private-b" in excluded_csv
    removed = client.patch(
        f"/api/v1/projects/{project_id}/cleaning-decisions",
        json={"decisions": [{"findingId": finding_id, "action": "remove"}]},
    )
    assert removed.status_code == 200
    cleaned_csv = client.get(f"/api/v1/projects/{project_id}/exports/cleaned-data.csv").text
    assert "private-b" not in cleaned_csv

    workspace = client.get(f"/api/v1/projects/{project_id}/workspace")
    assert workspace.status_code == 200
    assert workspace.json()["decisions"][0]["action"] == "remove"

    shared = client.post(f"/api/v1/projects/{project_id}/shares", json={"downloadsEnabled": False})
    token = shared.json()["token"]
    public = client.get(f"/api/v1/shares/{token}")
    assert public.status_code == 200
    assert "secret_note" not in {column["field"] for column in public.json()["preview"]["columns"]}
    assert public.json()["preview"]["totalRows"] == 2
    assert "private-b" not in public.text
    assert public.json()["analysis"]["series"][0]["points"]

    updated = client.patch(
        f"/api/v1/projects/{project_id}/shares/{token}",
        json={"downloadsEnabled": True},
    )
    assert updated.json()["downloadsEnabled"] is True
    duplicate = client.post(f"/api/v1/projects/{project_id}/duplicate")
    assert duplicate.status_code == 200
    assert duplicate.json()["projectId"] != project_id
    assert duplicate.json()["storageMode"] == "saved-cloud"

    anonymous = TestClient(client.app)
    denied = anonymous.get(f"/api/v1/projects/{project_id}/workspace")
    assert denied.status_code == 401
    assert denied.json()["code"] == "authentication-required"
    revoked = client.delete(f"/api/v1/projects/{project_id}/shares/{token}")
    assert revoked.status_code == 204
    assert client.get(f"/api/v1/shares/{token}").status_code == 404

    logged_out = client.post("/api/v1/auth/logout")
    assert logged_out.json() == {"apiVersion": "v1", "authenticated": False, "user": None}
    assert client.get("/api/v1/auth/me").json()["authenticated"] is False


def test_data_or_chart_changes_invalidate_prepared_exports_and_shared_downloads(
    api_client: tuple[TestClient, MemoryEmailSender],
) -> None:
    client, sender = api_client
    session = _sample_project(client)
    project_id = session["projectId"]
    _sign_in(client, sender)
    assert client.post(f"/api/v1/projects/{project_id}/save").status_code == 200

    first_export = client.post(
        f"/api/v1/projects/{project_id}/exports",
        json={"chart": _chart()},
    )
    assert first_export.status_code == 200
    first_download = first_export.json()["downloadUrl"]
    assert client.get(first_download).status_code == 200
    share = client.post(
        f"/api/v1/projects/{project_id}/shares",
        json={"downloadsEnabled": True},
    ).json()
    assert client.get(f"/api/v1/shares/{share['token']}").json()["downloads"]["png"]

    quality = client.get(f"/api/v1/projects/{project_id}/quality").json()
    finding_id = quality["findings"][0]["id"]
    changed_data = client.patch(
        f"/api/v1/projects/{project_id}/cleaning-decisions",
        json={"decisions": [{"findingId": finding_id, "action": "exclude"}]},
    )
    assert changed_data.status_code == 200
    assert client.get(first_download).status_code == 404
    public_after_cleaning = client.get(f"/api/v1/shares/{share['token']}").json()
    assert public_after_cleaning["downloads"]["png"] is None

    second_export = client.post(
        f"/api/v1/projects/{project_id}/exports",
        json={"chart": _chart()},
    )
    second_download = second_export.json()["downloadUrl"]
    changed_chart = _chart()
    changed_chart["title"] = "Updated thermal response"
    assert (
        client.put(
            f"/api/v1/projects/{project_id}/chart",
            json={"chart": changed_chart},
        ).status_code
        == 200
    )
    assert client.get(second_download).status_code == 404


def test_auth_challenges_and_sessions_survive_service_restart(
    api_client: tuple[TestClient, MemoryEmailSender],
) -> None:
    client, sender = api_client
    repository = cast(ProjectRepository, cast(FastAPI, client.app).state.repository)
    first = AuthService(sender, 3_600, repository)
    challenge_id, _expires, _resend = first.request_code("persistent@example.com")
    email, code = sender.messages[-1]
    stored = repository.get_auth_challenge(challenge_id)
    assert stored is not None
    cheap_digest = hashlib.sha256(f"{stored['salt']}:{code}".encode()).hexdigest()
    assert stored["code_digest"] != cheap_digest

    restarted = AuthService(sender, 3_600, repository)
    user, token = restarted.verify_code(challenge_id, code)
    assert user["email"] == email
    second_restart = AuthService(sender, 3_600, repository)
    assert second_restart.get_user(token) == user
    second_restart.logout(token)
    assert first.get_user(token) is None
