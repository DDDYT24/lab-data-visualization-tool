from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from labviz_api.auth import AuthService, MemoryEmailSender
from labviz_api.config import Settings
from labviz_api.main import create_app
from labviz_api.models import ChartSpec
from labviz_api.processing import ProcessingError, analyze_chart, render_chart
from labviz_api.repository import ProjectRepository

ANSWER_KEY_PATH = (
    Path(__file__).resolve().parents[1] / "samples" / "v22" / "statistics_answer_keys.json"
)


@pytest.fixture
def api_client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        database_path=tmp_path / "labviz-v22-statistics.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
    )
    repository = ProjectRepository(settings.database_path, settings.project_ttl_seconds)
    auth = AuthService(MemoryEmailSender(), settings.session_ttl_seconds, repository)
    with TestClient(create_app(settings, repository, auth)) as client:
        yield client


def _chart(case: dict[str, Any], export_format: str = "png") -> ChartSpec:
    fitting: dict[str, Any] = {
        "model": case["model"],
        "polynomialOrder": case.get("polynomialOrder", 2),
        "fitMethod": case["fitMethod"],
        "confidenceBand": case["confidenceBand"],
        "confidenceMethod": case["confidenceMethod"],
        "confidenceLevel": 95,
        "intervalKind": case["intervalKind"],
    }
    return ChartSpec.model_validate(
        {
            "schemaVersion": 1,
            "type": "scatter",
            "title": f"Synthetic {case['id']}",
            "xAxis": {"field": "x", "title": "X", "unit": ""},
            "yAxis": {"field": "y", "title": "Y", "unit": ""},
            "series": [{"field": "y", "label": "Y", "color": "#2563EB"}],
            "panelCount": 1,
            "fitting": fitting,
            "uncertainty": {"mode": "none"},
            "export": {
                "format": export_format,
                "dpi": 300,
                "sizePreset": "single-column",
                "grayscalePreview": False,
            },
        }
    )


def _disclosure(series: dict[str, Any], method: str) -> dict[str, Any]:
    return next(item for item in series["disclosures"] if item["method"] == method)


def test_independent_statistics_answer_key_covers_each_shipped_method() -> None:
    payload = json.loads(ANSWER_KEY_PATH.read_text(encoding="utf-8"))
    assert payload["keyVersion"] == "v1"
    assert payload["syntheticOnly"] is True
    assert {case["id"] for case in payload["cases"]} == {
        "pointwise-linear",
        "prediction-linear",
        "simultaneous-polynomial",
        "robust-huber",
    }

    for case in payload["cases"]:
        frame = pd.DataFrame(case["rows"])
        chart = _chart(case)
        first = analyze_chart(frame, chart)
        second = analyze_chart(frame, chart)
        series = first["series"][0]
        fit = series["fit"]
        expected = case["expected"]

        assert fit is not None
        assert fit == second["series"][0]["fit"]
        assert fit["sampleSize"] == expected["sampleSize"]
        assert fit["excludedCount"] == expected["excludedCount"]
        assert fit["fitMethod"] == case["fitMethod"]
        assert fit["intervalKind"] == expected.get("intervalKind", case["intervalKind"])
        if "confidenceMethod" in expected:
            assert fit["confidenceMethod"] == expected["confidenceMethod"]
        if "rSquaredMin" in expected:
            assert fit["rSquared"] >= expected["rSquaredMin"]
        if "slopeRange" in expected:
            assert expected["slopeRange"][0] <= fit["coefficients"][0] <= expected["slopeRange"][1]
        if expected.get("intervalStrict"):
            assert all(point["lower"] < point["upper"] for point in fit["points"])
        if case["fitMethod"] == "robust-huber":
            assert fit["robustConverged"] is expected["robustConverged"]
            assert fit["robustIterations"] <= expected["robustIterationsMax"]
        for method, status in (
            ("confidence-band", expected.get("confidenceDisclosure")),
            ("prediction-band", expected.get("predictionDisclosure")),
            ("simultaneous-band", expected.get("simultaneousDisclosure")),
            ("robust-fitting", expected.get("robustDisclosure")),
        ):
            if status:
                assert _disclosure(series, method)["status"] == status

        for export_format, signature in (
            ("png", b"\x89PNG\r\n\x1a\n"),
            ("svg", b"<?xml"),
            ("pdf", b"%PDF"),
        ):
            exported = render_chart(frame, _chart(case, export_format))
            assert exported.startswith(signature)
            assert (
                f"n={expected['sampleSize']}"
                in (exported.decode("utf-8") if export_format == "svg" else "")
                or export_format != "svg"
            )


def test_prediction_interval_is_wider_than_the_pointwise_mean_interval() -> None:
    payload = json.loads(ANSWER_KEY_PATH.read_text(encoding="utf-8"))
    case = next(item for item in payload["cases"] if item["id"] == "prediction-linear")
    frame = pd.DataFrame(case["rows"])
    prediction = analyze_chart(frame, _chart(case))["series"][0]["fit"]
    pointwise_case = {**case, "id": "pointwise-comparison", "intervalKind": "pointwise-mean"}
    pointwise = analyze_chart(frame, _chart(pointwise_case))["series"][0]["fit"]

    assert prediction is not None and pointwise is not None
    prediction_width = (
        prediction["points"][len(prediction["points"]) // 2]["upper"]
        - prediction["points"][len(prediction["points"]) // 2]["lower"]
    )
    pointwise_width = (
        pointwise["points"][len(pointwise["points"]) // 2]["upper"]
        - pointwise["points"][len(pointwise["points"]) // 2]["lower"]
    )
    assert prediction_width > pointwise_width
    assert (
        _disclosure(analyze_chart(frame, _chart(case))["series"][0], "prediction-band")["status"]
        == "supported"
    )


def test_answer_keyed_diagnostics_cover_heteroscedastic_and_misleading_data() -> None:
    payload = json.loads(ANSWER_KEY_PATH.read_text(encoding="utf-8"))
    diagnostics = {case["id"]: case for case in payload["diagnosticCases"]}

    hetero = diagnostics["heteroscedastic-linear"]
    hetero_series = analyze_chart(pd.DataFrame(hetero["rows"]), _chart(hetero))["series"][0]
    assert hetero_series["fit"]["sampleSize"] == hetero["expected"]["sampleSize"]
    assert all(point["lower"] < point["upper"] for point in hetero_series["fit"]["points"])
    assert _disclosure(hetero_series, "prediction-band")["status"] == "supported"

    misleading = diagnostics["misleading-linear"]
    misleading_series = analyze_chart(pd.DataFrame(misleading["rows"]), _chart(misleading))[
        "series"
    ][0]
    assert (
        misleading_series["residualDiagnostic"]["residualTrend"]
        == misleading["expected"]["residualTrend"]
    )

    insufficient = diagnostics["insufficient-polynomial"]
    insufficient_series = analyze_chart(pd.DataFrame(insufficient["rows"]), _chart(insufficient))[
        "series"
    ][0]
    assert insufficient_series["fit"] is None
    assert _disclosure(insufficient_series, "fit")["status"] == "insufficient-data"


def test_advanced_statistics_failures_are_explicit_and_machine_readable() -> None:
    payload = json.loads(ANSWER_KEY_PATH.read_text(encoding="utf-8"))
    assert {item["code"] for item in payload["failureCases"]} == {
        "unsupported-robust-model",
        "unsupported-interval-model",
        "unsupported-interval-method",
        "invalid-interval",
        "robust-interval-unsupported",
        "fit-not-suitable",
    }
    base = next(item for item in payload["cases"] if item["id"] == "pointwise-linear")
    frame = pd.DataFrame(base["rows"])

    def expect_code(**fitting: Any) -> None:
        case = {**base, **fitting}
        with pytest.raises(ProcessingError) as error:
            from labviz_api.processing import validate_chart_fields

            validate_chart_fields(frame, _chart(case))
        assert error.value.code == fitting["expectedCode"]

    expect_code(
        model="polynomial",
        fitMethod="robust-huber",
        confidenceBand=False,
        intervalKind="pointwise-mean",
        expectedCode="unsupported-robust-model",
    )
    expect_code(
        model="exponential",
        fitMethod="ordinary-least-squares",
        confidenceBand=True,
        intervalKind="prediction",
        expectedCode="unsupported-interval-model",
    )
    expect_code(
        model="linear",
        fitMethod="ordinary-least-squares",
        confidenceBand=True,
        confidenceMethod="bootstrap",
        intervalKind="prediction",
        expectedCode="unsupported-interval-method",
    )
    expect_code(
        model="linear",
        fitMethod="ordinary-least-squares",
        confidenceBand=False,
        intervalKind="prediction",
        expectedCode="invalid-interval",
    )
    expect_code(
        model="linear",
        fitMethod="robust-huber",
        confidenceBand=True,
        intervalKind="pointwise-mean",
        expectedCode="robust-interval-unsupported",
    )

    singular = {**base, "model": "polynomial", "polynomialOrder": 3}
    singular_frame = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0], "y": [1.0, 2.0, 3.0, 4.0]})
    singular_series = analyze_chart(singular_frame, _chart(singular))["series"][0]
    assert singular_series["fit"] is None
    assert any(
        "needs more distinct complete points" in item for item in singular_series["warnings"]
    )


def test_api_statistics_contract_and_export_disclose_the_same_prediction_method(
    api_client: TestClient,
) -> None:
    created = api_client.post("/api/v1/samples/scatter-fit/projects")
    assert created.status_code == 202, created.text
    project_id = created.json()["projectId"]
    job = api_client.get(f"/api/v1/jobs/{created.json()['job']['id']}")
    assert job.json()["stage"] == "ready", job.text
    chart_payload = {
        "schemaVersion": 1,
        "type": "scatter",
        "title": "Synthetic prediction API check",
        "xAxis": {"field": "dose_uM", "title": "Dose", "unit": ""},
        "yAxis": {"field": "response_mV", "title": "Response", "unit": ""},
        "series": [{"field": "response_mV", "label": "Response", "color": "#2563EB"}],
        "panelCount": 1,
        "fitting": {
            "model": "linear",
            "fitMethod": "ordinary-least-squares",
            "confidenceBand": True,
            "confidenceMethod": "student-t",
            "confidenceLevel": 95,
            "intervalKind": "prediction",
        },
        "uncertainty": {"mode": "none"},
        "export": {
            "format": "svg",
            "dpi": 300,
            "sizePreset": "single-column",
            "grayscalePreview": False,
        },
    }
    analysis = api_client.post(
        f"/api/v1/projects/{project_id}/chart-analysis", json={"chart": chart_payload}
    )
    assert analysis.status_code == 200, analysis.text
    fit = analysis.json()["series"][0]["fit"]
    assert fit["intervalKind"] == "prediction"
    assert fit["confidenceMethod"] == "student-t"
    assert any(
        item["method"] == "prediction-band" and item["status"] == "supported"
        for item in analysis.json()["series"][0]["disclosures"]
    )

    exported = api_client.post(
        f"/api/v1/projects/{project_id}/exports", json={"chart": chart_payload}
    )
    assert exported.status_code == 200, exported.text
    download = api_client.get(exported.json()["downloadUrl"])
    assert download.status_code == 200
    body = download.content.decode("utf-8")
    assert "prediction interval" in body
    assert "Deferred:" in body
    assert "multiplicity correction" in body
