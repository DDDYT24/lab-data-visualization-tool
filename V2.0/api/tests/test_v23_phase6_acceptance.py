"""Source integration acceptance with independently calculated unequal-repeat answers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_research_api import settings, unlock
from test_research_chart_api import applied
from test_research_figure_composition import legacy
from test_research_figure_jobs import wait_state

from labviz_api.main import create_app

# Sample means are 1, 5, 10 and 13 mV: mean=7.25 mV, unlike row mean=6.625.
# Group means are 3 and 11.5 mV. The zero is an observed measurement.
SOURCE = (
    b"s,repeat,v,unit,g\n"
    b"a,1,0,mV,A\n"
    b"a,2,2,mV,A\n"
    b"b,1,3,mV,A\n"
    b"b,2,5,mV,A\n"
    b"b,3,7,mV,A\n"
    b"c,1,10,mV,B\n"
    b"d,1,12,mV,B\n"
    b"d,2,14,mV,B\n"
)

PLAN = {
    "mapping": {
        "sample": "s",
        "experimentalUnit": "s",
        "technicalRepeat": "repeat",
        "value": "v",
        "unit": "unit",
        "group": "g",
    },
    "analysisUnit": "experimental_unit",
    "steps": [{"kind": "convert_unit", "valueField": "v", "unitField": "unit", "toUnit": "V"}],
}


def test_unequal_repeats_conversion_mixed_figure_frozen_output_and_reopen(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier, _, old_source = legacy(client, "line")
        base = f"/api/v1/projects/{identifier}/research"
        revision = applied(client, identifier, SOURCE, PLAN)
        prepared = client.get(base).json()["data"]
        assert prepared["summary"]["independentUnits"] == 4
        assert prepared["summary"]["observations"] == 8
        assert prepared["summary"]["analysisN"] == 4
        assert prepared["summary"]["mean"] == pytest.approx(0.00725)
        saved = []
        results = []
        for kind in ("grouped_strip", "violin"):
            response = client.post(
                base + "/charts",
                json={
                    "revisionId": revision,
                    "spec": {"kind": kind, "summary": "mean"},
                },
            )
            assert response.status_code == 200, response.text
            view = response.json()
            groups = {g["label"]: g for g in view["groups"]}
            assert groups["A"]["summary"] == pytest.approx(0.003)
            assert groups["B"]["summary"] == pytest.approx(0.0115)
            assert [groups[g]["summaryN"] for g in ("A", "B")] == [2, 2]
            assert [groups[g]["observations"] for g in ("A", "B")] == [5, 3]
            result = client.get(base + f"/charts/{view['id']}/artifact/json").json()["result"]
            assert len(result["points"]) == 8
            assert sorted(p["y"] for p in result["points"]) == pytest.approx(
                [0, 0.002, 0.003, 0.005, 0.007, 0.010, 0.012, 0.014]
            )
            assert result["resultSha256"] == view["resultSha256"]
            saved.append(view["id"])
            results.append(result)
        response = client.post(
            base + "/figure-boards?deferArtifacts=true",
            json={
                "title": "Unequal technical repeats",
                "panelChartIds": [saved[0], saved[1], old_source, saved[0]],
            },
        )
        assert response.status_code == 202, response.text
        board_url = base + "/figure-boards/" + response.json()["id"]
        ready = wait_state(client, board_url)
        assert [p["kind"] for p in ready["panels"]] == [
            "grouped_strip",
            "violin",
            "line",
            "grouped_strip",
        ]
        artifacts = {
            k: client.get(board_url + "/artifact/" + k).content for k in ("png", "svg", "pdf")
        }
        point = results[0]["points"][0]["observation"]
        interaction = client.get(board_url + "/interaction", params={"row": point["id"]})
        assert interaction.status_code == 200, interaction.text
        assert {r["panel"] for r in interaction.json()["rows"]} == {"A", "B", "D"}
        assert client.get(base).json()["data"]["summary"] == prepared["summary"]
        for kind, payload in artifacts.items():
            assert client.get(board_url + "/artifact/" + kind).content == payload
        preview = client.post(base + "/outputs/preview", json={"boardId": ready["id"]})
        assert preview.status_code == 200, preview.text
        check = preview.json()
        assert [p["sourceCsvRows"] for p in check["manifest"]["panels"]] == [8, 8, 4, 8]
        response = client.post(
            base + "/outputs",
            json={
                "token": check["token"],
                "notes": check["notes"],
                "reviewedIncludedData": True,
            },
        )
        assert response.status_code == 200, response.text
        output = response.json()
        download_url = base + f"/outputs/{output['id']}/download"
        frozen = client.get(download_url).content
        assert hashlib.sha256(frozen).hexdigest() == output["sha256"]
        with zipfile.ZipFile(io.BytesIO(frozen)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            for item in manifest["files"]:
                content = archive.read(item["name"])
                assert hashlib.sha256(content).hexdigest() == item["sha256"]
            for label, result in zip(
                ("A", "B", "D"), (results[0], results[1], results[0]), strict=True
            ):
                rows = list(
                    csv.DictReader(
                        io.StringIO(archive.read(f"panels/{label}-source.csv").decode("utf-8-sig"))
                    )
                )
                expected = {p["observation"]["id"]: p["observation"] for p in result["points"]}
                assert len(rows) == 8 and {r["id"] for r in rows} == set(expected)
                for row in rows:
                    observation = expected[row["id"]]
                    assert float(row["derivedValue"]) == observation["value"]
                    assert float(row["reportedText"]) / 1000 == pytest.approx(
                        float(row["derivedValue"])
                    )
                    assert (row["unit"], row["reportedUnit"]) == ("V", "mV")
                    assert json.loads(row["sourceReferences"]) == observation["lineage"]
                assert len({r["experimentalUnit"] for r in rows}) == 4
            for kind, payload in artifacts.items():
                assert archive.read("figure." + kind) == payload
        applied(client, identifier, SOURCE.replace(b",14,", b",140,"), PLAN)
        assert client.get(download_url).content == frozen
    with TestClient(create_app(config)) as reopened:
        unlock(reopened)
        assert reopened.get(download_url).content == frozen
        assert reopened.get(board_url).json()["spec"] == ready["spec"]
        for kind, payload in artifacts.items():
            assert reopened.get(board_url + "/artifact/" + kind).content == payload
