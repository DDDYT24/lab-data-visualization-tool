"""Actual stopped-source backup CLI and isolated-copy recovery of research records."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from test_research_api import project
from test_research_chart_api import applied
from test_research_figure_jobs import wait_state
from test_v23_recovery_rehearsal import app_at, hashes, unlock
from test_v23_source_data import make_data

from labviz_api.research_store import ResearchStore

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows source-data CLI")


def invoke_cli(action: str, **paths: Path) -> subprocess.CompletedProcess[str]:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-File",
        str(ROOT / "source-data-v23.ps1"),
        "-Action",
        action,
    ]
    for name, value in paths.items():
        command.extend(["-" + name, str(value)])
    return subprocess.run(
        command,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def cli(action: str, **paths: Path) -> dict[str, Any]:
    result = invoke_cli(action, **paths)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["verified"] is True
    return dict(data)


@pytest.mark.parametrize("interrupted", [False, True], ids=["complete", "interrupted-job"])
def test_cli_restores_board_template_output_and_interrupted_job(interrupted: bool) -> None:
    # Never select the user's default data. Every CLI directory is private and unique.
    original = ROOT / "V2.0/api/.labviz/v23-restored" / ("phase6-fixture-" + uuid4().hex)
    with TestClient(app_at(original)) as client:
        unlock(client)
        identifier = project(client)
        revision = applied(client, identifier)
        base = f"/api/v1/projects/{identifier}/research"
        chart_ids = []
        for kind in ("grouped_strip", "violin"):
            response = client.post(
                base + "/charts", json={"revisionId": revision, "spec": {"kind": kind}}
            )
            assert response.status_code == 200, response.text
            chart_ids.append(response.json()["id"])
        response = client.post(
            base + "/figure-boards?deferArtifacts=true",
            json={"panelChartIds": chart_ids, "title": "Recovery fixture"},
        )
        assert response.status_code == 202, response.text
        board_url = base + "/figure-boards/" + response.json()["id"]
        board = wait_state(client, board_url)
        frozen = {
            kind: client.get(board_url + "/artifact/" + kind).content
            for kind in ("png", "svg", "pdf")
        }
        template_response = client.post(
            base + "/templates", json={"boardId": board["id"], "name": "Recovered recipe"}
        )
        assert template_response.status_code == 200, template_response.text
        template = template_response.json()
        preview = client.post(base + "/outputs/preview", json={"boardId": board["id"]})
        assert preview.status_code == 200, preview.text
        check = preview.json()
        response = client.post(
            base + "/outputs",
            json={"token": check["token"], "notes": check["notes"], "reviewedIncludedData": True},
        )
        assert response.status_code == 200, response.text
        output = response.json()
        output_url = base + f"/outputs/{output['id']}/download"
        archive = client.get(output_url).content
        assert hashlib.sha256(archive).hexdigest() == output["sha256"]
        expected_research = client.get(base).json()
    if interrupted:
        pending_id = uuid4().hex
        # Persisted abandoned-state fixture, not a claim that a live process was killed.
        with ResearchStore(original / "labviz.db").connect() as connection:
            connection.execute(
                "INSERT INTO research_figure_boards VALUES (?,?,?,?)",
                (pending_id, identifier, json.dumps(board["spec"]), board["createdAt"]),
            )
            connection.execute(
                "INSERT INTO research_figure_jobs (board_id,state,panels_json,attempt) "
                "VALUES (?,'running',?,'abandoned-phase6-fixture')",
                (pending_id, json.dumps(board["panels"])),
            )
    (original / "objects").mkdir(exist_ok=True)
    (original / "objects/recovery-note.txt").write_bytes(b"synthetic future object payload")
    before = hashes(original)
    backup = Path(cli("Backup", DataDirectory=original)["backupDirectory"])
    cli("Verify", BackupDirectory=backup)
    restored = Path(cli("Restore", BackupDirectory=backup)["restoredDirectory"])
    assert restored != original and hashes(original) == before
    assert hashes(restored / "objects") == hashes(original / "objects")
    assert (restored / "restore-manifest.json").is_file()
    assert (
        hashlib.sha256((restored / "labviz.db").read_bytes()).hexdigest()
        == hashlib.sha256((backup / "labviz.db").read_bytes()).hexdigest()
    )
    with TestClient(app_at(restored)) as client:
        assert client.get(base).status_code == 403
        unlock(client)
        assert client.get(base).json() == expected_research
        assert client.get(board_url).json()["spec"] == board["spec"]
        assert client.get(base + f"/templates/{template['id']}").json() == template["spec"]
        assert client.get(output_url).content == archive
        for kind, content in frozen.items():
            assert client.get(board_url + "/artifact/" + kind).content == content
        if interrupted:
            resumed = wait_state(client, base + "/figure-boards/" + pending_id)
            assert resumed["spec"] == board["spec"] and resumed["panels"] == board["panels"]
            assert (
                client.get(base + "/figure-boards/" + pending_id + "/artifact/png").content
                == frozen["png"]
            )
        # The copied recipe remains usable on a replacement source.
        response = client.post(
            base + f"/templates/{template['id']}/preview",
            data={"expectedRevision": revision, "mapping": "{}"},
            files={
                "file": (
                    "replacement.csv",
                    b"s,repeat,v,unit\na,1,2,mV\na,2,4,mV\nb,1,6,mV\nb,2,8,mV\n",
                    "text/csv",
                )
            },
        )
        assert response.status_code == 200 and response.json()["figureReady"], response.text
    assert hashes(original) == before


def test_cli_rejects_damaged_backup_without_creating_restore() -> None:
    restored_root = ROOT / "V2.0/api/.labviz/v23-restored"
    original = restored_root / ("phase6-corruption-fixture-" + uuid4().hex)
    make_data(original)
    before = hashes(original)
    backup = Path(cli("Backup", DataDirectory=original)["backupDirectory"])
    assert backup.resolve().is_relative_to((ROOT / "outputs/v23-backups").resolve())
    (backup / "objects/raw.csv").write_bytes(b"damaged synthetic backup")
    existing = set(restored_root.iterdir())
    result = invoke_cli("Restore", BackupDirectory=backup)
    assert result.returncode != 0
    assert "Backup hash mismatch" in result.stderr
    assert not result.stdout.strip()
    assert set(restored_root.iterdir()) == existing
    assert hashes(original) == before
