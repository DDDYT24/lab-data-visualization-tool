from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select, text

from labviz_api.auth import AuthService, MemoryEmailSender
from labviz_api.config import Settings
from labviz_api.db.models import Experiment, ExperimentRun, ProcessingRun, Project, User
from labviz_api.db.session import Database
from labviz_api.main import create_app
from labviz_api.persistence.postgres import PostgresProjectStore
from labviz_api.repository import ProjectRepository
from labviz_api.storage import LocalObjectStorage

API_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_URL = os.environ.get(
    "LABVIZ_TEST_POSTGRES_URL",
    "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz_test",
)


def _alembic_config() -> Config:
    config = Config(str(API_ROOT / "alembic.ini"))
    config.attributes["database_url"] = POSTGRES_URL
    return config


@pytest.fixture
def local_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, MemoryEmailSender, ProjectRepository]]:
    settings = Settings(
        database_path=tmp_path / "experiments.db",
        allowed_origins=("http://localhost:3000",),
        public_web_url="http://localhost:3000",
    )
    repository = ProjectRepository(settings.database_path, settings.project_ttl_seconds)
    sender = MemoryEmailSender()
    auth = AuthService(sender, settings.session_ttl_seconds, repository)
    with TestClient(create_app(settings, repository, auth)) as client:
        yield client, sender, repository


def _sign_in(client: TestClient, sender: MemoryEmailSender) -> None:
    requested = client.post(
        "/api/v1/auth/email-code", json={"email": "experiment-owner@example.com"}
    )
    verified = client.post(
        "/api/v1/auth/email-code/verify",
        json={"challengeId": requested.json()["challengeId"], "code": sender.messages[-1][1]},
    )
    assert verified.status_code == 200


def _chart() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "type": "line",
        "title": "Repeated response",
        "xAxis": {"field": "time", "title": "Time", "unit": "s"},
        "yAxis": {"field": "response", "title": "Response", "unit": "mV"},
        "series": [{"field": "response", "label": "Response", "color": "#2563EB"}],
        "panelCount": 1,
        "export": {
            "format": "svg",
            "dpi": 300,
            "sizePreset": "double-column",
            "grayscalePreview": False,
        },
    }


def test_multi_file_replicates_share_experiment_and_export_provenance(
    local_client: tuple[TestClient, MemoryEmailSender, ProjectRepository],
) -> None:
    client, sender, repository = local_client
    project_ids: list[str] = []
    experiment_ids: set[str] = set()
    run_ids: set[str] = set()

    for index in (1, 2):
        response = client.post(
            "/api/v1/projects",
            files={
                "file": (
                    f"replicate-{index}.csv",
                    b"time,response\n0,1.0\n1,1.5\n2,2.1\n",
                    "text/csv",
                )
            },
            data={
                "experimentTitle": "Dose response study",
                "runLabel": f"Acquisition {index}",
                "replicateId": f"R{index}",
                "batchId": "B-2026-09",
            },
        )
        assert response.status_code == 202
        body = response.json()
        project_ids.append(body["projectId"])
        experiment_ids.add(body["experiment"]["experimentId"])
        run_ids.add(body["experiment"]["experimentRunId"])
        assert body["experiment"]["replicateId"] == f"R{index}"
        if index == 1:
            _sign_in(client, sender)
        saved = client.post(f"/api/v1/projects/{body['projectId']}/save")
        assert saved.status_code == 200

    assert len(experiment_ids) == 1
    assert len(run_ids) == 2
    with repository._connect() as connection:
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM experiment_runs").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2

    history = client.get("/api/v1/projects")
    assert history.status_code == 200
    assert {item["experiment"]["replicateId"] for item in history.json()["projects"]} == {
        "R1",
        "R2",
    }

    exported = client.post(f"/api/v1/projects/{project_ids[0]}/exports", json={"chart": _chart()})
    assert exported.status_code == 200
    assert exported.json()["experiment"]["runLabel"] == "Acquisition 1"
    download = client.get(exported.json()["downloadUrl"])
    assert download.status_code == 200
    assert download.headers["x-labviz-experiment-id"] in experiment_ids
    assert download.headers["x-labviz-experiment-run-id"] in run_ids
    assert b"Dose response study" in download.content
    assert b"Acquisition 1" in download.content
    with repository._connect() as connection:
        snapshot = connection.execute(
            "SELECT experiment_json FROM exports WHERE id = ?", (exported.json()["id"],)
        ).fetchone()[0]
    assert json.loads(snapshot)["replicateId"] == "R1"

    deleted = client.delete(f"/api/v1/projects/{project_ids[0]}")
    assert deleted.status_code == 204
    with repository._connect() as connection:
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM experiment_runs").fetchone()[0] == 1


def test_experiment_metadata_requires_an_experiment_title(
    local_client: tuple[TestClient, MemoryEmailSender, ProjectRepository],
) -> None:
    client, _sender, _repository = local_client
    response = client.post(
        "/api/v1/projects",
        files={"file": ("run.csv", b"x,y\n1,2\n", "text/csv")},
        data={"replicateId": "R1"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "experiment-title-required"


def test_reimport_reuses_the_same_physical_run(
    local_client: tuple[TestClient, MemoryEmailSender, ProjectRepository],
) -> None:
    client, _sender, repository = local_client
    metadata = {
        "experimentTitle": "Header correction study",
        "runLabel": "Acquisition A",
        "replicateId": "R1",
    }
    first = client.post(
        "/api/v1/projects",
        files={"file": ("run.csv", b"note\ntime,response\n0,1\n", "text/csv")},
        data={**metadata, "headerRow": "2"},
    )
    assert first.status_code == 202
    physical_run_id = first.json()["experiment"]["experimentRunId"]

    corrected = client.post(
        "/api/v1/projects",
        files={"file": ("run.csv", b"note\ntime,response\n0,1\n", "text/csv")},
        data={**metadata, "headerRow": "2", "experimentRunId": physical_run_id},
    )
    assert corrected.status_code == 202
    assert corrected.json()["experiment"]["experimentRunId"] == physical_run_id
    with repository._connect() as connection:
        assert connection.execute("SELECT count(*) FROM experiment_runs").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2


def test_postgres_migration_and_project_restore_preserve_physical_run(tmp_path: Path) -> None:
    database = Database(POSTGRES_URL)
    if not database.health().ready:
        database.dispose()
        pytest.skip("Local PostgreSQL is not running; start it with docker compose.")
    config = _alembic_config()
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    try:
        columns = {column["name"] for column in inspect(database.engine).get_columns("projects")}
        assert "experiment_run_id" in columns
        export_columns = {
            column["name"] for column in inspect(database.engine).get_columns("publication_exports")
        }
        assert {"experiment_id", "experiment_run_id", "replicate_id_snapshot"}.issubset(
            export_columns
        )

        owner_id = uuid4()
        experiment_id = uuid4()
        run_id = uuid4()
        project_id = uuid4()
        now = datetime.now(UTC)
        with database.session() as session:
            owner = User(id=owner_id, email="p1-3-owner@example.com")
            experiment = Experiment(
                id=experiment_id,
                owner=owner,
                title="Migration experiment",
                created_at=now,
                updated_at=now,
            )
            run = ExperimentRun(
                id=run_id,
                experiment=experiment,
                run_label="Acquisition A",
                replicate_id="R1",
                batch_id="B1",
                created_at=now,
            )
            session.add(
                Project(
                    id=project_id,
                    owner=owner,
                    experiment_run=run,
                    storage_mode="saved-cloud",
                    title="Restorable analysis",
                    description="",
                    saved_at=now,
                    last_activity_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )

        store = PostgresProjectStore(database, LocalObjectStorage(tmp_path / "objects"), 7_200)
        assert store.delete_project(
            project_id.hex, owner_user_id=owner_id.hex, guest_token_digest=None
        )
        with database.session() as session:
            deleted = session.get(Project, project_id)
            assert deleted is not None and deleted.experiment_run_id == run_id
            assert session.scalar(select(ExperimentRun).where(ExperimentRun.id == run_id))
            software_run = session.scalar(
                select(ProcessingRun).where(ProcessingRun.project_id == project_id)
            )
            assert software_run is None

        store.restore_deleted_project(project_id.hex, owner_id.hex)
        with database.session() as session:
            restored = session.get(Project, project_id)
            assert restored is not None and restored.deleted_at is None
            assert restored.experiment_run_id == run_id

        with pytest.raises(RuntimeError, match="Experiment provenance exists"):
            command.downgrade(config, "0009_atomic_auth_rate_limits")
        assert inspect(database.engine).has_table("experiment_runs")
    finally:
        with database.engine.begin() as connection:
            connection.execute(text("TRUNCATE TABLE users, projects, experiments CASCADE"))
        command.downgrade(config, "base")
        database.dispose()
