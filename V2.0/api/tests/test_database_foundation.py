from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from labviz_api.db.models import (
    ChartSpecRevision,
    Dataset,
    DatasetVersion,
    GuestSession,
    ProcessingRun,
    Project,
    ProjectRevision,
    SourceFile,
    StoredObject,
    User,
)
from labviz_api.db.session import Database

API_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_URL = "postgresql+psycopg://labviz:labviz-local@127.0.0.1:54329/labviz_test"
CORE_TABLES = {
    "users",
    "projects",
    "stored_objects",
    "source_files",
    "datasets",
    "dataset_versions",
    "processing_runs",
    "chart_spec_revisions",
    "project_revisions",
    "quality_reports",
    "quality_findings",
    "cleaning_decision_sets",
    "cleaning_decisions",
    "guest_sessions",
    "auth_challenges",
    "auth_sessions",
    "auth_requests",
    "project_claims",
    "project_origins",
    "project_lifecycle_events",
    "idempotency_records",
}


def alembic_config(url: str) -> Config:
    config = Config(str(API_ROOT / "alembic.ini"))
    config.attributes["database_url"] = url
    return config


@pytest.fixture(scope="module")
def postgres_database() -> Iterator[Database]:
    configured_url = os.environ.get("LABVIZ_TEST_POSTGRES_URL")
    database = Database(configured_url or DEFAULT_TEST_URL)
    health = database.health()
    if not health.ready:
        database.dispose()
        if configured_url:
            pytest.fail("LABVIZ_TEST_POSTGRES_URL is configured but PostgreSQL is unavailable.")
        pytest.skip("Local PostgreSQL is not running; start it with docker compose.")

    config = alembic_config(database.engine.url.render_as_string(hide_password=False))
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    try:
        yield database
    finally:
        command.upgrade(config, "head")
        database.dispose()


@pytest.fixture
def db_session(postgres_database: Database) -> Iterator[Session]:
    connection = postgres_database.engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


def test_initial_migration_is_upgradeable_reversible_and_current(
    postgres_database: Database,
) -> None:
    config = alembic_config(postgres_database.engine.url.render_as_string(hide_password=False))
    command.downgrade(config, "base")
    assert not CORE_TABLES.intersection(inspect(postgres_database.engine).get_table_names())

    command.upgrade(config, "head")
    assert CORE_TABLES.issubset(inspect(postgres_database.engine).get_table_names())
    command.check(config)


def test_database_health_and_postgres_only_guard(postgres_database: Database) -> None:
    assert postgres_database.health().ready
    with pytest.raises(ValueError, match="must use PostgreSQL"):
        Database("sqlite+pysqlite:///:memory:")


def test_project_constraints_require_owner_expiry_and_valid_recovery_window(
    db_session: Session,
) -> None:
    db_session.add(Project(storage_mode="saved-cloud", title="No owner"))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()

    db_session.add(Project(storage_mode="temporary-cloud", title="No expiry"))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()

    now = datetime.now(UTC)
    owner = User(email="recovery@example.com")
    db_session.add(
        Project(
            owner=owner,
            storage_mode="saved-cloud",
            title="Recoverable for exactly 24 hours",
            deleted_at=now,
            purge_after=now + timedelta(hours=24),
        )
    )
    db_session.flush()

    db_session.add(
        Project(
            storage_mode="local",
            title="Invalid recovery",
            deleted_at=now,
            purge_after=now - timedelta(hours=1),
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_version_lineage_relationships_form_a_reproducible_project(
    db_session: Session,
) -> None:
    now = datetime.now(UTC)
    user = User(email="researcher@example.com")
    project = Project(
        owner=user,
        storage_mode="saved-cloud",
        title="Thermal response",
        description="Reproducible experiment",
    )
    source = SourceFile(
        project=project,
        original_name="measurements.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        size_bytes=1_963_595,
        sha256="a" * 64,
        sheet_name="Measurements",
        header_row=1,
        parser_name="pandas-openpyxl",
        parser_version="1",
        binary_deleted_at=now,
        parsed_at=now,
    )
    dataset = Dataset(
        project=project,
        source_file=source,
        name="Measurements",
        sheet_name="Measurements",
        header_row=1,
    )
    data_object = StoredObject(
        storage_backend="local",
        object_key="datasets/project/version-1.parquet",
        purpose="dataset",
        status="available",
        media_type="application/vnd.apache.parquet",
        size_bytes=1234,
        sha256="b" * 64,
    )
    baseline = DatasetVersion(
        project=project,
        dataset=dataset,
        stored_object=data_object,
        version_number=1,
        kind="parsed",
        schema_document={"columns": [{"name": "time", "type": "number"}]},
        row_count=2_260,
        column_count=120,
    )
    cleaned_object = StoredObject(
        storage_backend="local",
        object_key="datasets/project/version-2.parquet",
        purpose="dataset",
        status="available",
        media_type="application/vnd.apache.parquet",
        size_bytes=1200,
        sha256="c" * 64,
    )
    cleaned_version_id = uuid4()
    cleaned = DatasetVersion(
        id=cleaned_version_id,
        project=project,
        dataset=dataset,
        parent_version=baseline,
        stored_object=cleaned_object,
        version_number=2,
        kind="cleaned",
        schema_document={"columns": [{"name": "time", "type": "number"}]},
        row_count=2_250,
        column_count=120,
    )
    chart = ChartSpecRevision(
        project=project,
        dataset_version=cleaned,
        created_by=user,
        revision_number=1,
        schema_version=1,
        spec_document={"schemaVersion": 1, "type": "line"},
    )
    project_revision = ProjectRevision(
        project=project,
        active_dataset_version=cleaned,
        chart_spec_revision=chart,
        created_by=user,
        revision_number=1,
        spec_schema_version=1,
        spec_document={
            "schemaVersion": 1,
            "source": {"datasetVersionId": str(cleaned_version_id)},
            "chart": {"schemaVersion": 1},
        },
    )
    run = ProcessingRun(
        project=project,
        input_dataset_version=baseline,
        output_dataset_version=cleaned,
        operation="clean",
        status="succeeded",
        parameters={"sheetName": "Measurements", "headerRow": 1},
        algorithm_version="cleaning-v1",
        code_version="test-commit",
        started_at=now,
        finished_at=now,
    )
    db_session.add_all([project_revision, run])
    db_session.flush()
    project.current_revision = project_revision
    db_session.flush()
    db_session.expire_all()

    stored = db_session.scalar(select(Project).where(Project.id == project.id))
    assert stored is not None
    assert stored.current_revision is not None
    active = stored.current_revision.active_dataset_version
    assert active.version_number == 2
    assert active.parent_version is not None
    assert active.parent_version.version_number == 1
    assert active.dataset.source_file.original_name == "measurements.xlsx"
    assert stored.current_revision.spec_document["source"]["datasetVersionId"] == str(active.id)
    assert stored.current_revision.chart_spec_revision.spec_document["type"] == "line"
    assert stored.current_revision.chart_spec_revision.dataset_version.id == active.id
    assert active.output_of_run is not None
    assert active.output_of_run.operation == "clean"
    assert active.output_of_run.input_dataset_version is not None
    assert active.output_of_run.input_dataset_version.version_number == 1


def test_dataset_version_numbers_are_unique_per_dataset(db_session: Session) -> None:
    now = datetime.now(UTC)
    project = Project(
        storage_mode="temporary-cloud",
        title="Temporary",
        expires_at=now + timedelta(hours=2),
    )
    source = SourceFile(
        project=project,
        original_name="data.csv",
        media_type="text/csv",
        size_bytes=20,
        sha256="c" * 64,
        parser_name="pandas-csv",
        parser_version="1",
    )
    dataset = Dataset(project=project, source_file=source, name="data")
    first_object = StoredObject(
        storage_backend="local",
        object_key="datasets/unique-1.parquet",
        purpose="dataset",
        status="available",
        media_type="application/vnd.apache.parquet",
        size_bytes=10,
        sha256="d" * 64,
    )
    second_object = StoredObject(
        storage_backend="local",
        object_key="datasets/unique-2.parquet",
        purpose="dataset",
        status="available",
        media_type="application/vnd.apache.parquet",
        size_bytes=10,
        sha256="e" * 64,
    )
    db_session.add_all(
        [
            DatasetVersion(
                project=project,
                dataset=dataset,
                stored_object=first_object,
                version_number=1,
                kind="parsed",
                schema_document={"columns": ["x"]},
                row_count=1,
                column_count=1,
            ),
            DatasetVersion(
                project=project,
                dataset=dataset,
                stored_object=second_object,
                version_number=1,
                kind="derived",
                schema_document={"columns": ["x"]},
                row_count=1,
                column_count=1,
            ),
        ]
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_chart_revision_cannot_reference_another_projects_dataset(
    db_session: Session,
) -> None:
    now = datetime.now(UTC)

    def project_with_version(label: str, digest: str) -> tuple[Project, DatasetVersion]:
        guest_session = GuestSession(
            token_digest=(digest * 64)[:64],
            status="active",
            expires_at=now + timedelta(hours=2),
            last_seen_at=now,
            created_at=now,
        )
        project = Project(
            guest_session=guest_session,
            storage_mode="temporary-cloud",
            title=label,
            expires_at=now + timedelta(hours=2),
        )
        source = SourceFile(
            project=project,
            original_name=f"{label}.csv",
            media_type="text/csv",
            size_bytes=10,
            sha256=digest * 64,
            parser_name="pandas-csv",
            parser_version="1",
        )
        dataset = Dataset(project=project, source_file=source, name=label)
        stored_object = StoredObject(
            storage_backend="local",
            object_key=f"datasets/{label}.parquet",
            purpose="dataset",
            status="available",
            media_type="application/vnd.apache.parquet",
            size_bytes=10,
            sha256=digest * 64,
        )
        version = DatasetVersion(
            project=project,
            dataset=dataset,
            stored_object=stored_object,
            version_number=1,
            kind="parsed",
            schema_document={"columns": ["x"]},
            row_count=1,
            column_count=1,
        )
        return project, version

    first_project, _first_version = project_with_version("first", "1")
    _second_project, second_version = project_with_version("second", "2")
    db_session.add_all([first_project, _second_project])
    db_session.flush()
    db_session.add(
        ChartSpecRevision(
            project_id=first_project.id,
            dataset_version_id=second_version.id,
            revision_number=1,
            schema_version=1,
            spec_document={"schemaVersion": 1, "type": "line"},
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()
