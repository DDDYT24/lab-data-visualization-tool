from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from labviz_api.local_schema import CURRENT_REVISION, install_local_baseline, prepare_local_database
from labviz_api.repository import ProjectRepository
from labviz_api.research_dataset import decode_raw, encode_raw
from labviz_api.research_preparation import PreparationPlan, parse_sources, prepare
from labviz_api.research_rows import count, observation_page, sample_page
from labviz_api.research_store import ResearchStore, snapshot

TTL = 7_200
FIRST = b"sample,v,unit\nA,1.5,mg\nB,,mg\nC,3,mg\n"
SECOND = "sample,v,unit\nD,4,mg\nÉ,5.25,mg\n".encode()


def sources(*files: tuple[str, bytes]) -> dict[str, Any]:
    return parse_sources("p", list(files))


def plan(**extra: Any) -> PreparationPlan:
    return PreparationPlan.model_validate(
        {"mapping": {"sample": "sample", "value": "v", "unit": "unit"}, **extra}
    )


def legacy_profile(database: Path) -> list[tuple[str, dict[str, Any], str]]:
    """Revisions written before 0103: sources and snapshot inline, a two-step undo chain."""
    install_local_baseline(database, project_ttl_seconds=TTL)
    stacked = sources(("one.csv", FIRST), ("two.csv", SECOND))
    single = sources(("one.csv", FIRST))
    chain = [
        ("r1", stacked, snapshot(prepare(stacked, plan()))),
        ("r2", stacked, snapshot(prepare(stacked, plan(analysisUnit="sample")))),
        ("r3", single, snapshot(prepare(single, plan()))),
    ]
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO projects (id, title, source_json, storage_mode, updated_at) "
            "VALUES ('p', 'Kept', '{}', 'local', '2026-01-01T00:00:00Z')"
        )
        parent = None
        for revision_id, raw, prepared in chain:
            connection.execute(
                "INSERT INTO research_revisions (id, project_id, parent_id, raw_json, "
                "prepared_json, created_at) VALUES (?, 'p', ?, ?, ?, '2026-01-01T00:00:00Z')",
                (revision_id, parent, json.dumps(raw), prepared),
            )
            parent = revision_id
        connection.execute(
            "INSERT INTO research_heads (project_id, revision_id, future_json) "
            "VALUES ('p', 'r2', '[\"r3\"]')"
        )
    return chain


def test_parsed_sources_round_trip_through_parquet() -> None:
    raw = sources(("one.csv", FIRST), ("two.csv", SECOND))

    dataset = encode_raw(raw)

    assert decode_raw(dataset.payload, dataset.sha256, namespace="p", sources=raw["sources"]) == raw
    assert dataset.row_count == 5


def test_rows_whose_origins_cannot_be_recorded_are_refused() -> None:
    raw = sources(("one.csv", FIRST))
    raw["rows"][0]["origins"]["v"][0]["row"] = 99

    with pytest.raises(ValueError, match="origins"):
        encode_raw(raw)


def test_upgrading_moves_revisions_onto_shared_dataset_versions(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    chain = legacy_profile(database)

    assert prepare_local_database(database, project_ttl_seconds=TTL) == CURRENT_REVISION

    store = ResearchStore(database)
    with store.connect() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(research_revisions)")}
        versions = dict(
            connection.execute("SELECT id, dataset_version_id FROM research_revisions").fetchall()
        )
        parents = dict(connection.execute("SELECT id, parent_id FROM research_revisions"))
        head = connection.execute("SELECT revision_id, future_json FROM research_heads").fetchone()
        for revision_id, raw, prepared in chain:
            assert store.raw(connection, "p", revision_id) == raw
            assert store.prepared(connection, "p", revision_id) == prepared
    assert {"raw_json", "prepared_json"}.isdisjoint(columns)
    # The two revisions of the stacked input share one version; the single file has its own.
    assert versions["r1"] == versions["r2"] != versions["r3"]
    assert parents == {"r1": None, "r2": "r1", "r3": "r2"}
    assert tuple(head) == ("r2", '["r3"]')


def test_a_missing_snapshot_is_rebuilt_from_the_sources_and_plan(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    chain = legacy_profile(database)
    prepare_local_database(database, project_ttl_seconds=TTL)
    store = ResearchStore(database)
    with store.connect() as connection:
        connection.execute("DELETE FROM research_prepared")

    current = store.current("p")

    assert current is not None
    assert json.loads(current["prepared_json"]) == json.loads(chain[1][2])
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM research_prepared").fetchone()[0] == 1


def test_applying_a_preview_of_the_same_sources_reuses_their_version(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database)
    prepare_local_database(database, project_ttl_seconds=TTL)
    store = ResearchStore(database)
    with store.connect() as connection:
        raw = store.raw(connection, "p", "r2")
        before = connection.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0]
    prepared = prepare(raw, plan(analysisUnit="observation"))

    token = store.stage("p", "r2", raw, prepared)
    applied = store.apply("p", token, "r2")
    store.move("p", applied, "undo")

    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0] == before
        assert store.raw(connection, "p", applied) == raw
        assert json.loads(store.prepared(connection, "p", applied)) == prepared
        head = connection.execute("SELECT revision_id, future_json FROM research_heads").fetchone()
    assert tuple(head) == ("r2", json.dumps([applied]))


def test_a_copied_project_keeps_its_research_history(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    chain = legacy_profile(database)
    prepare_local_database(database, project_ttl_seconds=TTL)
    repository = ProjectRepository(database, TTL)

    repository.duplicate_project(
        source_project_id="p", project_id="copy", job_id="job", owner_user_id="user"
    )
    assert repository.delete_project("p")
    repository.cleanup_expired()

    store = ResearchStore(database)
    current = store.current("copy")
    assert current is not None
    assert current["prepared_json"] == chain[1][2]
    with store.connect() as connection:
        assert store.raw(connection, "copy", current["id"]) == chain[1][1]


def snapshot_of(database: Path, revision_id: str) -> dict[str, Any]:
    store = ResearchStore(database)
    with store.connect() as connection:
        return dict(json.loads(store.prepared(connection, "p", revision_id)))


def test_upgrading_writes_every_revision_as_rows(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    chain = legacy_profile(database)

    prepare_local_database(database, project_ttl_seconds=TTL)

    store = ResearchStore(database)
    with store.connect() as connection:
        for revision_id, _, prepared_json in chain:
            prepared = json.loads(prepared_json)
            observations = prepared["observations"]
            assert count(connection, revision_id, "observations") == len(observations)
            assert observation_page(connection, revision_id, 0, 200) == observations
            assert observation_page(connection, revision_id, 1, 2) == observations[1:3]
            assert sample_page(connection, revision_id, 0, 200) == prepared["samples"]
            analysis = connection.execute(
                "SELECT analysis_unit, observation_count, sample_count, independent_unit_count, "
                "summary_json FROM research_analyses WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            summary = prepared["summary"]
            assert tuple(analysis)[:4] == (
                summary["analysisUnit"],
                summary["observations"],
                summary["samples"],
                summary["independentUnits"],
            )
            assert json.loads(analysis["summary_json"]) == summary


def test_rows_answer_provenance_and_state_queries_in_sql(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    chain = legacy_profile(database)
    prepare_local_database(database, project_ttl_seconds=TTL)
    raw = chain[0][1]
    second = raw["sources"][1]["id"]

    with ResearchStore(database).connect() as connection:
        traced = connection.execute(
            "SELECT o.observation_id, o.sample_label FROM research_lineage l "
            "JOIN research_observations o ON o.revision_id=l.revision_id "
            "AND o.ordinal=l.observation_ordinal "
            "WHERE l.revision_id='r1' AND l.source_id=? AND l.source_row=3",
            (second,),
        ).fetchall()
        missing = connection.execute(
            "SELECT sample_label FROM research_observations WHERE revision_id='r1' "
            "AND state='missing'"
        ).fetchall()
    # Row 3 of two.csv is É; B has no reading in one.csv.
    assert [tuple(row) for row in traced] == [(raw["rows"][4]["id"], "É")]
    assert [row[0] for row in missing] == ["B"]


def test_applied_and_copied_revisions_keep_their_rows(tmp_path: Path) -> None:
    database = tmp_path / "labviz.db"
    legacy_profile(database)
    prepare_local_database(database, project_ttl_seconds=TTL)
    store = ResearchStore(database)
    with store.connect() as connection:
        raw = store.raw(connection, "p", "r2")
    steps = [{"kind": "filter", "field": "sample", "values": ["A", "D"]}]
    prepared = prepare(raw, plan(steps=steps))
    applied = store.apply("p", store.stage("p", "r2", raw, prepared), "r2")
    repository = ProjectRepository(database, TTL)
    repository.duplicate_project(
        source_project_id="p", project_id="copy", job_id="job", owner_user_id="user"
    )
    copied = ResearchStore(database).current("copy")
    assert copied is not None

    with store.connect() as connection:
        for revision_id in (applied, copied["id"]):
            assert observation_page(connection, revision_id, 0, 200) == prepared["observations"]
            recorded = connection.execute(
                "SELECT kind, input_rows, output_rows FROM research_steps WHERE revision_id=?",
                (revision_id,),
            ).fetchall()
            assert [tuple(row) for row in recorded] == [("filter", 5, 2)]

    assert repository.delete_project("p")
    with store.connect() as connection:
        assert count(connection, applied, "observations") == 0
        assert count(connection, copied["id"], "observations") == 2
