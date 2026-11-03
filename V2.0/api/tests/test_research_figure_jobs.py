"""Durability, nonblocking writes and atomic artifact publication for saved layouts."""

from __future__ import annotations

import gc
import json
import threading
import time
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_research_api import project, settings, unlock
from test_research_figure_board import charts

import labviz_api.research_figure_jobs as jobs
from labviz_api.main import create_app
from labviz_api.research_figure_board import artifact_row, read_board
from labviz_api.research_figure_render import render_board
from labviz_api.research_store import ResearchStore


def wait_state(client: TestClient, url: str, state: str = "ready") -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        view = response.json()
        if view["artifactState"] == state:
            return dict(view)
        if state != "failed":
            assert view["artifactState"] != "failed", view
        time.sleep(0.02)
    raise AssertionError("Figure generation did not reach the expected state")


def test_layout_commits_before_render_and_does_not_block_other_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original_render = render_board

    def blocked_render(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(30)
        return original_render(*args, **kwargs)

    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        monkeypatch.setattr(jobs, "render_board", blocked_render)
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        try:
            saved = client.post(
                base + "?deferArtifacts=true", json={"panelChartIds": [chart["id"]]}
            )
            assert saved.status_code == 202, saved.text
            body = saved.json()
            url = base + "/" + body["id"]
            assert body["artifactState"] == "queued" and not body["composed"]
            assert entered.wait(5)
            assert (
                read_board(ResearchStore(config.database_path), identifier, body["id"])["spec"]
                == body["spec"]
            )
            assert client.get(url + "/artifact/png").status_code == 409
            # A separate connection can commit while the render is deliberately blocked.
            with ResearchStore(config.database_path).connect() as connection:
                connection.execute(
                    "UPDATE research_figure_boards SET created_at=created_at WHERE id=?",
                    (body["id"],),
                )
            second = client.post(
                base + "?deferArtifacts=true",
                json={"title": "Next", "panelChartIds": [chart["id"]]},
            )
            assert second.status_code == 202, second.text
            third = client.post(
                base + "?deferArtifacts=true", json={"panelChartIds": [chart["id"]]}
            )
            assert third.status_code == 429
            assert len(client.get(base).json()["boards"]) == 2
        finally:
            release.set()
        ready = wait_state(client, url)
        assert ready["spec"] == body["spec"] and ready["composed"]
        assert ready["panels"][0]["resultSha256"] == chart["resultSha256"]
        for format_, signature in [("svg", b"<?xml"), ("png", b"\x89PNG"), ("pdf", b"%PDF")]:
            artifact = client.get(url + "/artifact/" + format_)
            assert artifact.status_code == 200 and artifact.content.startswith(signature)
        wait_state(client, base + "/" + second.json()["id"])


def test_failed_generation_retains_layout_and_retry_preserves_inputs_and_ready_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(tmp_path)
    original_render = render_board

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Injected renderer failure")

    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        monkeypatch.setattr(jobs, "render_board", fail)
        response = client.post(
            base + "?deferArtifacts=true",
            json={"title": "Keep my draft", "panelChartIds": [chart["id"]]},
        )
        assert response.status_code == 202
        url = base + "/" + response.json()["id"]
        failed = wait_state(client, url, "failed")
        assert failed["spec"]["title"] == "Keep my draft" and not failed["composed"]
        assert client.get(url + "/artifact/pdf").status_code == 409
        other = project(client)
        assert (
            client.post(
                f"/api/v1/projects/{other}/research/figure-boards/{failed['id']}/retry"
            ).status_code
            == 404
        )
        monkeypatch.setattr(jobs, "render_board", original_render)
        assert client.post(url + "/retry").status_code == 202
        ready = wait_state(client, url)
        assert ready["id"] == failed["id"] and ready["panels"] == failed["panels"]
        frozen = client.get(url + "/artifact/svg").content
        assert client.post(url + "/retry").status_code == 409
        assert client.get(url + "/artifact/svg").content == frozen
    with TestClient(create_app(config)) as reopened:
        unlock(reopened)
        assert reopened.get(url).json()["artifactState"] == "ready"
        assert reopened.get(url + "/artifact/svg").content == frozen


def test_restart_resumes_interrupted_immutable_generation(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        completed = client.post(base, json={"panelChartIds": [chart["id"]]}).json()
    new_id = uuid4().hex
    with ResearchStore(config.database_path).connect() as connection:
        connection.execute(
            "INSERT INTO research_figure_boards VALUES (?,?,?,?)",
            (new_id, identifier, json.dumps(completed["spec"]), completed["createdAt"]),
        )
        connection.execute(
            "INSERT INTO research_figure_jobs (board_id,state,panels_json,attempt) "
            "VALUES (?,'running',?,'abandoned-process')",
            (new_id, json.dumps(completed["panels"])),
        )
    with TestClient(create_app(config)) as reopened:
        unlock(reopened)
        ready = wait_state(reopened, base + "/" + new_id)
        assert ready["panels"] == completed["panels"] and ready["composed"]
        assert (
            reopened.get(base + "/" + completed["id"] + "/artifact/png").content
            == reopened.get(base + "/" + new_id + "/artifact/png").content
        )


def test_screen_preview_is_warmed_once_without_changing_frozen_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    calls = []
    original = Image.Image.thumbnail

    def counted(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        calls.append(image.size)
        original(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "thumbnail", counted)
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        saved = client.post(base + "?deferArtifacts=true", json={"panelChartIds": [chart["id"]]})
        url = base + "/" + saved.json()["id"]
        wait_state(client, url)
        frozen = client.get(url + "/artifact/png").content
    # Shutdown joins the worker, including its optional preview work.
    assert len(calls) == 1
    with TestClient(create_app(config)) as reopened:
        unlock(reopened)
        assert reopened.get(url + "/artifact/png?preview=true").status_code == 200
        assert len(calls) == 1
        assert reopened.get(url + "/artifact/png?download=true").content == frozen


def test_optional_preview_failure_preserves_committed_exports(tmp_path: Path) -> None:
    config = settings(tmp_path)
    called = threading.Event()
    app = create_app(config)
    with TestClient(app) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"

        def fail(project_id: str, board_id: str, payload: bytes) -> None:
            store = ResearchStore(config.database_path)
            assert read_board(store, project_id, board_id)["artifactState"] == "ready"
            assert artifact_row(store, project_id, board_id, "png")["png"] == payload
            called.set()
            raise RuntimeError("Injected optional preview failure")

        app.state.figure_jobs.on_preview = fail
        saved = client.post(base + "?deferArtifacts=true", json={"panelChartIds": [chart["id"]]})
        url = base + "/" + saved.json()["id"]
        assert called.wait(10)
        frozen = client.get(url + "/artifact/png").content
    with TestClient(create_app(config)) as reopened:
        unlock(reopened)
        assert reopened.get(url).json()["artifactState"] == "ready"
        assert reopened.get(url + "/artifact/png").content == frozen


def test_deleted_project_is_not_cached_by_a_late_warm_up(tmp_path: Path) -> None:
    from labviz_api import research_preview as previews

    entered, release = threading.Event(), threading.Event()
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        original = app.state.figure_jobs.on_preview

        def delayed(project_id: str, board_id: str, payload: bytes) -> None:
            entered.set()
            assert release.wait(10)
            original(project_id, board_id, payload)

        app.state.figure_jobs.on_preview = delayed
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        saved = client.post(base + "?deferArtifacts=true", json={"panelChartIds": [chart["id"]]})
        assert saved.status_code == 202
        try:
            assert entered.wait(10)
            assert client.delete(f"/api/v1/projects/{identifier}").status_code == 204
        finally:
            release.set()
    assert not any(key[0] == identifier for key in previews._cache)


def test_pending_copy_is_recoverable_without_reusing_original_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original_render = render_board

    def blocked(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(30)
        return original_render(*args, **kwargs)

    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        monkeypatch.setattr(jobs, "render_board", blocked)
        response = client.post(base + "?deferArtifacts=true", json={"panelChartIds": [chart["id"]]})
        assert response.status_code == 202
        try:
            assert entered.wait(5)
            copied = client.post(f"/api/v1/projects/{identifier}/duplicate")
            assert copied.status_code == 200, copied.text
            copied_base = f"/api/v1/projects/{copied.json()['projectId']}/research/figure-boards"
            copied_id = client.get(copied_base).json()["boards"][0]["id"]
            copied_view = client.get(copied_base + "/" + copied_id).json()
            assert copied_view["artifactState"] == "failed"
            assert copied_view["panels"][0]["chartId"] != chart["id"]
        finally:
            release.set()
        wait_state(client, base + "/" + response.json()["id"])
        assert client.post(copied_base + "/" + copied_id + "/retry").status_code == 202
        wait_state(client, copied_base + "/" + copied_id)


def test_generation_reuses_its_complete_scene_for_first_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import labviz_api.research_figure_routes as routes

    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        worker = cast(FastAPI, client.app).state.figure_jobs
        original_publish = worker.on_ready
        published = threading.Event()

        def publish(*args: Any) -> None:
            original_publish(*args)
            published.set()

        monkeypatch.setattr(worker, "on_ready", publish)
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        response = client.post(base + "?deferArtifacts=true", json={"panelChartIds": [chart["id"]]})
        assert response.status_code == 202
        assert published.wait(30)
        url = base + "/" + response.json()["id"]
        assert client.get(url).json()["artifactState"] == "ready"
        original_artifact = artifact_row

        def no_scene_reload(*args: Any) -> Any:
            assert args[-1] != "scene_json", "First inspection decoded the scene again"
            return original_artifact(*args)

        monkeypatch.setattr(routes, "artifact_row", no_scene_reload)
        inspected = client.get(url + "/interaction")
        assert inspected.status_code == 200 and inspected.json()["totalRows"] > 0


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stage", ["source", "render", "success"])
def test_build_restores_original_gc_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, stage: str
) -> None:
    config = settings(tmp_path)
    previous = gc.isenabled()
    try:
        with TestClient(create_app(config)) as client:
            unlock(client)
            identifier = project(client)
            chart = charts(client, identifier)[0]

            def unavailable(*args: Any, **kwargs: Any) -> Any:
                assert not gc.isenabled()
                if stage == "success":
                    return render_board(*args, **kwargs)
                raise RuntimeError("Injected source loading failure")

            monkeypatch.setattr(
                jobs, "source" if stage == "source" else "render_board", unavailable
            )
            (gc.enable if enabled else gc.disable)()
            base = f"/api/v1/projects/{identifier}/research/figure-boards"
            response = client.post(
                base + "?deferArtifacts=true", json={"panelChartIds": [chart["id"]]}
            )
            assert response.status_code == 202
            wait_state(
                client,
                base + "/" + response.json()["id"],
                "ready" if stage == "success" else "failed",
            )
            assert gc.isenabled() == enabled
    finally:
        (gc.enable if previous else gc.disable)()


def test_completed_copy_keeps_frozen_bytes_and_owns_new_generation(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[0]
        base = f"/api/v1/projects/{identifier}/research/figure-boards"
        saved = client.post(base + "?deferArtifacts=true", json={"panelChartIds": [chart["id"]]})
        assert saved.status_code == 202
        original = wait_state(client, base + "/" + saved.json()["id"])
        png = client.get(base + "/" + original["id"] + "/artifact/png").content
        copied = client.post(f"/api/v1/projects/{identifier}/duplicate")
        assert copied.status_code == 200
        copy_base = f"/api/v1/projects/{copied.json()['projectId']}/research/figure-boards"
        copy_id = client.get(copy_base).json()["boards"][0]["id"]
        copy = client.get(copy_base + "/" + copy_id).json()
        assert copy["artifactState"] == "ready" and copy["composed"]
        assert copy["panels"][0]["chartId"] != chart["id"]
        assert client.delete(f"/api/v1/projects/{identifier}").status_code == 204
        assert client.get(copy_base + "/" + copy_id + "/artifact/png").content == png
        revised = client.post(copy_base + "?deferArtifacts=true", json=copy["spec"])
        assert revised.status_code == 202
        ready = wait_state(client, copy_base + "/" + revised.json()["id"])
        assert ready["panels"][0]["resultSha256"] == original["panels"][0]["resultSha256"]


def test_evicting_scene_during_new_chart_keeps_old_selection_and_exports(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        unlock(client)
        identifier = project(client)
        chart = charts(client, identifier)[-1]
        base = f"/api/v1/projects/{identifier}/research"
        board = client.post(
            base + "/figure-boards", json={"panelChartIds": [chart["id"]] * 4}
        ).json()
        url = base + "/figure-boards/" + board["id"]
        view = client.get(url + "/interaction").json()
        mark = view["marks"][0]["id"]
        expected = client.get(url + "/interaction", params={"mark": mark}).json()
        store = ResearchStore(config.database_path)
        full_scene = artifact_row(store, identifier, board["id"])["scene_json"]
        binary_scene = artifact_row(store, identifier, board["id"], "scene_json")["scene_json"]
        assert isinstance(full_scene, str) and isinstance(binary_scene, bytes)
        assert json.loads(binary_scene) == json.loads(full_scene)
        artifacts = {
            kind: client.get(url + "/artifact/" + kind).content for kind in ("png", "svg", "pdf")
        }
        response = client.post(
            base + "/charts", json={"revisionId": chart["revisionId"], "spec": {"kind": "violin"}}
        )
        assert response.status_code == 200, response.text
        assert client.get(url + "/interaction", params={"mark": mark}).json() == expected
        compact = client.get(url + "/interaction", params={"mark": mark, "highlightsOnly": True})
        assert compact.status_code == 200
        assert compact.json() == {
            **expected,
            "marks": [item for item in expected["marks"] if item["highlighted"]],
        }
        for kind, payload in artifacts.items():
            assert client.get(url + "/artifact/" + kind).content == payload
