"""Project-scoped local workflow reuse and immutable research output routes."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import APIRouter, Cookie, Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import Field, ValidationError

from .research_contract import ResearchModel
from .research_outputs import OutputNotes, freeze_output, preview_output
from .research_preparation import LIMITS, PreparationError, parse_sources
from .research_reuse import (
    WorkflowTemplate,
    apply_template,
    install_reuse,
    preview_template,
    read_template,
    save_template,
    template_from_board,
)
from .research_store import ResearchStore


class TemplateRequest(ResearchModel):
    board_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=100)


class ApplyTemplate(ResearchModel):
    token: str = Field(min_length=1, max_length=128)
    expected_revision: str | None = None


class OutputRequest(ResearchModel):
    board_id: str = Field(min_length=1, max_length=128)


class FreezeRequest(ResearchModel):
    token: str = Field(min_length=1, max_length=128)
    notes: OutputNotes
    reviewed_included_data: bool


class FieldMapping(ResearchModel):
    fields: dict[str, str] = Field(max_length=128)


def register_reuse_routes(
    app: FastAPI,
    store: ResearchStore | None,
    available: Callable[[], ResearchStore],
    user_dep: Any,
    repository_dep: Any,
    require_access: Callable[..., Any],
    guest_cookie: str,
    slots: Any,
    execution: Any,
) -> None:
    if store is not None:
        install_reuse(store)

    def access(
        project_id: str,
        user: Any = user_dep,
        guest_token: str | None = Cookie(default=None, alias=guest_cookie),
        repository: Any = repository_dep,
    ) -> None:
        require_access(repository, project_id, user, guest_token, ready=True)
        available()

    router = APIRouter(
        prefix="/api/v1/projects/{project_id}/research", dependencies=[Depends(access)]
    )

    @contextmanager
    def work() -> Iterator[None]:
        if not slots.acquire(blocking=False):
            raise PreparationError("research-busy", "Research processing is busy. Retry shortly.")
        try:
            with execution:
                yield
        finally:
            slots.release()

    @router.get("/templates")
    def templates(project_id: str) -> dict[str, Any]:
        with available().connect() as c:
            return {
                "templates": [
                    {
                        "id": r["id"],
                        "spec": json.loads(r["spec_json"]),
                        "createdAt": r["created_at"],
                    }
                    for r in c.execute(
                        (
                            "SELECT * FROM research_templates WHERE project_id=? ORDER BY "
                            "created_at DESC"
                        ),
                        (project_id,),
                    )
                ]
            }

    @router.post("/templates")
    def create(project_id: str, body: TemplateRequest) -> dict[str, Any]:
        with work():
            active = available()
            return save_template(
                active,
                project_id,
                template_from_board(active, project_id, body.board_id, body.name),
            )

    @router.post("/templates/import")
    def import_template(project_id: str, body: WorkflowTemplate) -> dict[str, Any]:
        return save_template(available(), project_id, body)

    @router.get("/templates/{template_id}")
    def read(project_id: str, template_id: str) -> dict[str, Any]:
        return read_template(available(), project_id, template_id).model_dump()

    upload = File(...)
    mapping_form = Form("{}")
    revision_form = Form(None, alias="expectedRevision")

    @router.post("/templates/{template_id}/preview")
    def preview(
        project_id: str,
        template_id: str,
        file: UploadFile = upload,
        mapping: str = mapping_form,
        expected_revision: str | None = revision_form,
    ) -> dict[str, Any]:
        with work():
            try:
                fields = FieldMapping.model_validate({"fields": json.loads(mapping)}).fields
            except (ValidationError, ValueError) as error:
                raise PreparationError(
                    "template-mapping", "Use a JSON field-to-field mapping."
                ) from error
            raw = parse_sources(
                project_id,
                [
                    (
                        file.filename or "replacement.csv",
                        file.file.read(LIMITS["singleFileBytes"] + 1),
                    )
                ],
            )
            active = available()
            return preview_template(
                active,
                project_id,
                read_template(active, project_id, template_id),
                raw,
                fields,
                expected_revision,
            )

    @router.post("/template-apply")
    def apply(project_id: str, body: ApplyTemplate) -> dict[str, Any]:
        with work():
            return apply_template(available(), project_id, body.token, body.expected_revision)

    @router.post("/outputs/preview")
    def output_preview(project_id: str, body: OutputRequest) -> dict[str, Any]:
        with work():
            return preview_output(available(), project_id, body.board_id)

    @router.post("/outputs")
    def freeze(project_id: str, body: FreezeRequest) -> dict[str, Any]:
        if not body.reviewed_included_data:
            raise PreparationError("output-review", "Review included data and identifiers first.")
        with work():
            return freeze_output(available(), project_id, body.token, body.notes)

    @router.get("/outputs")
    def outputs(project_id: str) -> dict[str, Any]:
        with available().connect() as c:
            return {
                "outputs": [
                    {
                        "id": r["id"],
                        "boardId": r["board_id"],
                        "sha256": r["sha256"],
                        "createdAt": r["created_at"],
                    }
                    for r in c.execute(
                        "SELECT id,board_id,sha256,created_at FROM research_outputs "
                        "WHERE project_id=? ORDER BY created_at DESC",
                        (project_id,),
                    )
                ]
            }

    @router.get("/outputs/{output_id}/download")
    def download(project_id: str, output_id: str) -> Response:
        with available().connect() as c:
            row = c.execute(
                "SELECT id,zip_blob,sha256 FROM research_outputs WHERE project_id=? AND id=?",
                (project_id, output_id),
            ).fetchone()
        if row is None:
            raise PreparationError(
                "output-not-found", "This output is not available in the project."
            )
        return Response(
            bytes(row["zip_blob"]),
            media_type="application/zip",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-SHA256": row["sha256"],
                "Content-Disposition": f'attachment; filename="research-{row["id"]}.zip"',
            },
        )

    app.include_router(router)
