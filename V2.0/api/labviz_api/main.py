"""FastAPI application implementing the LabViz V2 API contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    Request,
    Response,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .auth import AuthError, AuthRepository, AuthService, build_auth_service
from .client_identity import ClientIdentityError, ClientIdentityResolver
from .config import Settings
from .models import (
    AuthenticatedUser,
    AuthState,
    ChartAnalysis,
    ChartRequest,
    ChartSpec,
    CleaningDecision,
    CleaningDecisionsRequest,
    CleaningDecisionsResponse,
    CreateShareRequest,
    DataPreview,
    ExportJob,
    HealthResponse,
    ProcessingJob,
    ProjectDescriptionResponse,
    ProjectList,
    ProjectSession,
    ProjectSummary,
    ProjectWorkspace,
    QualityReport,
    QualityRulesRequest,
    RequestEmailCode,
    RequestEmailCodeResponse,
    SavedChartResponse,
    SharedChart,
    SharedDownloads,
    ShareLink,
    ShareSummary,
    SourceFile,
    UpdateProjectDescriptionRequest,
    UpdateShareRequest,
    VerifyEmailCode,
    VerifyEmailCodeResponse,
)
from .persistence import PersistenceError, ProjectStore, build_project_store
from .persistence.contracts import ProjectReader
from .persistence.exceptions import (
    PersistenceConflict,
    PersistenceNotFound,
    PersistenceUnavailable,
)
from .processing import (
    ProcessingError,
    analyze_chart,
    build_preview,
    build_quality_report,
    default_chart_spec,
    load_dataframe,
    render_chart,
    sample_csv_bytes,
    validate_upload,
)
from .rate_limits import AuthRateLimitUnavailable
from .repository import ProjectRepository

LOGGER = logging.getLogger(__name__)
SESSION_COOKIE = "labviz_session"
GUEST_COOKIE = "labviz_guest"
API_PREFIX = "/api/v1"


class ApiProblem(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message


def _model_json(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value.model_dump(mode="json", by_alias=True))


def _guest_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _guest_token(existing: str | None) -> str:
    if existing and len(existing) >= 40:
        return existing
    return secrets.token_urlsafe(32)


def _upload_request_sha256(
    *,
    payload: bytes,
    filename: str,
    media_type: str,
    sheet_name: str | None,
    header_row: int,
) -> str:
    document = {
        "apiContractVersion": "api-v1",
        "filename": filename,
        "mediaType": media_type,
        "sheetName": sheet_name,
        "headerRow": header_row,
        "payloadSha256": hashlib.sha256(payload).hexdigest(),
    }
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _set_guest_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        key=GUEST_COOKIE,
        value=token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def _job_from_row(row: dict[str, Any]) -> ProcessingJob:
    return ProcessingJob(
        id=row["id"],
        project_id=row["project_id"],
        stage=row["stage"],
        progress=row["progress"],
        message=row["message"],
        error_code=row["error_code"],
    )


def _project_session(repository: ProjectReader, project: dict[str, Any]) -> ProjectSession:
    job_row = repository.get_job_for_project(project["id"])
    return ProjectSession(
        project_id=project["id"],
        storage_mode=project["storage_mode"],
        description=project.get("description", ""),
        current_revision_id=project.get("current_revision_id"),
        source=SourceFile.model_validate_json(project["source_json"]),
        job=_job_from_row(job_row) if job_row else None,
        expires_at=project["expires_at"],
    )


def _require_project(repository: ProjectReader, project_id: str) -> dict[str, Any]:
    project = repository.get_project(project_id)
    if project is None:
        raise ApiProblem(404, "project-not-found", "This project does not exist or has expired.")
    return project


def _require_ready_project(repository: ProjectReader, project_id: str) -> dict[str, Any]:
    project = _require_project(repository, project_id)
    job = repository.get_job_for_project(project_id)
    if job and job["stage"] == "failed":
        raise ApiProblem(
            422,
            job["error_code"] or "processing-failed",
            job["message"],
        )
    ready = project.get("ready")
    if ready is None:
        ready = bool(project["preview_json"] and project["quality_json"] and project["data_blob"])
    if not ready:
        raise ApiProblem(409, "processing-not-ready", "The uploaded file is still being processed.")
    return project


def _require_project_access(
    repository: ProjectReader,
    project_id: str,
    user: dict[str, str] | None,
    guest_token: str | None,
    *,
    ready: bool = False,
) -> dict[str, Any]:
    project = (
        _require_ready_project(repository, project_id)
        if ready
        else _require_project(repository, project_id)
    )
    if project["storage_mode"] != "saved-cloud":
        stored_digest = project.get("guest_token_digest")
        if (
            not guest_token
            or not stored_digest
            or not secrets.compare_digest(stored_digest, _guest_digest(guest_token))
        ):
            raise ApiProblem(
                403,
                "project-access-denied",
                "This temporary project belongs to another browser session.",
            )
        return project
    if user is None:
        raise ApiProblem(401, "authentication-required", "Sign in to open this saved project.")
    if project["owner_user_id"] != user["id"]:
        raise ApiProblem(403, "project-access-denied", "This project belongs to another user.")
    return project


def _export_response(export: dict[str, Any]) -> Response:
    formats = {
        "png": ("image/png", "png"),
        "svg": ("image/svg+xml", "svg"),
        "pdf": ("application/pdf", "pdf"),
    }
    media_type, extension = formats[export["format"]]
    return Response(
        content=bytes(export["payload"]),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="labviz-{export["id"]}.{extension}"'
        },
    )


def _shared_preview(preview: DataPreview, chart: ChartSpec) -> DataPreview:
    fields = {chart.x_axis.field, chart.y_axis.field}
    fields.update(item.field for item in chart.series)
    if chart.uncertainty.error_field:
        fields.add(chart.uncertainty.error_field)
    if chart.secondary_y_axis.field:
        fields.add(chart.secondary_y_axis.field)
    columns = [column for column in preview.columns if column.field in fields]
    rows = [
        {key: value for key, value in row.items() if key == "rowId" or key in fields}
        for row in preview.rows
    ]
    return DataPreview(
        project_id=preview.project_id,
        columns=columns,
        rows=rows,
        total_rows=preview.total_rows,
        sampled=preview.sampled,
        sample_strategy=preview.sample_strategy,
    )


def _process_project(
    repository: ProjectStore,
    settings: Settings,
    project_id: str,
    job_id: str,
    payload: bytes,
    filename: str,
    media_type: str,
    requested_sheet_name: str | None = None,
    header_row: int = 1,
) -> None:
    try:
        repository.update_job(
            job_id,
            stage="uploading",
            progress=15,
            message="The upload is complete. Preparing the file.",
        )
        validate_upload(payload, filename, settings.max_upload_bytes)
        repository.update_job(
            job_id,
            stage="parsing",
            progress=45,
            message="Reading the table and identifying columns.",
        )
        frame, sheet_name, available_sheets, resolved_header_row = load_dataframe(
            payload,
            filename,
            requested_sheet_name=requested_sheet_name,
            header_row=header_row,
        )
        repository.update_job(
            job_id,
            stage="profiling",
            progress=75,
            message="Checking data quality and preparing a safe preview.",
        )
        source = SourceFile(
            name=filename,
            size=len(payload),
            media_type=media_type,
            sheet_name=sheet_name,
            available_sheets=available_sheets,
            header_row=resolved_header_row,
        )
        preview = build_preview(project_id, frame)
        quality = build_quality_report(project_id, frame)
        chart = default_chart_spec(frame)
        repository.complete_project(
            project_id=project_id,
            source=_model_json(source),
            frame=frame,
            preview=preview,
            quality=quality,
            chart=chart,
        )
        repository.update_job(
            job_id,
            stage="ready",
            progress=100,
            message="Your data is ready to inspect.",
        )
    except ProcessingError as exc:
        repository.update_job(
            job_id,
            stage="failed",
            progress=100,
            message=str(exc),
            error_code=exc.code,
        )
    except Exception:
        LOGGER.exception("Unexpected processing failure for project %s", project_id)
        repository.update_job(
            job_id,
            stage="failed",
            progress=100,
            message="LabViz could not process this file safely.",
            error_code="processing-failed",
        )


def create_app(
    settings: Settings | None = None,
    repository: ProjectRepository | None = None,
    auth_service: AuthService | None = None,
    project_store: ProjectStore | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_repository = repository or ProjectRepository(
        resolved_settings.database_path, resolved_settings.project_ttl_seconds
    )
    resolved_project_store = project_store or build_project_store(
        resolved_settings, resolved_repository
    )
    resolved_auth_repository = cast(
        AuthRepository,
        (
            resolved_project_store
            if resolved_settings.persistence_backend == "postgresql"
            else resolved_repository
        ),
    )
    resolved_auth = auth_service or build_auth_service(resolved_settings, resolved_auth_repository)
    client_identity = ClientIdentityResolver(resolved_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        if resolved_settings.persistence_backend == "postgresql":
            # PostgreSQL maintenance is owned by the independently leased worker CLI.
            # FastAPI replicas must not race an unfenced lifespan cleanup loop.
            yield
            return
        resolved_project_store.recover_stale_jobs()

        async def maintain_temporary_data() -> None:
            while True:
                await asyncio.sleep(60)
                await asyncio.to_thread(resolved_project_store.cleanup_expired)
                await asyncio.to_thread(resolved_project_store.recover_stale_jobs)

        maintenance = asyncio.create_task(maintain_temporary_data())
        try:
            yield
        finally:
            maintenance.cancel()
            with suppress(asyncio.CancelledError):
                await maintenance

    app = FastAPI(
        title="LabViz API",
        version="2.0.0-alpha.1",
        description="Processing, quality review, chart export, history, sharing, and auth.",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.repository = resolved_repository
    app.state.project_store = resolved_project_store
    app.state.auth_repository = resolved_auth_repository
    app.state.auth_service = resolved_auth
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid4().hex)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = (
            "no-referrer" if request.url.path.startswith(f"{API_PREFIX}/shares/") else "same-origin"
        )
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path.startswith(API_PREFIX):
            response.headers["Cache-Control"] = "no-store"
        route = request.scope.get("route")
        route_template = getattr(route, "path", "<unmatched>")
        LOGGER.info(
            "%s %s %s %.1fms request_id=%s",
            request.method,
            route_template,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request_id,
        )
        return cast(Response, response)

    @app.exception_handler(ApiProblem)
    async def api_problem_handler(_request: Request, exc: ApiProblem) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    @app.exception_handler(AuthError)
    async def auth_error_handler(_request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": str(exc)},
        )

    @app.exception_handler(PersistenceError)
    async def persistence_error_handler(_request: Request, exc: PersistenceError) -> JSONResponse:
        if isinstance(exc, PersistenceNotFound):
            status_code = 404
        elif isinstance(exc, PersistenceConflict):
            status_code = 409
        elif isinstance(exc, PersistenceUnavailable):
            status_code = 503
        else:
            status_code = 500
        return JSONResponse(
            status_code=status_code,
            content={"code": exc.code, "message": str(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else None
        message = (
            first.get("msg", "The request is invalid.") if first else "The request is invalid."
        )
        return JSONResponse(
            status_code=422,
            content={"code": "validation-error", "message": message},
        )

    def get_repository() -> ProjectRepository:
        return resolved_repository

    def get_project_store() -> ProjectStore:
        return resolved_project_store

    def get_settings() -> Settings:
        return resolved_settings

    def get_auth() -> AuthService:
        return resolved_auth

    def get_auth_repository() -> AuthRepository:
        return resolved_auth_repository

    def optional_user(
        session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
        auth: AuthService = Depends(get_auth),
    ) -> dict[str, str] | None:
        return auth.get_user(session_token)

    def required_user(
        user: dict[str, str] | None = Depends(optional_user),
    ) -> dict[str, str]:
        if user is None:
            raise ApiProblem(401, "authentication-required", "Sign in to save and share projects.")
        return user

    @app.get("/health", response_model=HealthResponse)
    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get(f"{API_PREFIX}/ready", response_model=HealthResponse)
    def readiness(repository: ProjectStore = Depends(get_project_store)) -> HealthResponse:
        readiness_error = repository.readiness_error()
        if readiness_error == "database-unavailable":
            raise ApiProblem(503, "database-unavailable", "The database is not ready.")
        if readiness_error == "object-storage-unavailable":
            raise ApiProblem(
                503,
                "object-storage-unavailable",
                "The object storage provider is not ready.",
            )
        return HealthResponse()

    @app.post(
        f"{API_PREFIX}/projects",
        response_model=ProjectSession,
        status_code=202,
    )
    async def create_project(
        background_tasks: BackgroundTasks,
        response: Response,
        file: UploadFile = File(...),
        sheet_name: str | None = Form(default=None, alias="sheetName"),
        header_row: int = Form(default=1, alias="headerRow", ge=1, le=1_000),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        guest_cookie: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
        current_settings: Settings = Depends(get_settings),
    ) -> ProjectSession:
        filename = Path(file.filename or "uploaded-data").name
        payload = await file.read(current_settings.max_upload_bytes + 1)
        await file.close()
        try:
            validate_upload(payload, filename, current_settings.max_upload_bytes)
        except ProcessingError as exc:
            raise ApiProblem(422, exc.code, str(exc)) from exc

        project_id = uuid4().hex
        job_id = uuid4().hex
        media_type = file.content_type or "application/octet-stream"
        request_sha256 = (
            _upload_request_sha256(
                payload=payload,
                filename=filename,
                media_type=media_type,
                sheet_name=sheet_name,
                header_row=header_row,
            )
            if idempotency_key is not None
            else None
        )
        guest_token = _guest_token(guest_cookie)
        _set_guest_cookie(response, guest_token, current_settings)
        source = SourceFile(name=filename, size=len(payload), media_type=media_type)
        creation = repository.create_project(
            project_id=project_id,
            job_id=job_id,
            title=Path(filename).stem or "Untitled project",
            source=_model_json(source),
            source_sha256=hashlib.sha256(payload).hexdigest(),
            guest_token_digest=_guest_digest(guest_token),
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
        )
        if not creation.replayed:
            background_tasks.add_task(
                _process_project,
                repository,
                current_settings,
                creation.project_id,
                creation.job_id,
                payload,
                filename,
                media_type,
                sheet_name,
                header_row,
            )
        project = _require_project(repository, creation.project_id)
        return _project_session(repository, project)

    @app.post(
        f"{API_PREFIX}/samples/thermal-response/projects",
        response_model=ProjectSession,
        status_code=202,
    )
    def create_sample_project(
        background_tasks: BackgroundTasks,
        response: Response,
        guest_cookie: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
        current_settings: Settings = Depends(get_settings),
    ) -> ProjectSession:
        payload = sample_csv_bytes()
        filename = "thermal-response-sample.csv"
        project_id = uuid4().hex
        job_id = uuid4().hex
        media_type = "text/csv"
        guest_token = _guest_token(guest_cookie)
        _set_guest_cookie(response, guest_token, current_settings)
        source = SourceFile(name=filename, size=len(payload), media_type=media_type)
        repository.create_project(
            project_id=project_id,
            job_id=job_id,
            title="Thermal response sample",
            source=_model_json(source),
            source_sha256=hashlib.sha256(payload).hexdigest(),
            guest_token_digest=_guest_digest(guest_token),
        )
        background_tasks.add_task(
            _process_project,
            repository,
            current_settings,
            project_id,
            job_id,
            payload,
            filename,
            media_type,
            None,
            1,
        )
        project = _require_project(repository, project_id)
        return _project_session(repository, project)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}", response_model=ProjectSession)
    def get_project(
        project_id: str,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ProjectSession:
        project = _require_project_access(repository, project_id, user, guest_token)
        return _project_session(repository, project)

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}", response_model=ProcessingJob)
    def get_job(
        job_id: str,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ProcessingJob:
        row = repository.get_job(job_id)
        if row is None:
            raise ApiProblem(404, "job-not-found", "This processing job does not exist or expired.")
        _require_project_access(repository, row["project_id"], user, guest_token)
        return _job_from_row(row)

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/save", response_model=ProjectSession)
    def save_project_to_cloud(
        project_id: str,
        user: dict[str, str] = Depends(required_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ProjectSession:
        project = _require_project_access(repository, project_id, user, guest_token, ready=True)
        if project["owner_user_id"] and project["owner_user_id"] != user["id"]:
            raise ApiProblem(403, "project-access-denied", "This project belongs to another user.")
        repository.save_project(
            project_id,
            user["id"],
            guest_token_digest=_guest_digest(guest_token) if guest_token else None,
        )
        saved = _require_project_access(repository, project_id, user, guest_token, ready=True)
        return _project_session(repository, saved)

    @app.patch(
        f"{API_PREFIX}/projects/{{project_id}}/description",
        response_model=ProjectDescriptionResponse,
    )
    def update_project_description(
        project_id: str,
        body: UpdateProjectDescriptionRequest,
        user: dict[str, str] = Depends(required_user),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ProjectDescriptionResponse:
        project = _require_project_access(repository, project_id, user, None, ready=True)
        if project["storage_mode"] != "saved-cloud":
            raise ApiProblem(
                409,
                "project-must-be-saved",
                "Save this project before editing its description.",
            )
        updated = repository.update_project_description(
            project_id=project_id,
            owner_user_id=user["id"],
            description=body.description,
            expected_revision_id=body.expected_revision_id.hex,
        )
        return ProjectDescriptionResponse.model_validate(updated)

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/duplicate",
        response_model=ProjectSession,
    )
    def duplicate_project(
        project_id: str,
        user: dict[str, str] = Depends(required_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ProjectSession:
        _require_project_access(repository, project_id, user, guest_token, ready=True)
        copy_id = uuid4().hex
        copy_id = repository.duplicate_project(
            source_project_id=project_id,
            project_id=copy_id,
            job_id=uuid4().hex,
            owner_user_id=user["id"],
            guest_token_digest=_guest_digest(guest_token) if guest_token else None,
            idempotency_key=idempotency_key,
        )
        return _project_session(
            repository,
            _require_project_access(repository, copy_id, user, guest_token, ready=True),
        )

    @app.delete(f"{API_PREFIX}/projects/{{project_id}}", status_code=204)
    def delete_project(
        project_id: str,
        user: dict[str, str] = Depends(required_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> Response:
        _require_project_access(repository, project_id, user, guest_token)
        repository.delete_project(
            project_id,
            owner_user_id=user["id"],
            guest_token_digest=_guest_digest(guest_token) if guest_token else None,
        )
        return Response(status_code=204)

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/restore",
        response_model=ProjectSession,
    )
    def restore_deleted_project(
        project_id: str,
        user: dict[str, str] = Depends(required_user),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ProjectSession:
        repository.restore_deleted_project(project_id, user["id"])
        restored = _require_project_access(repository, project_id, user, None, ready=True)
        return _project_session(repository, restored)

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/revisions/{{revision_number}}/restore",
        response_model=ProjectSession,
    )
    def restore_project_revision(
        project_id: str,
        revision_number: int,
        user: dict[str, str] = Depends(required_user),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ProjectSession:
        repository.restore_project_revision(project_id, revision_number, user["id"])
        restored = _require_project_access(repository, project_id, user, None, ready=True)
        return _project_session(repository, restored)

    @app.get(
        f"{API_PREFIX}/projects/{{project_id}}/workspace",
        response_model=ProjectWorkspace,
    )
    def get_project_workspace(
        project_id: str,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
        current_settings: Settings = Depends(get_settings),
    ) -> ProjectWorkspace:
        project = _require_project_access(repository, project_id, user, guest_token, ready=True)
        workspace = repository.get_workspace(project_id)
        shares = [
            ShareSummary(
                token=row["token"],
                url=f"{current_settings.public_web_url}/share/{row['token']}",
                downloads_enabled=bool(row["downloads_enabled"]),
                created_at=row["created_at"],
            )
            for row in workspace["shares"]
        ]
        return ProjectWorkspace(
            session=_project_session(repository, project),
            preview=DataPreview.model_validate(workspace["preview"]),
            quality=QualityReport.model_validate(workspace["quality"]),
            decisions=[CleaningDecision.model_validate(item) for item in workspace["decisions"]],
            chart=ChartSpec.model_validate(workspace["chart"]),
            shares=shares,
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/preview", response_model=DataPreview)
    def get_preview(
        project_id: str,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> DataPreview:
        project = _require_project_access(repository, project_id, user, guest_token, ready=True)
        return DataPreview.model_validate_json(project["preview_json"])

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/quality", response_model=QualityReport)
    def get_quality(
        project_id: str,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> QualityReport:
        project = _require_project_access(repository, project_id, user, guest_token, ready=True)
        return QualityReport.model_validate_json(project["quality_json"])

    @app.put(
        f"{API_PREFIX}/projects/{{project_id}}/quality-rules",
        response_model=QualityReport,
    )
    def apply_quality_rules(
        project_id: str,
        body: QualityRulesRequest,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> QualityReport:
        _require_project_access(repository, project_id, user, guest_token, ready=True)
        frame = repository.load_quality_dataframe(project_id)
        ranges = {rule.field: (rule.minimum, rule.maximum) for rule in body.ranges}
        if len(ranges) != len(body.ranges):
            raise ApiProblem(
                422,
                "duplicate-quality-rule",
                "Each column can have only one valid range.",
            )
        try:
            quality = build_quality_report(project_id, frame, ranges)
        except ProcessingError as exc:
            raise ApiProblem(422, exc.code, str(exc)) from exc
        repository.save_quality_report(
            project_id,
            quality,
            parameters={"validRanges": [_model_json(rule) for rule in body.ranges]},
        )
        return QualityReport.model_validate(quality)

    @app.patch(
        f"{API_PREFIX}/projects/{{project_id}}/cleaning-decisions",
        response_model=CleaningDecisionsResponse,
    )
    def save_cleaning_decisions(
        project_id: str,
        body: CleaningDecisionsRequest,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> CleaningDecisionsResponse:
        project = _require_project_access(repository, project_id, user, guest_token, ready=True)
        quality = QualityReport.model_validate_json(project["quality_json"])
        finding_ids = {item.id for item in quality.findings}
        unknown = [item.finding_id for item in body.decisions if item.finding_id not in finding_ids]
        if unknown:
            raise ApiProblem(
                422,
                "unknown-quality-finding",
                "One or more cleaning decisions refer to findings that do not exist.",
            )
        decisions = [_model_json(item) for item in body.decisions]
        updated_at = repository.save_decisions(project_id, decisions)
        return CleaningDecisionsResponse(
            project_id=project_id,
            decisions=[
                CleaningDecision.model_validate(item)
                for item in repository.get_decisions(project_id)
            ],
            updated_at=updated_at,
        )

    @app.put(
        f"{API_PREFIX}/projects/{{project_id}}/chart",
        response_model=SavedChartResponse,
    )
    def save_chart(
        project_id: str,
        body: ChartRequest,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> SavedChartResponse:
        project = _require_project_access(repository, project_id, user, guest_token, ready=True)
        preview = DataPreview.model_validate_json(project["preview_json"])
        available_fields = {item.field for item in preview.columns}
        selected_fields = {body.chart.x_axis.field, body.chart.y_axis.field}
        selected_fields.update(item.field for item in body.chart.series)
        if body.chart.uncertainty.error_field:
            selected_fields.add(body.chart.uncertainty.error_field)
        if body.chart.secondary_y_axis.field:
            selected_fields.add(body.chart.secondary_y_axis.field)
        if not selected_fields.issubset(available_fields):
            raise ApiProblem(
                422,
                "invalid-chart-fields",
                "The chart refers to one or more columns that are not in this project.",
            )
        column_kinds = {item.field: item.kind for item in preview.columns}
        numeric_fields = {item.field for item in body.chart.series}
        if (
            body.chart.type == "surface3d"
            or body.chart.fitting.model != "none"
            or body.chart.uncertainty.mode != "none"
        ):
            numeric_fields.add(body.chart.x_axis.field)
        if body.chart.uncertainty.error_field:
            numeric_fields.add(body.chart.uncertainty.error_field)
        if any(column_kinds.get(field) != "number" for field in numeric_fields):
            raise ApiProblem(
                422,
                "invalid-chart-fields",
                "The selected chart or analysis requires numeric columns.",
            )
        updated_at = repository.save_chart(project_id, _model_json(body.chart))
        return SavedChartResponse(
            project_id=project_id,
            chart=body.chart,
            updated_at=updated_at,
        )

    @app.get(f"{API_PREFIX}/projects", response_model=ProjectList)
    def list_projects(
        user: dict[str, str] | None = Depends(optional_user),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ProjectList:
        projects = []
        for row in repository.list_projects(user["id"] if user else None):
            chart = ChartSpec.model_validate_json(row["chart_json"])
            source = SourceFile.model_validate_json(row["source_json"])
            projects.append(
                ProjectSummary(
                    id=row["id"],
                    title=row["title"],
                    source_name=source.name,
                    chart_type=chart.type,
                    updated_at=row["updated_at"],
                    storage_mode=row["storage_mode"],
                    thumbnail_url=None,
                )
            )
        return ProjectList(projects=projects)

    @app.get(f"{API_PREFIX}/recovery/projects", response_model=ProjectList)
    def list_deleted_projects(
        user: dict[str, str] = Depends(required_user),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ProjectList:
        projects = []
        for row in repository.list_deleted_projects(user["id"]):
            chart = ChartSpec.model_validate_json(row["chart_json"])
            source = SourceFile.model_validate_json(row["source_json"])
            projects.append(
                ProjectSummary(
                    id=row["id"],
                    title=row["title"],
                    source_name=source.name,
                    chart_type=chart.type,
                    updated_at=row["updated_at"],
                    storage_mode=row["storage_mode"],
                    thumbnail_url=None,
                )
            )
        return ProjectList(projects=projects)

    @app.post(f"{API_PREFIX}/auth/email-code", response_model=RequestEmailCodeResponse)
    def request_email_code(
        body: RequestEmailCode,
        request: Request,
        auth: AuthService = Depends(get_auth),
        repository: AuthRepository = Depends(get_auth_repository),
    ) -> RequestEmailCodeResponse:
        email = str(body.email).strip().lower()
        try:
            client_key = client_identity.resolve_request(request)
        except ClientIdentityError as exc:
            raise ApiProblem(
                400,
                "invalid-client-identity",
                "The request network path could not be verified.",
            ) from exc
        try:
            allowed = repository.allow_auth_request(
                client_key=client_key,
                email=email,
                client_limit=resolved_settings.auth_client_request_limit,
                email_limit=resolved_settings.auth_email_request_limit,
                window_seconds=resolved_settings.auth_rate_limit_window_seconds,
            )
        except AuthRateLimitUnavailable as exc:
            raise ApiProblem(503, exc.code, str(exc)) from exc
        if not allowed:
            LOGGER.warning("authentication-request-rate-limited")
            raise ApiProblem(
                429,
                "email-rate-limited",
                "Too many verification requests. Try again later.",
            )
        try:
            challenge_id, expires_seconds, resend_seconds = auth.request_code(email)
        except AuthError:
            raise
        except Exception as exc:
            LOGGER.warning(
                "authentication-code-delivery-failed error_type=%s",
                type(exc).__name__,
            )
            raise ApiProblem(
                503,
                "email-delivery-failed",
                "The verification email could not be sent. Try again later.",
            ) from exc
        return RequestEmailCodeResponse(
            challenge_id=challenge_id,
            expires_in_seconds=expires_seconds,
            resend_after_seconds=resend_seconds,
            delivery_mode=auth.delivery_mode,
        )

    @app.post(
        f"{API_PREFIX}/auth/email-code/verify",
        response_model=VerifyEmailCodeResponse,
    )
    def verify_email_code(
        body: VerifyEmailCode,
        response: Response,
        auth: AuthService = Depends(get_auth),
        current_settings: Settings = Depends(get_settings),
    ) -> VerifyEmailCodeResponse:
        user, token = auth.verify_code(body.challenge_id, body.code)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            max_age=current_settings.session_ttl_seconds,
            httponly=True,
            secure=current_settings.cookie_secure,
            samesite="lax",
            path="/",
        )
        return VerifyEmailCodeResponse(
            authenticated=True, user=AuthenticatedUser.model_validate(user)
        )

    @app.get(f"{API_PREFIX}/auth/me", response_model=AuthState)
    def get_current_user(
        user: dict[str, str] | None = Depends(optional_user),
    ) -> AuthState:
        return AuthState(
            authenticated=user is not None,
            user=AuthenticatedUser.model_validate(user) if user else None,
        )

    @app.post(f"{API_PREFIX}/auth/logout", response_model=AuthState)
    def logout(
        response: Response,
        session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
        auth: AuthService = Depends(get_auth),
    ) -> AuthState:
        auth.logout(session_token)
        response.delete_cookie(
            key=SESSION_COOKIE,
            path="/",
            secure=resolved_settings.cookie_secure,
            httponly=True,
            samesite="lax",
        )
        return AuthState(authenticated=False, user=None)

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/shares", response_model=ShareLink)
    def create_share(
        project_id: str,
        body: CreateShareRequest,
        user: dict[str, str] = Depends(required_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
        current_settings: Settings = Depends(get_settings),
    ) -> ShareLink:
        project = _require_project_access(repository, project_id, user, guest_token, ready=True)
        if project["storage_mode"] != "saved-cloud":
            repository.save_project(
                project_id,
                user["id"],
                guest_token_digest=_guest_digest(guest_token) if guest_token else None,
            )
        share = repository.create_share(
            project_id=project_id,
            owner_user_id=user["id"],
            downloads_enabled=body.downloads_enabled,
        )
        return ShareLink(
            token=share["token"],
            url=f"{current_settings.public_web_url}/share/{share['token']}",
            downloads_enabled=share["downloads_enabled"],
            created_at=share["created_at"],
        )

    @app.patch(
        f"{API_PREFIX}/projects/{{project_id}}/shares/{{token}}",
        response_model=ShareLink,
    )
    def update_share(
        project_id: str,
        token: str,
        body: UpdateShareRequest,
        user: dict[str, str] = Depends(required_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
        current_settings: Settings = Depends(get_settings),
    ) -> ShareLink:
        _require_project_access(repository, project_id, user, guest_token, ready=True)
        share = repository.update_share(
            token=token,
            project_id=project_id,
            owner_user_id=user["id"],
            downloads_enabled=body.downloads_enabled,
        )
        if share is None:
            raise ApiProblem(404, "share-not-found", "This shared chart is unavailable.")
        return ShareLink(
            token=share["token"],
            url=f"{current_settings.public_web_url}/share/{share['token']}",
            downloads_enabled=share["downloads_enabled"],
            created_at=share["created_at"],
        )

    @app.delete(
        f"{API_PREFIX}/projects/{{project_id}}/shares/{{token}}",
        status_code=204,
    )
    def revoke_share(
        project_id: str,
        token: str,
        user: dict[str, str] = Depends(required_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> Response:
        _require_project_access(repository, project_id, user, guest_token, ready=True)
        if not repository.revoke_share(
            token=token,
            project_id=project_id,
            owner_user_id=user["id"],
        ):
            raise ApiProblem(404, "share-not-found", "This shared chart is unavailable.")
        return Response(status_code=204)

    @app.get(f"{API_PREFIX}/shares/{{token}}", response_model=SharedChart)
    def get_shared_chart(
        token: str,
        response: Response,
        repository: ProjectStore = Depends(get_project_store),
    ) -> SharedChart:
        shared = repository.get_shared_project(token)
        if shared is None:
            raise ApiProblem(404, "share-not-found", "This shared chart is unavailable.")
        context, frame = shared
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        downloads = SharedDownloads()
        if bool(context["downloads_enabled"]):
            formats = set(context["download_formats"])
            downloads = SharedDownloads(
                png=(f"{API_PREFIX}/shares/{token}/downloads/png" if "png" in formats else None),
                svg=(f"{API_PREFIX}/shares/{token}/downloads/svg" if "svg" in formats else None),
                pdf=(f"{API_PREFIX}/shares/{token}/downloads/pdf" if "pdf" in formats else None),
            )
        chart = ChartSpec.model_validate(context["chart"])
        derived = analyze_chart(frame, chart)
        preview = DataPreview.model_validate(build_preview(context["project_id"], frame))
        return SharedChart(
            token=token,
            title=context["title"],
            description=context["description"],
            updated_at=context["updated_at"],
            chart=chart,
            preview=_shared_preview(preview, chart),
            analysis=ChartAnalysis(
                project_id=context["project_id"],
                series=derived["series"],
                preview=derived["preview"],
            ),
            downloads=downloads,
        )

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/exports",
        response_model=ExportJob,
    )
    def create_export(
        project_id: str,
        body: ChartRequest,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ExportJob:
        project = _require_project_access(repository, project_id, user, guest_token, ready=True)
        try:
            frame = repository.load_chart_dataframe(project_id)
            payload = render_chart(frame, body.chart)
        except (ProcessingError, ValidationError) as exc:
            code = exc.code if isinstance(exc, ProcessingError) else "invalid-chart"
            raise ApiProblem(422, code, str(exc)) from exc

        export = repository.create_publication_export(
            project_id=project_id,
            expected_revision_id=project.get("current_revision_id"),
            chart=_model_json(body.chart),
            payload=payload,
            owner_user_id=user["id"] if user else None,
            guest_token_digest=_guest_digest(guest_token) if guest_token else None,
            idempotency_key=idempotency_key,
        )
        return ExportJob(
            id=export["id"],
            project_id=project_id,
            status=export["status"],
            download_url=(
                f"{API_PREFIX}/exports/{export['id']}/download"
                if export["status"] == "ready"
                else None
            ),
            expires_at=export["expires_at"],
            message=export["message"],
        )

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/chart-analysis",
        response_model=ChartAnalysis,
    )
    def get_chart_analysis(
        project_id: str,
        body: ChartRequest,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> ChartAnalysis:
        _require_project_access(repository, project_id, user, guest_token, ready=True)
        try:
            frame = repository.load_chart_dataframe(project_id)
            analysis = analyze_chart(frame, body.chart)
        except (ProcessingError, ValidationError) as exc:
            code = exc.code if isinstance(exc, ProcessingError) else "invalid-chart"
            raise ApiProblem(422, code, str(exc)) from exc
        return ChartAnalysis(
            project_id=project_id,
            series=analysis["series"],
            preview=analysis["preview"],
        )

    @app.get(f"{API_PREFIX}/exports/{{export_id}}/download")
    def download_export(
        export_id: str,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> Response:
        metadata = repository.get_export_metadata(export_id)
        if metadata is None:
            raise ApiProblem(404, "export-not-found", "This export does not exist or has expired.")
        _require_project_access(repository, metadata["project_id"], user, guest_token)
        export = repository.get_export(export_id)
        if export is None:
            raise ApiProblem(404, "export-not-found", "This export does not exist or has expired.")
        return _export_response(export)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/exports/cleaned-data.csv")
    def download_cleaned_data(
        project_id: str,
        user: dict[str, str] | None = Depends(optional_user),
        guest_token: str | None = Cookie(default=None, alias=GUEST_COOKIE),
        repository: ProjectStore = Depends(get_project_store),
    ) -> Response:
        project = _require_project_access(repository, project_id, user, guest_token, ready=True)
        cleaned = repository.load_cleaned_dataframe(project_id)
        payload = cleaned.to_csv(index=False).encode("utf-8-sig")
        safe_name = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in str(project["title"])
        ).strip("-")
        filename = f"{safe_name or 'labviz'}-cleaned.csv"
        return Response(
            content=payload,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get(f"{API_PREFIX}/shares/{{token}}/downloads/{{format_name}}")
    def download_shared_export(
        token: str,
        format_name: str,
        repository: ProjectStore = Depends(get_project_store),
    ) -> Response:
        if format_name not in {"png", "svg", "pdf"}:
            raise ApiProblem(404, "export-not-found", "This shared export is unavailable.")
        shared_export = repository.get_shared_export(token, format_name)
        if shared_export is None:
            raise ApiProblem(404, "share-not-found", "This shared chart is unavailable.")
        if not bool(shared_export["downloads_enabled"]):
            raise ApiProblem(
                403,
                "share-download-disabled",
                "Downloads are disabled by the creator.",
            )
        export = shared_export["export"]
        if export is None:
            raise ApiProblem(404, "export-not-found", "This shared export is unavailable.")
        result = _export_response(export)
        result.headers["Referrer-Policy"] = "no-referrer"
        result.headers["Cache-Control"] = "no-store"
        return result

    return app


app = create_app()
