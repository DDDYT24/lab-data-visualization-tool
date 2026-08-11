"""SQLite persistence for processed LabViz projects and derived artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from labviz_api.persistence_types import ProjectCreation
from labviz_api.rate_limits import AuthRateLimitUnavailable, auth_bucket_specs, fixed_window


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_at(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def iso_now() -> str:
    return iso_at(utc_now())


def expires_in(seconds: int) -> str:
    return iso_at(utc_now() + timedelta(seconds=seconds))


class ProjectRepository:
    """Small SQLite repository suitable for the local V2 development service."""

    def __init__(self, database_path: Path, project_ttl_seconds: int) -> None:
        self.database_path = database_path
        self.project_ttl_seconds = project_ttl_seconds
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    storage_mode TEXT NOT NULL,
                    owner_user_id TEXT,
                    guest_token_digest TEXT,
                    expires_at TEXT,
                    updated_at TEXT NOT NULL,
                    chart_json TEXT,
                    preview_json TEXT,
                    quality_json TEXT,
                    data_blob BLOB
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    message TEXT NOT NULL,
                    error_code TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    finding_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, finding_id)
                );

                CREATE TABLE IF NOT EXISTS shares (
                    token TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    downloads_enabled INTEGER NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS exports (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    format TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload BLOB,
                    expires_at TEXT,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_challenges (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    code_digest TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    resend_at TEXT NOT NULL,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_digest TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    email TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_key TEXT NOT NULL,
                    email TEXT NOT NULL,
                    requested_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_rate_limit_buckets (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL CHECK (scope IN ('client', 'email')),
                    identity_key TEXT NOT NULL,
                    window_started_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0 CHECK (request_count >= 0),
                    updated_at TEXT NOT NULL,
                    CHECK (expires_at > window_started_at),
                    CHECK (
                        scope <> 'client' OR (
                            length(identity_key) = 64
                            AND identity_key NOT GLOB '*[^0-9a-f]*'
                        )
                    ),
                    CHECK (scope <> 'email' OR identity_key = lower(identity_key)),
                    UNIQUE (scope, identity_key, window_started_at)
                );

                CREATE TABLE IF NOT EXISTS upload_idempotency (
                    guest_token_digest TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (guest_token_digest, idempotency_key)
                );

                CREATE INDEX IF NOT EXISTS idx_projects_owner
                ON projects(owner_user_id, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_exports_project
                ON exports(project_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_auth_challenges_email
                ON auth_challenges(email, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_auth_requests_client
                ON auth_requests(client_key, requested_at);

                CREATE INDEX IF NOT EXISTS idx_auth_requests_email
                ON auth_requests(email, requested_at);

                CREATE INDEX IF NOT EXISTS idx_auth_rate_limit_buckets_expires
                ON auth_rate_limit_buckets(expires_at);

                CREATE INDEX IF NOT EXISTS idx_upload_idempotency_expires
                ON upload_idempotency(expires_at);
                """
            )
            job_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "updated_at" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN updated_at TEXT")
            project_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(projects)").fetchall()
            }
            if "guest_token_digest" not in project_columns:
                connection.execute("ALTER TABLE projects ADD COLUMN guest_token_digest TEXT")
            connection.execute(
                "UPDATE jobs SET updated_at = ? WHERE updated_at IS NULL", (iso_now(),)
            )

    def cleanup_expired(self) -> None:
        now = iso_now()
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM exports WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)
            )
            connection.execute(
                """
                DELETE FROM projects
                WHERE storage_mode = 'temporary-cloud'
                  AND expires_at IS NOT NULL
                  AND expires_at < ?
                """,
                (now,),
            )
            connection.execute("DELETE FROM auth_challenges WHERE expires_at < ?", (now,))
            connection.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now,))
            connection.execute("DELETE FROM upload_idempotency WHERE expires_at <= ?", (now,))
            request_cutoff = iso_at(utc_now() - timedelta(hours=1))
            connection.execute(
                "DELETE FROM auth_requests WHERE requested_at < ?", (request_cutoff,)
            )
            connection.execute("DELETE FROM auth_rate_limit_buckets WHERE expires_at <= ?", (now,))

    def create_project(
        self,
        *,
        project_id: str,
        job_id: str,
        title: str,
        source: dict[str, Any],
        guest_token_digest: str,
        idempotency_key: str | None = None,
        request_sha256: str | None = None,
    ) -> ProjectCreation:
        now = iso_now()
        project_expires_at = expires_in(self.project_ttl_seconds)
        with self._connect() as connection:
            if idempotency_key is not None:
                if not 1 <= len(idempotency_key) <= 255:
                    raise ValueError("Idempotency-Key must contain 1 to 255 characters.")
                if request_sha256 is None:
                    raise ValueError("Idempotency-Key requires a request fingerprint.")
                existing = connection.execute(
                    """
                    SELECT request_sha256, project_id, job_id, expires_at
                    FROM upload_idempotency
                    WHERE guest_token_digest = ? AND idempotency_key = ?
                    """,
                    (guest_token_digest, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if existing["expires_at"] <= now:
                        connection.execute(
                            """
                            DELETE FROM upload_idempotency
                            WHERE guest_token_digest = ? AND idempotency_key = ?
                            """,
                            (guest_token_digest, idempotency_key),
                        )
                    else:
                        project = connection.execute(
                            "SELECT id FROM projects WHERE id = ?",
                            (existing["project_id"],),
                        ).fetchone()
                        job = connection.execute(
                            "SELECT id FROM jobs WHERE id = ? AND project_id = ?",
                            (existing["job_id"], existing["project_id"]),
                        ).fetchone()
                        if project is None or job is None:
                            connection.execute(
                                """
                                DELETE FROM upload_idempotency
                                WHERE guest_token_digest = ? AND idempotency_key = ?
                                """,
                                (guest_token_digest, idempotency_key),
                            )
                        else:
                            if existing["request_sha256"] != request_sha256:
                                raise ValueError(
                                    "idempotency-key-reused: Idempotency-Key was already used "
                                    "with a different upload."
                                )
                            return ProjectCreation(
                                project_id=str(existing["project_id"]),
                                job_id=str(existing["job_id"]),
                                replayed=True,
                            )
            connection.execute(
                """
                INSERT INTO projects (
                    id, title, source_json, storage_mode, guest_token_digest,
                    expires_at, updated_at
                ) VALUES (?, ?, ?, 'temporary-cloud', ?, ?, ?)
                """,
                (
                    project_id,
                    title,
                    json.dumps(source, ensure_ascii=False),
                    guest_token_digest,
                    project_expires_at,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO jobs (
                    id, project_id, stage, progress, message, error_code, updated_at
                ) VALUES (?, ?, 'queued', 0, 'Waiting to process the uploaded file.', NULL, ?)
                """,
                (job_id, project_id, now),
            )
            if idempotency_key is not None:
                assert request_sha256 is not None
                connection.execute(
                    """
                    INSERT INTO upload_idempotency (
                        guest_token_digest, idempotency_key, request_sha256,
                        project_id, job_id, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        guest_token_digest,
                        idempotency_key,
                        request_sha256,
                        project_id,
                        job_id,
                        now,
                        project_expires_at,
                    ),
                )
            return ProjectCreation(project_id=project_id, job_id=job_id, replayed=False)

    def update_job(
        self,
        job_id: str,
        *,
        stage: str,
        progress: float,
        message: str,
        error_code: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET stage = ?, progress = ?, message = ?, error_code = ?, updated_at = ?
                WHERE id = ?
                """,
                (stage, progress, message, error_code, iso_now(), job_id),
            )

    def recover_stale_jobs(self, stale_after_seconds: int = 900) -> int:
        cutoff = iso_at(utc_now() - timedelta(seconds=stale_after_seconds))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET stage = 'failed', progress = 100,
                    message = 'Processing was interrupted. Upload the file again.',
                    error_code = 'processing-interrupted', updated_at = ?
                WHERE stage IN ('queued', 'uploading', 'parsing', 'profiling')
                  AND updated_at < ?
                """,
                (iso_now(), cutoff),
            )
        return cursor.rowcount

    def ping(self) -> bool:
        try:
            with self._connect() as connection:
                return bool(connection.execute("SELECT 1").fetchone()[0] == 1)
        except sqlite3.Error:
            return False

    def allow_auth_request(
        self,
        *,
        client_key: str,
        email: str,
        client_limit: int = 30,
        email_limit: int = 10,
        window_seconds: int = 3_600,
    ) -> bool:
        specs = auth_bucket_specs(
            client_key=client_key,
            email=email,
            client_limit=client_limit,
            email_limit=email_limit,
            window_seconds=window_seconds,
        )
        now_value = utc_now()
        window_started_at, expires_at = fixed_window(now_value, window_seconds)
        now = iso_at(now_value)
        window_start = iso_at(window_started_at)
        window_end = iso_at(expires_at)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                for scope, identity_key, _limit in specs:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO auth_rate_limit_buckets (
                            id, scope, identity_key, window_started_at,
                            expires_at, request_count, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 0, ?)
                        """,
                        (uuid4().hex, scope, identity_key, window_start, window_end, now),
                    )
                buckets = [
                    connection.execute(
                        """
                        SELECT request_count FROM auth_rate_limit_buckets
                        WHERE scope = ? AND identity_key = ? AND window_started_at = ?
                        """,
                        (scope, identity_key, window_start),
                    ).fetchone()
                    for scope, identity_key, _limit in specs
                ]
                if any(
                    bucket is None or int(bucket["request_count"]) >= limit
                    for bucket, (_scope, _identity_key, limit) in zip(buckets, specs, strict=True)
                ):
                    connection.rollback()
                    return False
                for scope, identity_key, _limit in specs:
                    connection.execute(
                        """
                        UPDATE auth_rate_limit_buckets
                        SET request_count = request_count + 1, updated_at = ?
                        WHERE scope = ? AND identity_key = ? AND window_started_at = ?
                        """,
                        (now, scope, identity_key, window_start),
                    )
                connection.execute(
                    """
                    INSERT INTO auth_requests (client_key, email, requested_at)
                    VALUES (?, ?, ?)
                    """,
                    (client_key, email, now),
                )
            return True
        except sqlite3.Error as exc:
            raise AuthRateLimitUnavailable("Authentication rate limiting is unavailable.") from exc

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def get_job_for_project(self, project_id: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE project_id = ?", (project_id,)
            ).fetchone()
        return dict(row) if row else None

    def complete_project(
        self,
        *,
        project_id: str,
        source: dict[str, Any],
        data_blob: bytes,
        preview: dict[str, Any],
        quality: dict[str, Any],
        chart: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE projects
                SET source_json = ?, data_blob = ?, preview_json = ?, quality_json = ?,
                    chart_json = ?, updated_at = ?, expires_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(source, ensure_ascii=False),
                    data_blob,
                    json.dumps(preview, ensure_ascii=False),
                    json.dumps(quality, ensure_ascii=False),
                    json.dumps(chart, ensure_ascii=False),
                    iso_now(),
                    expires_in(self.project_ttl_seconds),
                    project_id,
                ),
            )

    def _touch(self, connection: sqlite3.Connection, project_id: str) -> None:
        connection.execute(
            """
            UPDATE projects
            SET updated_at = ?,
                expires_at = CASE
                    WHEN storage_mode = 'temporary-cloud' THEN ?
                    ELSE expires_at
                END
            WHERE id = ?
            """,
            (iso_now(), expires_in(self.project_ttl_seconds), project_id),
        )

    def get_project(self, project_id: str, *, touch: bool = True) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if row and touch:
                self._touch(connection, project_id)
                row = connection.execute(
                    "SELECT * FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
        return dict(row) if row else None

    def get_project_json(self, project_id: str, column: str) -> dict[str, Any] | None:
        if column not in {"preview_json", "quality_json", "chart_json"}:
            raise ValueError("Unsupported project JSON column.")
        project = self.get_project(project_id)
        if not project or not project[column]:
            return None
        return cast(dict[str, Any], json.loads(project[column]))

    def get_data_blob(self, project_id: str) -> bytes | None:
        project = self.get_project(project_id)
        if not project or project["data_blob"] is None:
            return None
        return bytes(project["data_blob"])

    def replace_quality(self, project_id: str, quality: dict[str, Any]) -> str:
        updated_at = iso_now()
        with self._connect() as connection:
            self._touch(connection, project_id)
            connection.execute(
                "UPDATE projects SET quality_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(quality, ensure_ascii=False), updated_at, project_id),
            )
            connection.execute("DELETE FROM decisions WHERE project_id = ?", (project_id,))
            connection.execute("DELETE FROM exports WHERE project_id = ?", (project_id,))
        return updated_at

    def save_decisions(self, project_id: str, decisions: Iterable[dict[str, str]]) -> str:
        now = iso_now()
        decision_rows = [(project_id, item["findingId"], item["action"], now) for item in decisions]
        with self._connect() as connection:
            self._touch(connection, project_id)
            previous_rows = connection.execute(
                """
                SELECT finding_id, action FROM decisions
                WHERE project_id = ? ORDER BY finding_id
                """,
                (project_id,),
            ).fetchall()
            previous = [(row["finding_id"], row["action"]) for row in previous_rows]
            replacement = sorted((row[1], row[2]) for row in decision_rows)
            connection.execute("DELETE FROM decisions WHERE project_id = ?", (project_id,))
            connection.executemany(
                """
                INSERT INTO decisions (project_id, finding_id, action, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(project_id, finding_id)
                DO UPDATE SET action = excluded.action, updated_at = excluded.updated_at
                """,
                decision_rows,
            )
            if previous != replacement:
                connection.execute("DELETE FROM exports WHERE project_id = ?", (project_id,))
        return now

    def get_decisions(self, project_id: str) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT finding_id, action FROM decisions
                WHERE project_id = ? ORDER BY updated_at, finding_id
                """,
                (project_id,),
            ).fetchall()
        return [{"findingId": row["finding_id"], "action": row["action"]} for row in rows]

    def save_chart(self, project_id: str, chart: dict[str, Any]) -> str:
        now = iso_now()
        serialized = json.dumps(chart, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            self._touch(connection, project_id)
            row = connection.execute(
                "SELECT chart_json FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            previous = (
                json.dumps(json.loads(row["chart_json"]), ensure_ascii=False, sort_keys=True)
                if row and row["chart_json"]
                else None
            )
            connection.execute(
                "UPDATE projects SET chart_json = ?, title = ? WHERE id = ?",
                (
                    serialized,
                    chart["title"] or "Untitled figure",
                    project_id,
                ),
            )
            if previous != serialized:
                connection.execute("DELETE FROM exports WHERE project_id = ?", (project_id,))
        return now

    def list_projects(self, owner_user_id: str | None) -> list[dict[str, Any]]:
        self.cleanup_expired()
        if owner_user_id is None:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM projects
                WHERE owner_user_id = ? AND storage_mode = 'saved-cloud'
                ORDER BY updated_at DESC
                """,
                (owner_user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_project(self, project_id: str, owner_user_id: str) -> str:
        updated_at = iso_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE projects
                SET storage_mode = 'saved-cloud', owner_user_id = ?,
                    guest_token_digest = NULL, expires_at = NULL, updated_at = ?
                WHERE id = ? AND (owner_user_id IS NULL OR owner_user_id = ?)
                """,
                (owner_user_id, updated_at, project_id, owner_user_id),
            )
        return updated_at

    def duplicate_project(
        self,
        *,
        source_project_id: str,
        project_id: str,
        job_id: str,
        owner_user_id: str,
    ) -> None:
        now = iso_now()
        with self._connect() as connection:
            source = connection.execute(
                "SELECT * FROM projects WHERE id = ?", (source_project_id,)
            ).fetchone()
            if source is None:
                raise ValueError("Source project does not exist.")
            connection.execute(
                """
                INSERT INTO projects (
                    id, title, source_json, storage_mode, owner_user_id, expires_at,
                    updated_at, chart_json, preview_json, quality_json, data_blob
                ) VALUES (?, ?, ?, 'saved-cloud', ?, NULL, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    f"{source['title']} copy",
                    source["source_json"],
                    owner_user_id,
                    now,
                    source["chart_json"],
                    source["preview_json"],
                    source["quality_json"],
                    source["data_blob"],
                ),
            )
            connection.execute(
                """
                INSERT INTO jobs (
                    id, project_id, stage, progress, message, error_code, updated_at
                ) VALUES (?, ?, 'ready', 100, 'Copied project is ready.', NULL, ?)
                """,
                (job_id, project_id, now),
            )
            connection.execute(
                """
                INSERT INTO decisions (project_id, finding_id, action, updated_at)
                SELECT ?, finding_id, action, ? FROM decisions WHERE project_id = ?
                """,
                (project_id, now, source_project_id),
            )

    def delete_project(self, project_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount == 1

    def create_share(
        self,
        *,
        token: str,
        project_id: str,
        owner_user_id: str,
        downloads_enabled: bool,
    ) -> str:
        created_at = iso_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE projects
                SET storage_mode = 'saved-cloud', owner_user_id = ?,
                    guest_token_digest = NULL, expires_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (owner_user_id, created_at, project_id),
            )
            connection.execute(
                """
                INSERT INTO shares (token, project_id, downloads_enabled, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (token, project_id, int(downloads_enabled), created_at),
            )
        return created_at

    def get_share(self, token: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM shares WHERE token = ? AND disabled = 0", (token,)
            ).fetchone()
        return dict(row) if row else None

    def list_shares(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM shares
                WHERE project_id = ? AND disabled = 0
                ORDER BY created_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_share(self, token: str, project_id: str, downloads_enabled: bool) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE shares SET downloads_enabled = ?
                WHERE token = ? AND project_id = ? AND disabled = 0
                """,
                (int(downloads_enabled), token, project_id),
            )
        return cursor.rowcount == 1

    def revoke_share(self, token: str, project_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE shares SET disabled = 1
                WHERE token = ? AND project_id = ? AND disabled = 0
                """,
                (token, project_id),
            )
        return cursor.rowcount == 1

    def save_export(
        self,
        *,
        export_id: str,
        project_id: str,
        format_name: str,
        payload: bytes,
        expires_at: str,
        message: str,
    ) -> None:
        with self._connect() as connection:
            self._touch(connection, project_id)
            connection.execute(
                """
                INSERT INTO exports (
                    id, project_id, format, status, payload, expires_at, message, created_at
                ) VALUES (?, ?, ?, 'ready', ?, ?, ?, ?)
                """,
                (
                    export_id,
                    project_id,
                    format_name,
                    payload,
                    expires_at,
                    message,
                    iso_now(),
                ),
            )

    def get_export(self, export_id: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM exports WHERE id = ?", (export_id,)).fetchone()
        return dict(row) if row else None

    def latest_exports(self, project_id: str) -> dict[str, str]:
        self.cleanup_expired()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, format FROM exports
                WHERE project_id = ? AND status = 'ready'
                ORDER BY created_at DESC
                """,
                (project_id,),
            ).fetchall()
        latest: dict[str, str] = {}
        for row in rows:
            latest.setdefault(row["format"], row["id"])
        return latest

    def latest_auth_challenge(self, email: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM auth_challenges
                WHERE email = ? ORDER BY created_at DESC LIMIT 1
                """,
                (email,),
            ).fetchone()
        return dict(row) if row else None

    def create_auth_challenge(
        self,
        *,
        challenge_id: str,
        email: str,
        salt: str,
        code_digest: str,
        expires_at: str,
        resend_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_challenges (
                    id, email, salt, code_digest, expires_at, resend_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    challenge_id,
                    email,
                    salt,
                    code_digest,
                    expires_at,
                    resend_at,
                    iso_now(),
                ),
            )

    def get_auth_challenge(self, challenge_id: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_challenges WHERE id = ?", (challenge_id,)
            ).fetchone()
        return dict(row) if row else None

    def increment_auth_challenge_attempts(self, challenge_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE auth_challenges
                SET failed_attempts = failed_attempts + 1
                WHERE id = ?
                """,
                (challenge_id,),
            )

    def delete_auth_challenge(self, challenge_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM auth_challenges WHERE id = ?", (challenge_id,))

    def create_auth_session(
        self,
        *,
        token_digest: str,
        user_id: str,
        email: str,
        expires_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions (
                    token_digest, user_id, email, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (token_digest, user_id, email, expires_at, iso_now()),
            )

    def get_auth_session(self, token_digest: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_sessions WHERE token_digest = ?", (token_digest,)
            ).fetchone()
        return dict(row) if row else None

    def delete_auth_session(self, token_digest: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_digest = ?", (token_digest,))
