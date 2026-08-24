"""Create an encrypted PostgreSQL logical backup without logging credentials."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    raw_url = _required("LABVIZ_POSTGRES_URL").replace(
        "postgresql+psycopg://", "postgresql://", 1
    )
    parsed = urlsplit(raw_url)
    if parsed.scheme != "postgresql" or not parsed.hostname or not parsed.username:
        raise RuntimeError("LABVIZ_POSTGRES_URL must be a PostgreSQL URL")

    bucket = _required("LABVIZ_BACKUP_BUCKET")
    prefix = _required("LABVIZ_BACKUP_PREFIX").strip("/")
    key_arn = _required("LABVIZ_BACKUP_KMS_KEY_ARN")
    environment = _required("LABVIZ_ENVIRONMENT")
    timestamp = datetime.now(UTC)
    object_stem = (
        f"{prefix}/{environment}/{timestamp:%Y/%m/%d}/{timestamp:%H%M%SZ}-{uuid4().hex}"
    )

    child_env = os.environ.copy()
    child_env["PGPASSWORD"] = unquote(parsed.password or "")
    child_env["PGSSLMODE"] = "require"

    with tempfile.TemporaryDirectory(prefix="labviz-backup-") as temp_dir:
        dump_path = Path(temp_dir) / "labviz.dump"
        manifest_path = Path(temp_dir) / "labviz.dump.sha256"
        try:
            subprocess.run(
                [
                    "pg_dump",
                    "--format=custom",
                    "--no-owner",
                    "--no-acl",
                    "--host",
                    parsed.hostname,
                    "--port",
                    str(parsed.port or 5432),
                    "--username",
                    unquote(parsed.username),
                    "--file",
                    str(dump_path),
                    unquote(parsed.path.lstrip("/")),
                ],
                check=True,
                env=child_env,
            )
            digest = _sha256(dump_path)
            manifest_path.write_text(f"{digest}  labviz.dump\n", encoding="ascii")
            for path, suffix in ((dump_path, ".dump"), (manifest_path, ".dump.sha256")):
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "cp",
                        str(path),
                        f"s3://{bucket}/{object_stem}{suffix}",
                        "--only-show-errors",
                        "--sse",
                        "aws:kms",
                        "--sse-kms-key-id",
                        key_arn,
                    ],
                    check=True,
                )
        except Exception as exc:
            print(
                json.dumps(
                    {"event": "logical-backup-failed", "error_type": type(exc).__name__}
                )
            )
            raise

    print(
        json.dumps(
            {
                "event": "logical-backup-complete",
                "timestamp": timestamp.isoformat(),
                "object": f"{object_stem}.dump",
                "sha256": digest,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
