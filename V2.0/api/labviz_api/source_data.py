"""Offline complete SQLite/object snapshots; restore always creates a separate copy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_files(root: Path) -> list[Path]:
    if root.is_symlink() or root.is_junction():
        raise ValueError("Linked data directories are unsupported.")
    files: list[Path] = []

    def unreadable(error: OSError) -> None:
        raise error

    for directory, folders, names in os.walk(root, followlinks=False, onerror=unreadable):
        for name in folders + names:
            path = Path(directory) / name
            if (
                path.is_symlink()
                or path.is_junction()
                or not path.resolve().is_relative_to(root.resolve())
            ):
                raise ValueError("Linked files are unsupported.")
        files.extend(Path(directory) / name for name in names)
    return sorted(files)


def check_database(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        raise ValueError("Linked databases are unsupported.")
    if not path.is_file():
        raise ValueError("No existing SQLite database found.")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("SQLite integrity check failed.")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("SQLite foreign-key check failed.")
        tables = {
            r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "projects" not in tables:
            raise ValueError("This is not a LabViz database.")


def verify_backup(folder: Path) -> dict[str, Any]:
    members = safe_files(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Invalid backup manifest.")
    if manifest.get("format") != "labviz-v23-source-backup" or manifest.get("schemaVersion") != 1:
        raise ValueError("Unsupported backup manifest.")
    expected: dict[str, dict[str, Any]] = {}
    names = set()
    for record in manifest["files"]:
        name = record["path"]
        if not isinstance(name, str):
            raise ValueError("Unsafe backup path.")
        member = PurePosixPath(name)
        if (
            member.as_posix() != name
            or "\\" in name
            or ":" in name
            or member.is_absolute()
            or ".." in member.parts
        ):
            raise ValueError("Unsafe backup path.")
        if name != "labviz.db" and (len(member.parts) < 2 or member.parts[0] != "objects"):
            raise ValueError("Unexpected backup member.")
        if name.casefold() in names:
            raise ValueError("Duplicate backup member.")
        names.add(name.casefold())
        expected[name] = record
    actual = {
        p.relative_to(folder).as_posix(): p
        for p in members
        if p.relative_to(folder).as_posix() != "manifest.json"
    }
    if set(actual) != set(expected) or "labviz.db" not in actual:
        raise ValueError("Backup file set differs from its manifest.")
    for name, path in actual.items():
        if (
            path.stat().st_size != expected[name]["size"]
            or file_sha(path) != expected[name]["sha256"]
        ):
            raise ValueError(f"Backup hash mismatch: {name}")
    check_database(folder / "labviz.db")
    return dict(manifest)


def backup_data(data: Path, destination: Path) -> dict[str, Any]:
    # Caller must own the source-launcher mutex so database and objects have no writers.
    if destination.resolve().is_relative_to(data.resolve()):
        raise ValueError("Backup destination must be outside the source data directory.")
    database = data / "labviz.db"
    if not database.is_file() or database.is_symlink() or database.is_junction():
        raise ValueError("No regular source SQLite database found.")
    if (data / "objects").is_symlink() or (data / "objects").is_junction():
        raise ValueError("Linked object directories are unsupported.")
    if (data / "objects").exists() and not (data / "objects").is_dir():
        raise ValueError("Object storage path is not a directory.")
    # SQLite-only projects may never create the optional object directory.
    objects = safe_files(data / "objects") if (data / "objects").is_dir() else []
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "objects").mkdir()
    # SQLite read-only WAL connections can create shared-memory sidecars. Read a
    # stopped-writer copy instead, preserving every byte in the original directory.
    staging = destination / "_sqlite-staging"
    staging.mkdir()
    try:
        shutil.copy2(database, staging / "labviz.db")
        wal = data / "labviz.db-wal"
        if wal.exists():
            if wal.is_symlink() or wal.is_junction():
                raise ValueError("Linked WAL files are unsupported.")
            shutil.copy2(wal, staging / "labviz.db-wal")
        check_database(staging / "labviz.db")
        with (
            closing(
                sqlite3.connect((staging / "labviz.db").resolve().as_uri() + "?mode=ro", uri=True)
            ) as source,
            closing(sqlite3.connect(destination / "labviz.db")) as target_db,
        ):
            source.backup(target_db)
            target_db.execute("PRAGMA journal_mode=DELETE")
    finally:
        if not staging.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Unexpected staging directory.")
        for temporary in safe_files(staging):
            temporary.unlink()
        staging.rmdir()
    for file in objects:
        target = destination / "objects" / file.relative_to(data / "objects")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    check_database(destination / "labviz.db")
    manifest = {
        "format": "labviz-v23-source-backup",
        "schemaVersion": 1,
        "createdAt": datetime.now(UTC).isoformat(),
        "sourceDataDirectory": str(data.resolve()),
        "files": [
            {
                "path": p.relative_to(destination).as_posix(),
                "size": p.stat().st_size,
                "sha256": file_sha(p),
            }
            for p in safe_files(destination)
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return verify_backup(destination)


def restore_copy(backup: Path, destination: Path) -> dict[str, Any]:
    manifest = verify_backup(backup)
    if destination.resolve().is_relative_to(backup.resolve()):
        raise ValueError("Restore destination must be outside the backup directory.")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "objects").mkdir()
    for record in manifest["files"]:
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup / record["path"], target)
        if file_sha(target) != record["sha256"]:
            raise ValueError("Restored file hash mismatch.")
    check_database(destination / "labviz.db")
    restored = {
        "schemaVersion": 1,
        "format": "labviz-v23-restored-copy",
        "backupManifestSha256": file_sha(backup / "manifest.json"),
        "restoredAt": datetime.now(UTC).isoformat(),
        "files": manifest["files"],
    }
    (destination / "restore-manifest.json").write_text(
        json.dumps(restored, indent=2) + "\n", encoding="utf-8"
    )
    return restored


def unique_directory(parent: Path) -> Path:
    return parent / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
