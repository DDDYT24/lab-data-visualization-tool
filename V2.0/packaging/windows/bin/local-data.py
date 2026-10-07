"""Offline SQLite/object snapshots. Call only while both LabViz services are stopped.

Never merges databases or deletes data: displaced trees remain beside the destination.
The desktop coordinator owns the installation mutex for the entire transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4


def safe_tree(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink() or path.is_junction():
            raise ValueError("Linked data directories are not supported.")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rename_tree(source: Path, destination: Path) -> None:
    # Windows scanners may briefly hold a newly closed SQLite file. Keep both
    # directories intact if sharing violations persist; never delete to retry.
    for attempt in range(20):
        try:
            source.rename(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "snapshot.json"
    }


def snapshot(source: Path, destination: Path) -> None:
    source, destination = source.absolute(), destination.absolute()
    safe_tree(source)
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Snapshot destination must be new and outside source data.")
    database = source / "labviz-v2.db"
    if not database.is_file():
        raise ValueError("Select a LabViz data folder containing labviz-v2.db.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + ".partial-" + uuid4().hex)
    # A write reservation rejects a running writer and keeps the SQLite view stable.
    # Operators must also stop the old app: this lock cannot protect external objects.
    with closing(
        sqlite3.connect(database.as_uri() + "?mode=rw", uri=True, timeout=1)
    ) as guard:
        guard.execute("BEGIN IMMEDIATE")
        try:
            stage.mkdir()
            with closing(
                sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
            ) as src:
                tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master")}
                if not {"projects", "exports", "auth_sessions"}.issubset(tables):
                    raise ValueError(
                        "The source is not a supported LabViz SQLite database."
                    )
                with closing(sqlite3.connect(stage / "labviz-v2.db")) as dst:
                    src.backup(dst)
                    if dst.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                        raise ValueError("SQLite integrity check failed.")
            for item in source.iterdir():
                if item.name in {
                    "labviz-v2.db",
                    "labviz-v2.db-wal",
                    "labviz-v2.db-shm",
                    "snapshot.json",
                }:
                    continue
                if item.is_dir():
                    shutil.copytree(item, stage / item.name)
                else:
                    shutil.copy2(item, stage / item.name)
            (stage / "snapshot.json").write_text(
                json.dumps({"format": 1, "files": manifest(stage)}, indent=2),
                encoding="utf-8",
            )
            rename_tree(stage, destination)
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise
        finally:
            guard.rollback()


def restore(source: Path, destination: Path, *, replace: bool = False) -> None:
    source, destination = source.absolute(), destination.absolute()
    safe_tree(source)
    safe_tree(destination)
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("Snapshot and data paths must not overlap.")
    metadata = json.loads((source / "snapshot.json").read_text(encoding="utf-8"))
    if metadata.get("format") != 1 or manifest(source) != metadata.get("files"):
        raise ValueError(
            "Snapshot is incomplete or has changed; data was not replaced."
        )
    if destination.exists() and any(destination.iterdir()) and not replace:
        raise ValueError(
            "Destination already contains data. Import never merges or overwrites it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + ".restore-" + uuid4().hex)
    shutil.copytree(source, stage)
    displaced = destination.with_name(destination.name + ".retained-" + uuid4().hex)
    existed = destination.exists()
    if existed:
        rename_tree(destination, displaced)
    try:
        rename_tree(stage, destination)
    except OSError:
        if existed:
            rename_tree(displaced, destination)
        raise


def refresh_metadata(root: Path) -> None:
    (root / "snapshot.json").write_text(
        json.dumps({"format": 1, "files": manifest(root)}, indent=2),
        encoding="utf-8",
    )


def sanitize_import(destination: Path) -> None:
    """Reset transient browser/auth state while retaining guest project data."""
    database = destination / "labviz-v2.db"
    with closing(sqlite3.connect(database)) as connection:
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master")
        }
        for table in (
            "auth_sessions",
            "auth_challenges",
            "auth_requests",
            "auth_rate_limit_buckets",
        ):
            if table in tables:
                connection.execute(f'DELETE FROM "{table}"')
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    refresh_metadata(destination)


def import_data(source: Path, destination: Path) -> None:
    backup = destination.parent / "backups" / ("import-" + uuid4().hex)
    snapshot(source, backup)
    restore(backup, destination)
    # A copied browser session is not a supported migration contract. Keep
    # legacy guest and account ownership unchanged: the account-free profile
    # must not silently claim these records. Users can re-import source data.
    sanitize_import(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "restore", "import"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.action == "snapshot":
        snapshot(args.source, args.destination)
    elif args.action == "restore":
        restore(args.source, args.destination, replace=True)
    else:
        import_data(args.source, args.destination)
    print("OK")
