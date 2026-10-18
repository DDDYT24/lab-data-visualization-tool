"""Offline installed data operations. Scientific identifiers and frozen bytes are retained."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from .installed_profile import APP_ID, DATA_EPOCH, InstalledProfile, atomic_json, unlinked
from .source_data import (
    backup_data,
    file_sha,
    restore_copy,
    safe_files,
    unique_directory,
    verify_backup,
)


@contextmanager
def readonly_source_guard(root: Path) -> Iterator[None]:
    """On Windows, reject existing writers and deny new writes during the copy."""
    files = safe_files(root)
    handles: list[Any] = []
    library: Any = None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        library = ctypes.WinDLL("kernel32", use_last_error=True)
        library.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        library.CreateFileW.restype = wintypes.HANDLE
        library.CloseHandle.argtypes = [wintypes.HANDLE]
        try:
            for path in files:
                if path.name == ".writer.lock":
                    continue
                handle = library.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
                if handle == ctypes.c_void_p(-1).value:
                    raise OSError(
                        ctypes.get_last_error(),
                        "Stop source writers before copying or deleting data.",
                    )
                handles.append(handle)
        except BaseException:
            for handle in handles:
                library.CloseHandle(handle)
            raise
    try:
        yield
    finally:
        if library is not None:
            for handle in handles:
                library.CloseHandle(handle)


def schema_fingerprint(database: Path) -> str:
    import hashlib

    with closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        schema = db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def verify_installed(backup: Path) -> dict[str, Any]:
    unlinked(backup)
    manifest = verify_backup(backup)
    info = manifest.get("installed", {})
    if (
        info.get("appId") != APP_ID
        or type(info.get("dataEpoch")) is not int
        or info["dataEpoch"] < 1
        or info.get("schemaSha256") != schema_fingerprint(backup / "labviz.db")
    ):
        raise ValueError("Backup identity or data-schema metadata is invalid.")
    return manifest


def _backup(profile: InstalledProfile) -> Path:
    safe_files(profile.data)
    parent = unlinked(profile.root / "backups")
    destination = unique_directory(parent)
    manifest = backup_data(profile.data, destination)
    manifest["installed"] = {
        "appId": APP_ID,
        "sourceProfileId": profile.identifier,
        "dataEpoch": profile.epoch(),
        "schemaSha256": schema_fingerprint(destination / "labviz.db"),
    }
    atomic_json(destination / "manifest.json", manifest)
    verify_installed(destination)
    return destination


def backup_profile(profile: InstalledProfile) -> Path:
    with profile.lock(), readonly_source_guard(profile.data):
        return _backup(profile)


def _restore_new(backup: Path, target: InstalledProfile) -> dict[str, Any]:
    manifest = verify_installed(backup)
    epoch = manifest["installed"]["dataEpoch"]
    if epoch > DATA_EPOCH:
        raise ValueError("This runtime cannot restore a newer data schema.")
    if target.data.exists():
        raise ValueError("Restore requires an empty independent profile.")
    restore_copy(backup, target.data)
    target.set_epoch(epoch)
    return manifest


def restore_independent(backup: Path) -> InstalledProfile:
    manifest = verify_installed(backup)
    if manifest["installed"]["dataEpoch"] > DATA_EPOCH:
        raise ValueError("A compatible runtime is required for this backup.")
    root = Path(os.environ["LOCALAPPDATA"]) / "LabVizV23Profiles" / uuid4().hex
    target = InstalledProfile.open(root, create=True)
    with target.lock():
        _restore_new(backup, target)
    return target


def _journal_paths(profile: InstalledProfile, document: dict[str, Any]) -> tuple[Path, Path]:
    if document.get("profileId") != profile.identifier:
        raise ValueError("Restore journal belongs to a different profile.")
    incoming, retained = (unlinked(Path(document[k])) for k in ("incoming", "retained"))
    if (
        incoming.parent != profile.root
        or retained.parent != profile.root
        or not incoming.name.startswith("restore-staging-")
        or not retained.name.startswith("data.retained-")
        or type(document.get("targetEpoch")) is not int
        or not 1 <= document["targetEpoch"] <= DATA_EPOCH
    ):
        raise ValueError("Unsafe restore journal.")
    return incoming, retained


def _recover(profile: InstalledProfile) -> None:
    journal = unlinked(profile.root / "restore-journal.json")
    if not journal.exists():
        return
    document = json.loads(journal.read_text(encoding="utf-8"))
    incoming, retained = _journal_paths(profile, document)
    if incoming.exists():
        # Validate the complete staged copy before any active directory is moved.
        safe_files(incoming)
        files = document["files"]
        for record in files:
            member = unlinked(incoming / record["path"])
            if not member.is_relative_to(incoming) or file_sha(member) != record["sha256"]:
                raise ValueError("Staged restore is incomplete or changed.")
        if profile.data.exists():
            if retained.exists():
                raise ValueError("Ambiguous restore state; all copies retained.")
            unlinked(profile.data)
            os.replace(profile.data, retained)
        os.replace(incoming, profile.data)
    elif not profile.data.exists():
        if retained.exists():
            os.replace(retained, profile.data)
            profile.set_epoch(document["previousEpoch"])
            journal.unlink()
            return
        raise ValueError("No verified restore copy or previous data is available.")
    # An interrupted rename can be completed without opening data with an older app.
    profile.set_epoch(document["targetEpoch"])
    journal.unlink()


def recover_profile(profile: InstalledProfile) -> None:
    with profile.lock():
        _recover(profile)


def replace_active(profile: InstalledProfile, backup: Path) -> Path:
    manifest = verify_installed(backup)
    if manifest["installed"].get("sourceProfileId") != profile.identifier:
        raise ValueError("An active rollback requires a backup of the same installed profile.")
    epoch = manifest["installed"]["dataEpoch"]
    if epoch > DATA_EPOCH:
        raise ValueError("Incompatible rollback backup; current data was not changed.")
    with profile.lock():
        _recover(profile)
        previous_epoch = profile.epoch()
        incoming = profile.root / ("restore-staging-" + uuid4().hex)
        retained = profile.root / ("data.retained-" + uuid4().hex)
        restore_copy(backup, incoming)
        atomic_json(
            profile.root / "restore-journal.json",
            {
                "profileId": profile.identifier,
                "incoming": str(incoming),
                "retained": str(retained),
                "targetEpoch": epoch,
                "previousEpoch": previous_epoch,
                "files": manifest["files"],
            },
        )
        _recover(profile)
        return retained


def import_legacy(profile: InstalledProfile, source: Path) -> Path:
    source = unlinked(source)
    safe_files(source)
    if source.is_relative_to(profile.root) or profile.root.is_relative_to(source):
        raise ValueError("Import source and installed destination must not overlap.")
    database = next(
        (source / n for n in ("labviz.db", "labviz-v2.db") if (source / n).is_file()), None
    )
    if database is None:
        raise ValueError("No legacy SQLite database found.")
    with profile.lock(), readonly_source_guard(source):
        if profile.data.exists() and any(profile.data.iterdir()):
            raise ValueError("Import never merges or overwrites an installed data directory.")
        # The caller must stop the source app and own its launcher mutex. Copying
        # DB/WAL bytes avoids SQLite creating SHM sidecars in the old directory.
        staging = unlinked(profile.root / "backups") / ("import-source-" + uuid4().hex)
        staging.mkdir(parents=True)
        shutil.copy2(database, staging / "labviz.db")
        wal = database.with_name(database.name + "-wal")
        if wal.exists():
            shutil.copy2(wal, staging / "labviz.db-wal")
        if (source / "objects").exists():
            shutil.copytree(source / "objects", staging / "objects")
        backup = unique_directory(profile.root / "backups")
        manifest = backup_data(staging, backup)
        manifest["installed"] = {
            "appId": APP_ID,
            "sourceProfileId": None,
            "dataEpoch": 1,
            "schemaSha256": schema_fingerprint(backup / "labviz.db"),
        }
        atomic_json(backup / "manifest.json", manifest)
        if profile.data.exists():
            profile.data.rmdir()
        _restore_new(backup, profile)
        return backup


def remove_profile(profile: InstalledProfile, confirmed_id: str) -> None:
    if confirmed_id != profile.identifier:
        raise ValueError("Explicit deletion requires the exact profile ID.")
    # Validate the exact owned root and every entry before recursive deletion.
    current = InstalledProfile.open(profile.root)
    if current != profile:
        raise ValueError("Profile identity changed.")
    with profile.lock():
        safe_files(profile.root)
        for marker in profile.root.rglob("profile.json"):
            if marker.parent != profile.root:
                raise ValueError("Nested profiles must be retained independently.")
        with readonly_source_guard(profile.root):
            pass
    if unlinked(profile.root) != profile.root:
        raise ValueError("Profile path changed.")
    shutil.rmtree(profile.root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "init",
            "info",
            "backup",
            "verify",
            "restore",
            "import",
            "replace",
            "recover",
            "remove",
        ],
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--confirm-profile-id", default="")
    args = parser.parse_args()
    profile = InstalledProfile.open(args.profile, create=args.action == "init")
    result: dict[str, Any] = {
        "appId": APP_ID,
        "profileId": profile.identifier,
        "root": str(profile.root),
    }
    if args.action in {"verify", "restore", "import", "replace"} and args.source is None:
        parser.error("--source is required for this action")
    if args.action == "backup":
        result["backupDirectory"] = str(backup_profile(profile))
    elif args.action == "verify":
        assert args.source is not None
        result["manifest"] = verify_installed(args.source)
    elif args.action == "restore":
        assert args.source is not None
        target = restore_independent(args.source)
        result.update(restoredProfileRoot=str(target.root), restoredProfileId=target.identifier)
    elif args.action == "import":
        assert args.source is not None
        result["backupDirectory"] = str(import_legacy(profile, args.source))
    elif args.action == "replace":
        assert args.source is not None
        result["retainedDataDirectory"] = str(replace_active(profile, args.source))
    elif args.action == "recover":
        recover_profile(profile)
    elif args.action == "remove":
        remove_profile(profile, args.confirm_profile_id)
        result["removed"] = True
    if args.action != "remove":
        result["dataEpoch"] = profile.epoch()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
