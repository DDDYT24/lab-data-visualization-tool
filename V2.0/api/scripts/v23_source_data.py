"""Internal offline helper; use source-data-v23.ps1 to acquire the launcher mutex."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from labviz_api.source_data import backup_data, restore_copy, unique_directory, verify_backup


def within(path: Path, parents: list[Path]) -> Path:
    if any(part.is_symlink() or part.is_junction() for part in (path, *path.parents)):
        raise ValueError("Linked source data paths are unsupported.")
    resolved = path.resolve()
    workspace = Path(__file__).resolve().parents[3]
    if not resolved.is_relative_to(workspace) or not any(
        resolved.is_relative_to(parent.resolve()) for parent in parents
    ):
        raise ValueError("The path is outside this V2.3 workspace's permitted data directories.")
    return resolved


def ps_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["backup", "verify", "restore"])
    parser.add_argument("--data-directory", type=Path)
    parser.add_argument("--backup-directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    backup_root = root / "outputs/v23-backups"
    data_root = root / "V2.0/api/.labviz"
    result: dict[str, Any]
    try:
        if args.action == "backup":
            data = within(
                args.data_directory or data_root / "v23-development",
                [
                    data_root / "v23-development",
                    data_root / "v23-restored",
                    root / "outputs/v23-phase5",
                ],
            )
            destination = within(unique_directory(backup_root), [backup_root])
            manifest = backup_data(data, destination)
            result = {
                "backupDirectory": str(destination),
                "verified": True,
                "fileCount": len(manifest["files"]),
            }
        else:
            if args.backup_directory is None:
                raise ValueError("Choose an existing backup directory.")
            backup = within(args.backup_directory, [backup_root])
            if args.action == "verify":
                manifest = verify_backup(backup)
                result = {
                    "backupDirectory": str(backup),
                    "verified": True,
                    "fileCount": len(manifest["files"]),
                }
            else:
                destination = within(
                    unique_directory(data_root / "v23-restored"), [data_root / "v23-restored"]
                )
                restore_copy(backup, destination)
                result = {
                    "restoredDirectory": str(destination),
                    "verified": True,
                    "startCommand": "& "
                    + ps_quote(root / "start-labviz.ps1")
                    + " -DataDirectory "
                    + ps_quote(destination),
                }
        print(json.dumps(result))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        parser.exit(2, json.dumps({"code": "source-data-failed", "message": str(error)}) + "\n")


if __name__ == "__main__":
    main()
