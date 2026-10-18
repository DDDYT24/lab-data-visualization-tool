"""V2.3 installed identities; source and inherited local profiles stay separate."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

APP_ID = "55B46BB7-101E-46DC-951A-3DE7E402EDD0"
DATA_EPOCH = 1


def unlinked(path: Path) -> Path:
    path = path.absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise ValueError("Profile paths must not contain links or junctions.")
    return path.resolve()


def atomic_json(path: Path, value: Any) -> None:
    unlinked(path)
    temporary = path.with_name(path.name + ".new-" + uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ProfileLock:
    def __init__(self, path: Path):
        unlinked(path)
        self.stream: IO[bytes] = path.open("a+b")
        if path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise ValueError("Stop this installed profile before changing its data.") from None

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()

    def __enter__(self) -> ProfileLock:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


@dataclass(frozen=True)
class InstalledProfile:
    root: Path
    identifier: str

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def cookie_suffix(self) -> str:
        return "_v23_installed_" + self.identifier

    def lock(self) -> ProfileLock:
        return ProfileLock(self.root / ".writer.lock")

    def epoch(self) -> int:
        path = self.root / "data-schema.json"
        unlinked(path)
        if not path.exists():
            if (self.data / "labviz.db").exists():
                raise ValueError("Existing data requires verified import or schema metadata.")
            return DATA_EPOCH
        document = json.loads(path.read_text(encoding="utf-8"))
        value = document.get("dataEpoch")
        if document.get("profileId") != self.identifier or type(value) is not int or value < 1:
            raise ValueError("Invalid installed data-schema identity.")
        return int(value)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or not 1 <= epoch <= DATA_EPOCH:
            raise ValueError("This runtime cannot write the requested data epoch.")
        atomic_json(
            self.root / "data-schema.json", {"profileId": self.identifier, "dataEpoch": epoch}
        )

    @classmethod
    def open(cls, path: Path, *, create: bool = False) -> InstalledProfile:
        root = unlinked(path)
        local = unlinked(Path(os.environ["LOCALAPPDATA"]))
        default = local / "LabVizV23"
        copies = local / "LabVizV23Profiles"
        if root != default and not (
            root.parent == copies and re.fullmatch(r"[a-f0-9]{32}", root.name)
        ):
            raise ValueError("The destination is not an owned V2.3 installed profile root.")
        marker = root / "profile.json"
        unlinked(marker)
        if not marker.exists():
            if not create or (root.exists() and any(root.iterdir())):
                raise ValueError(
                    "Profile identity is missing; existing directories are never adopted."
                )
            root.mkdir(parents=True, exist_ok=True)
            identifier = root.name if root.parent == copies else uuid4().hex
            atomic_json(
                marker, {"format": 1, "appId": APP_ID, "profileId": identifier, "root": str(root)}
            )
        document = json.loads(marker.read_text(encoding="utf-8"))
        identifier = document.get("profileId", "")
        if (
            document.get("format") != 1
            or document.get("appId") != APP_ID
            or document.get("root") != str(root)
            or not isinstance(identifier, str)
            or not re.fullmatch(r"[a-f0-9]{32}", identifier)
            or (root.parent == copies and identifier != root.name)
        ):
            raise ValueError("Invalid installed profile identity.")
        return cls(root, identifier)


def environment_profile() -> InstalledProfile | None:
    line = os.environ.get("LABVIZ_INSTALL_LINE")
    if not line:
        return None
    if line != "2.3":
        raise ValueError("Unknown installed release line.")
    profile = InstalledProfile.open(Path(os.environ["LABVIZ_PROFILE_ROOT"]))
    if os.environ.get("LABVIZ_PROFILE_ID") != profile.identifier:
        raise ValueError("Installed cookie identity differs from the data profile.")
    return profile


def cookie_suffix() -> str:
    profile = environment_profile()
    if profile:
        return profile.cookie_suffix
    return "_v23" if os.environ.get("LABVIZ_DEVELOPMENT_LINE") == "2.3" else ""
