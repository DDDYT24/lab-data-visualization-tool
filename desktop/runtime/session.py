"""Durable desktop draft snapshots and content-addressed, owned file staging."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

MAX_FILE = 50 * 1024 * 1024
MAX_FILES_TOTAL = 100 * 1024 * 1024
MAX_SNAPSHOT = 16 * 1024 * 1024
MAX_STAGING = 512 * 1024 * 1024
MAX_CAPTURE_RESERVE = MAX_FILES_TOTAL
NAMESPACES = {
    "home",
    "workspace",
    "navigation",
    "research",
    "chart",
    "figure",
    "reuse",
    "description",
    "import",
    "inspect",
    "source-tools",
    "inspector",
}


def io_path(path: Path) -> Path:
    """Use extended Windows paths for IO without changing persisted profile identity."""
    if os.name != 'nt':
        return path
    value = str(path.absolute())
    if value.startswith('\\\\?\\'):
        return Path(value)
    return Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)


def unlinked(path: Path) -> Path:
    for item in (path, *path.parents):
        if io_path(item).is_symlink() or io_path(item).is_junction():
            raise ValueError("Linked session paths are not supported")
    return path


def encoded(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic(path: Path, data: bytes):
    unlinked(path)
    temporary = path.with_name(path.name + ".new-" + uuid.uuid4().hex)
    try:
        with io_path(temporary).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(io_path(temporary), io_path(path))
    finally:
        if io_path(temporary).exists():
            io_path(temporary).unlink()


class Session:
    def __init__(self, root: Path, profile_id: str, bases, figures=None):
        self.root, self.profile_id, self.bases = unlinked(root), profile_id, bases
        self.figures = figures
        root.mkdir(exist_ok=True)
        self.files = unlinked(root / "files")
        self.files.mkdir(exist_ok=True)
        self.path = root / "checkpoint.json"
        self.lock = threading.RLock()
        self.warning = None
        self.initial = None
        self.pending = None
        self.acknowledged = None
        self.staged_in_checkpoint = set()
        self.activity = {"ready": False, "busy": True, "pendingRequests": 0}
        self.last_heartbeat = 0.0
        self.error = None
        self.loaded()

    def loaded(self):
        if not unlinked(self.path).exists():
            return
        try:
            if self.path.stat().st_size > MAX_SNAPSHOT:
                raise ValueError("The checkpoint is too large")
            envelope = json.loads(self.path.read_bytes())
            if (
                envelope["format"] != "labviz-desktop-checkpoint-envelope-v1"
                or hashlib.sha256(encoded(envelope["payload"])).hexdigest()
                != envelope["sha256"]
            ):
                raise ValueError("The checkpoint checksum differs")
            self.validate(envelope["payload"], check_bases=False)
            self.initial = envelope["payload"]
        except (KeyError, ValueError, TypeError, OSError) as error:
            self.warning = (
                "The previous desktop draft could not be restored. Its original file was retained. "
                + str(error)
            )

    def validate(self, value, check_bases=True):
        if not isinstance(value, dict) or set(value) != {
            "format",
            "profileId",
            "route",
            "entries",
            "checkpointId",
        }:
            raise ValueError("Unknown checkpoint fields")
        if (
            value["format"] != "labviz-desktop-checkpoint-v1"
            or value["profileId"] != self.profile_id
        ):
            raise ValueError("Checkpoint identity or version differs")
        route = urlsplit(value["route"])
        if (
            route.scheme
            or route.netloc
            or route.fragment
            or not re.fullmatch(
                r"/(?:workspace/[a-zA-Z0-9-]+|history|help|about|settings)?", route.path
            )
            or len(value["route"]) > 2048
        ):
            raise ValueError("Invalid recovery route")
        entries = value["entries"]
        if not isinstance(entries, dict) or len(entries) > 100:
            raise ValueError("Too many draft components")
        references = {}

        def walk(node, depth=0):
            if depth > 60:
                raise ValueError("Draft nesting is too deep")
            if isinstance(node, dict):
                if "__labvizFile" in node:
                    if (
                        set(node)
                        != {"__labvizFile", "name", "type", "lastModified", "size"}
                        or not isinstance(node["name"], str)
                        or len(node["name"]) > 512
                    ):
                        raise ValueError("Invalid staged file reference")
                    identifier = node["__labvizFile"]
                    data = self.file(identifier)
                    if len(data) != node["size"]:
                        raise ValueError("Staged file size differs")
                    references[identifier] = len(data)
                else:
                    for item in node.values():
                        walk(item, depth + 1)
            elif isinstance(node, list):
                for item in node:
                    walk(item, depth + 1)

        projects = set()
        for key, entry in entries.items():
            if (
                not isinstance(key, str)
                or key.split(":", 1)[0] not in NAMESPACES
                or len(key) > 180
            ):
                raise ValueError("Unknown draft component")
            if (
                not isinstance(entry, dict)
                or set(entry) != {"version", "projectId", "base", "value"}
                or entry["version"] != 1
            ):
                raise ValueError("Unknown draft component version")
            project = entry["projectId"]
            if project is not None:
                if not isinstance(project, str) or not re.fullmatch(
                    "[a-zA-Z0-9-]{1,128}", project
                ):
                    raise ValueError("Invalid project identity")
                projects.add(project)
            if not isinstance(entry["base"], dict) or not set(entry["base"]) <= {
                "project",
                "research",
                "figure",
            }:
                raise ValueError("Unknown draft baseline")
            walk(entry["value"])
        if sum(references.values()) > MAX_FILES_TOTAL or len(references) > 15:
            raise ValueError("Staged source files exceed the checkpoint limit")
        if check_bases and projects:
            actual = self.bases(sorted(projects))
            for entry in entries.values():
                project = entry["projectId"]
                if project and (
                    actual.get(project) is None
                    or any(
                        actual[project].get(scope) != revision
                        for scope, revision in entry["base"].items()
                        if scope != "figure" or self.figures is None
                    )
                ):
                    raise ValueError(
                        "A saved project revision changed. The draft was retained; reopen and resolve it before pausing."
                    )
            bindings = [
                {"projectId": entry["projectId"], "figureId": entry["base"]["figure"]}
                for entry in entries.values()
                if entry["projectId"] and entry["base"].get("figure")
            ]
            if bindings and self.figures and not self.figures(bindings):
                raise ValueError(
                    "A draft figure source is unavailable. The original draft was retained."
                )
        return value

    def file(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(
            "[a-f0-9]{64}", identifier
        ):
            raise ValueError("Invalid staging identity")
        path = unlinked(self.files / identifier)
        if path.stat().st_size > MAX_FILE:
            raise ValueError("Staged file exceeds the limit")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != identifier:
            raise ValueError("Staged source checksum differs")
        return data

    def stage(self, data: bytes):
        if len(data) > MAX_FILE:
            raise ValueError("A source file exceeds 50 MiB")
        identifier = hashlib.sha256(data).hexdigest()
        target = unlinked(self.files / identifier)
        with self.lock:
            self.staged_in_checkpoint.add(identifier)
            if not target.exists():
                total = sum(unlinked(p).stat().st_size for p in self.files.iterdir())
                reserved = False
                if total + len(data) > MAX_STAGING and self.pending and self.initial and not self.warning:
                    # A bounded reserve permits an atomic replacement without
                    # deleting failed recovery material before the new ACK.
                    try:
                        protected = self.protected_sources()
                        if protected is not None:
                            references, candidates = protected
                            reclaimable = sum(p.stat().st_size for p in candidates if p.name not in references)
                            reserved = (total + len(data) <= MAX_STAGING + MAX_CAPTURE_RESERVE
                                        and total + len(data) - reclaimable <= MAX_STAGING)
                    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
                        pass
                if total + len(data) > MAX_STAGING and not reserved:
                    raise ValueError(
                        "Desktop staging is full. Remove newly selected files and save a checkpoint "
                        "to reclaim unused staged sources. Retained recovery sources are protected."
                    )
                atomic(target, data)
        return {"id": identifier, "size": len(data)}

    def begin(self, identifier: str):
        with self.lock:
            self.pending = identifier
            self.staged_in_checkpoint.clear()
            self.acknowledged = None
            self.error = None

    def cancel(self):
        with self.lock:
            self.pending = None
            self.acknowledged = None

    def commit(self, value):
        with self.lock:
            if not self.pending or value.get("checkpointId") != self.pending:
                raise ValueError("This checkpoint request was cancelled or replaced")
            try:
                self.validate(value)
            except ValueError as error:
                if str(error).startswith(
                    ("A saved project revision changed", "A draft figure source")
                ):
                    retained = encoded(
                        {
                            "format": "labviz-desktop-checkpoint-envelope-v1",
                            "sha256": hashlib.sha256(encoded(value)).hexdigest(),
                            "payload": value,
                        }
                    )
                    if len(retained) <= MAX_SNAPSHOT:
                        atomic(
                            self.root
                            / ("checkpoint.conflict-" + uuid.uuid4().hex + ".json"),
                            retained,
                        )
                raise
            data = encoded(
                {
                    "format": "labviz-desktop-checkpoint-envelope-v1",
                    "sha256": hashlib.sha256(encoded(value)).hexdigest(),
                    "payload": value,
                }
            )
            if len(data) > MAX_SNAPSHOT:
                raise ValueError("The draft snapshot exceeds 16 MiB")
            if self.warning and self.path.exists():
                atomic(
                    self.root / ("checkpoint.invalid-" + uuid.uuid4().hex + ".json"),
                    self.path.read_bytes(),
                )
            elif self.path.exists():
                atomic(self.root / "checkpoint.previous.json", self.path.read_bytes())
            atomic(self.path, data)
            self.initial = value
            self.acknowledged = self.pending
            self.error = None
            self.warning = None
            self.staged_in_checkpoint.clear()
            # Cleanup is deliberately after durable replacement. A failed write
            # must leave every source needed by the previous state untouched.
            try:
                self.collect_unreferenced()
            except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
                pass  # Saving succeeded; optional cleanup must not revoke its ACK.
            return {
                "saved": True,
                "checkpointId": self.pending,
                "sha256": hashlib.sha256(data).hexdigest(),
            }

    def protected_sources(self):
        """Inspect all retained snapshots before reserving or reclaiming space.

        Current, previous and all retained recovery snapshots protect their
        sources. Unknown/corrupt snapshots or unfamiliar file entries stop
        cleanup; preserving recovery material takes precedence over capacity.
        """
        references = set(self.staged_in_checkpoint)

        def walk(node):
            if isinstance(node, dict):
                if "__labvizFile" in node:
                    identifier = node["__labvizFile"]
                    if not isinstance(identifier, str) or not re.fullmatch("[a-f0-9]{64}", identifier):
                        raise ValueError("Unknown recovery source")
                    references.add(identifier)
                else:
                    for value in node.values():
                        walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        with self.lock:
            if not self.path.exists():
                return None
            for path in io_path(self.root).glob("checkpoint*.json"):
                path = unlinked(path)
                if path.stat().st_size > MAX_SNAPSHOT:
                    return None
                envelope = json.loads(path.read_bytes())
                if (envelope.get("format") != "labviz-desktop-checkpoint-envelope-v1"
                        or envelope.get("sha256") != hashlib.sha256(encoded(envelope["payload"])).hexdigest()
                        or envelope["payload"].get("profileId") != self.profile_id):
                    return None
                self.validate(envelope["payload"], check_bases=False)
                walk(envelope["payload"].get("entries"))
            candidates = list(io_path(self.files).iterdir())
            for path in candidates:
                unlinked(path)
                if (not path.is_file() or path.stat().st_nlink != 1
                        or not re.fullmatch("[a-f0-9]{64}", path.name)):
                    return None
            return references, candidates

    def collect_unreferenced(self):
        """Delete unreferenced owned sources only after a durable replacement."""
        with self.lock:
            if self.acknowledged is None or self.acknowledged != self.pending:
                return 0
            protected = self.protected_sources()
            if protected is None:
                return 0
            references, candidates = protected
            removed = 0
            for path in candidates:
                if path.name not in references:
                    io_path(path).unlink()
                    removed += 1
            return removed

    def idle(self):
        return (
            time.monotonic() - self.last_heartbeat < 4
            and self.activity.get("ready") is True
            and self.activity.get("busy") is False
            and self.activity.get("pendingRequests") == 0
            and not self.warning
        )

    def descriptor(self):
        return {
            "initial": self.initial,
            "warning": self.warning,
            "format": "labviz-desktop-session-v1",
        }

    def handle(self, handler):
        """Called only after the exact Origin and random token have been checked."""
        path = handler.path
        try:
            if handler.command == "GET" and path.startswith("/session/files/"):
                data = self.file(path.removeprefix("/session/files/"))
                handler.send_response(200)
                handler.send_header(
                    "Access-Control-Allow-Origin", handler.headers["Origin"]
                )
                handler.send_header("Content-Type", "application/octet-stream")
                handler.send_header("Cache-Control", "no-store")
                handler.send_header("Content-Length", str(len(data)))
                handler.end_headers()
                handler.wfile.write(data)
                return
            count = int(handler.headers.get("Content-Length", "0"))
            limit = MAX_FILE if path == "/session/files" else MAX_SNAPSHOT
            if not 0 < count <= limit:
                raise ValueError("Desktop request exceeds the supported size")
            raw = handler.rfile.read(count)
            if len(raw) != count:
                raise ValueError("Incomplete desktop request")
            if path == "/session/files":
                result = self.stage(raw)
            else:
                value = json.loads(raw)
                if path == "/session/checkpoint":
                    result = self.commit(value)
                elif path == "/session/activity":
                    with self.lock:
                        if (
                            set(value) != {"ready", "busy", "pendingRequests"}
                            or type(value["pendingRequests"]) is not int
                            or not 0 <= value["pendingRequests"] <= 10000
                        ):
                            raise ValueError("Invalid client activity")
                        self.activity, self.last_heartbeat = value, time.monotonic()
                    result = {"error": self.error, "warning": self.warning}
                elif path == "/session/checkpoint-error":
                    with self.lock:
                        if value.get("checkpointId") == self.pending:
                            self.error = str(
                                value.get("message", "Draft capture failed")
                            )[:1000]
                    result = {"retained": True}
                elif path == "/session/acknowledge-warning":
                    with self.lock:
                        self.warning = None
                    result = {"acknowledged": True}
                else:
                    raise ValueError("Unknown desktop session request")
            handler.respond(200, result)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.error = str(error)
            handler.respond(409, {"saved": False, "message": self.error})
