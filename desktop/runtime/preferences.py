"""Persist a bounded set of desktop preferences behind an exact-origin token."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

KEYS = {"labviz:user-preferences:v1", "labviz:v23:interface-mode:v1"}
FIELDS = {
    "appearance": {"system", "light", "dark"},
    "figureLanguage": {"same", "en", "zh"},
    "fontFamily": {"Arial", "Times New Roman"},
    "sizePreset": {"single-column", "double-column", "a4", "custom"},
    "unit": {"mm", "cm", "in"},
    "dpi": {300, 600},
    "grayscalePreview": {True, False},
}


def checked(value: dict, profile_id: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"format", "profileId", "locale", "storage"}:
        raise ValueError("Unknown desktop preference fields")
    if value["format"] != "labviz-desktop-preferences-v1" or value["profileId"] != profile_id or value["locale"] not in ("en", "zh"):
        raise ValueError("Desktop preference identity differs")
    storage = value["storage"]
    if not isinstance(storage, dict) or not set(storage) <= KEYS:
        raise ValueError("Only application preferences can be stored")
    for key, text in storage.items():
        if not isinstance(text, str) or len(text) > 4096:
            raise ValueError("Desktop preference value is too large")
        if key.endswith("interface-mode:v1"):
            if text not in ("classic", "overview"):
                raise ValueError("Unknown interface mode")
        else:
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or set(parsed) != set(FIELDS):
                raise ValueError("Unknown user preference fields")
            for field, choices in FIELDS.items():
                item = parsed[field]
                if field == "grayscalePreview" and type(item) is not bool:
                    raise ValueError("Grayscale must be boolean")
                if field == "dpi" and type(item) is not int:
                    raise ValueError("DPI must be an integer")
                if item not in choices:
                    raise ValueError("Unknown user preference value")
    return value


class Preferences:
    def __init__(self, path: Path, origin: str, profile_id: str):
        self.path = path
        self.origin = origin
        self.profile_id = profile_id
        self.token = secrets.token_urlsafe(32)
        self.generation = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.warning = None
        self.session = None
        self.maintenance = None
        self.value = {"format": "labviz-desktop-preferences-v1", "profileId": profile_id, "locale": "en", "storage": {}}
        if path.is_symlink() or path.is_junction():
            raise ValueError("Linked preference files are not supported")
        if path.exists():
            try:
                if path.stat().st_size > 8192:
                    raise ValueError("Existing preferences exceed the supported size")
                self.value = checked(json.loads(path.read_text(encoding="utf-8")), profile_id)
            except (ValueError, TypeError):
                self.warning = "Existing preferences could not be loaded; the original file was retained."
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                return

            def headers_for_origin(self) -> bool:
                return self.headers.get("Origin") == service.origin

            def respond(self, status: int, value: dict) -> None:
                body = json.dumps(value).encode()
                self.send_response(status)
                if self.headers_for_origin():
                    self.send_header("Access-Control-Allow-Origin", service.origin)
                    self.send_header("Vary", "Origin")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type, X-LabViz-Desktop")
                self.send_header("Content-Type", "application/json")
                self.send_header('Cache-Control', 'no-store')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:
                self.respond(200 if (self.path == "/preferences" or self.path.startswith(('/session/', '/maintenance/'))) and self.headers_for_origin() else 403, {})

            def do_GET(self) -> None:
                if service.maintenance and self.path.startswith('/maintenance/') and self.headers_for_origin() and hmac.compare_digest(self.headers.get('X-LabViz-Desktop', ''), service.token):
                    self.session_timeout()
                    service.maintenance.handle(self)
                    return
                if service.session and self.path.startswith('/session/') and self.headers_for_origin() and hmac.compare_digest(self.headers.get('X-LabViz-Desktop', ''), service.token):
                    self.session_timeout()
                    service.session.handle(self)
                else:
                    self.respond(403, {})

            def session_timeout(self):
                self.connection.settimeout(30)

            def deny_post(self):
                # Drain small rejected requests before closing on Windows; otherwise
                # an unread request body can reset the connection before the 403 arrives.
                self.connection.settimeout(1)
                try:
                    count=int(self.headers.get('Content-Length','0'))
                    if 0 < count <= 8192:self.rfile.read(count)
                except (OSError, ValueError):
                    pass
                self.respond(403, {'saved':False})

            def do_POST(self) -> None:
                if service.maintenance and self.path.startswith('/maintenance/') and self.headers_for_origin() and hmac.compare_digest(self.headers.get('X-LabViz-Desktop', ''), service.token):
                    self.session_timeout()
                    service.maintenance.handle(self)
                    return
                if service.session and self.path.startswith('/session/') and self.headers_for_origin() and hmac.compare_digest(self.headers.get('X-LabViz-Desktop', ''), service.token):
                    self.session_timeout()
                    service.session.handle(self)
                    return
                if self.path != "/preferences" or not self.headers_for_origin() or not hmac.compare_digest(self.headers.get("X-LabViz-Desktop", ""), service.token):
                    self.deny_post()
                    return
                try:
                    count = int(self.headers.get("Content-Length", "0"))
                    if not 0 < count <= 8192:
                        # Drain bounded rejected bodies before closing on
                        # Windows, so the caller can receive the 413 response.
                        if count > 0:
                            self.connection.settimeout(1)
                            try:
                                self.rfile.read(min(count, 65536))
                            except OSError:
                                pass
                        self.respond(413, {"saved": False})
                        return
                    value = checked(json.loads(self.rfile.read(count)), service.profile_id)
                    with service.lock:
                        temporary = service.path.with_name(service.path.name + ".new-" + uuid.uuid4().hex)
                        with temporary.open("x", encoding="utf-8") as stream:
                            json.dump(value, stream)
                            stream.flush()
                            os.fsync(stream.fileno())
                        if service.warning and service.path.exists():
                            retained = service.path.with_name(service.path.name + ".invalid-" + uuid.uuid4().hex)
                            os.replace(service.path, retained)
                        os.replace(temporary, service.path)
                        service.value = value
                        service.warning = None
                    self.respond(200, {"saved": True})
                except (ValueError, TypeError, OSError):
                    self.respond(400, {"saved": False})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def descriptor(self) -> dict:
        return {"url": f"http://127.0.0.1:{self.server.server_port}/preferences", "token": self.token, "generation":self.generation, "initial": self.value, "warning": self.warning}

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
