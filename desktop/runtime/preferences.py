"""Persist a bounded set of desktop preferences behind an exact-origin token."""

from __future__ import annotations

import hmac
import json
import os
import ctypes
from ctypes import wintypes
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
CLIPBOARD_BODY_LIMIT = 262144
CLIPBOARD_TEXT_LIMIT = 131072


def clipboard_bytes(text: str) -> bytes:
    if not isinstance(text, str) or "\0" in text:
        raise ValueError("Clipboard text is invalid")
    encoded = text.encode("utf-16-le") + b"\0\0"
    if len(encoded) > CLIPBOARD_TEXT_LIMIT:
        raise ValueError("Clipboard text exceeds the supported size")
    return encoded


class WindowsClipboard:
    def __init__(self):
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, arguments, result in (
            ("CreateWindowExW", [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID], wintypes.HWND),
            ("DestroyWindow", [wintypes.HWND], wintypes.BOOL),
            ("OpenClipboard", [wintypes.HWND], wintypes.BOOL),
            ("CloseClipboard", [], wintypes.BOOL),
            ("EmptyClipboard", [], wintypes.BOOL),
            ("SetClipboardData", [wintypes.UINT, wintypes.HANDLE], wintypes.HANDLE),
            ("GetClipboardData", [wintypes.UINT], wintypes.HANDLE),
        ):
            function = getattr(self.user, name)
            function.argtypes, function.restype = arguments, result
        for name, arguments, result in (
            ("GlobalAlloc", [wintypes.UINT, ctypes.c_size_t], wintypes.HANDLE),
            ("GlobalLock", [wintypes.HANDLE], ctypes.c_void_p),
            ("GlobalUnlock", [wintypes.HANDLE], wintypes.BOOL),
            ("GlobalSize", [wintypes.HANDLE], ctypes.c_size_t),
            ("GlobalFree", [wintypes.HANDLE], wintypes.HANDLE),
        ):
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result

    def write(self, encoded: bytes) -> bool:
        # A message-only window supplies a real owner without displaying UI.
        # OpenClipboard(NULL) followed by EmptyClipboard cannot SetClipboardData.
        owner = self.user.CreateWindowExW(0, "STATIC", "", 0, 0, 0, 0, 0,
            wintypes.HWND(-3), None, None, None)
        if not owner:
            return False
        handle, opened, transferred, copied = None, False, False, False
        try:
            handle = self.kernel.GlobalAlloc(0x42, len(encoded))
            pointer = self.kernel.GlobalLock(handle) if handle else None
            if not pointer:
                return False
            try:
                ctypes.memmove(pointer, encoded, len(encoded))
            finally:
                self.kernel.GlobalUnlock(handle)
            # Failure to acquire the clipboard must leave its existing data intact.
            if not self.user.OpenClipboard(owner):
                return False
            opened = True
            if not self.user.EmptyClipboard() or not self.user.SetClipboardData(13, handle):
                return False
            transferred = True
            observed = self.user.GetClipboardData(13)
            if not observed or self.kernel.GlobalSize(observed) < len(encoded):
                return False
            pointer = self.kernel.GlobalLock(observed)
            if not pointer:
                return False
            try:
                copied = ctypes.string_at(pointer, len(encoded)) == encoded
            finally:
                self.kernel.GlobalUnlock(observed)
        finally:
            if opened and not self.user.CloseClipboard():
                copied = False
            if handle and not transferred:
                self.kernel.GlobalFree(handle)
            self.user.DestroyWindow(owner)
        return copied


def write_clipboard_text(text: str) -> bool:
    encoded = clipboard_bytes(text)
    if os.name != "nt":
        return False
    return WindowsClipboard().write(encoded)


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
                self.respond(200 if (self.path in ("/preferences", "/clipboard/write") or self.path.startswith(('/session/', '/maintenance/'))) and self.headers_for_origin() else 403, {})

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
                if self.path == '/clipboard/write':
                    if not self.headers_for_origin() or not hmac.compare_digest(self.headers.get('X-LabViz-Desktop', ''), service.token):
                        self.deny_post()
                        return
                    self.connection.settimeout(2)
                    try:
                        count = int(self.headers.get('Content-Length', '0'))
                        if not 0 < count <= CLIPBOARD_BODY_LIMIT:
                            if count > 0:
                                try:
                                    self.rfile.read(min(count, CLIPBOARD_BODY_LIMIT * 2))
                                except OSError:
                                    pass
                            self.respond(413, {'copied': False, 'verified': False})
                            return
                        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                            raise ValueError('Clipboard requests require JSON')
                        body = self.rfile.read(count)
                        if len(body) != count:
                            raise ValueError('Incomplete clipboard request')
                        value = json.loads(body)
                        if not isinstance(value, dict) or set(value) != {'text'}:
                            raise ValueError('Unknown clipboard request fields')
                        clipboard_bytes(value['text'])
                        copied = write_clipboard_text(value['text'])
                        self.respond(200 if copied else 409, {'copied': copied, 'verified': copied})
                    except (ValueError, TypeError, OSError):
                        self.respond(400, {'copied': False, 'verified': False})
                    return
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
