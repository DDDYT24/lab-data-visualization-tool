"""Test the private clipboard bridge and Windows ownership using memory-only fakes."""
import ctypes
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
from test_preferences import module


class Kernel:
    def __init__(self):
        self.buffers, self.freed = {}, []
    def GlobalAlloc(self, flags, size):
        handle = len(self.buffers) + 10
        self.buffers[handle] = ctypes.create_string_buffer(size)
        return handle
    def GlobalLock(self, handle):
        return ctypes.addressof(self.buffers[handle]) if handle in self.buffers else 0
    def GlobalUnlock(self, handle):
        return 1
    def GlobalSize(self, handle):
        return ctypes.sizeof(self.buffers[handle]) if handle in self.buffers else 0
    def GlobalFree(self, handle):
        self.freed.append(handle)


class User:
    def __init__(self, busy=False, wrong=False):
        self.busy, self.wrong, self.emptied, self.closed = busy, wrong, False, False
        self.owner, self.destroyed, self.handle = None, None, None
    def CreateWindowExW(self, *args):
        return 123
    def DestroyWindow(self, owner):
        self.destroyed = owner
        return 1
    def OpenClipboard(self, owner):
        self.owner = owner
        return not self.busy
    def CloseClipboard(self):
        self.closed = True
        return 1
    def EmptyClipboard(self):
        self.emptied = True
        return 1
    def SetClipboardData(self, kind, handle):
        self.handle = handle
        return handle
    def GetClipboardData(self, kind):
        return 0 if self.wrong else self.handle


class NativeClipboardTests(unittest.TestCase):
    def clipboard(self, **options):
        result = object.__new__(module.WindowsClipboard)
        result.kernel, result.user = Kernel(), User(**options)
        return result

    def test_exact_unicode_and_newline_write_is_verified_before_unlocking(self):
        value = module.clipboard_bytes("科研资料\nvalue,4.5")
        clipboard = self.clipboard()
        self.assertTrue(clipboard.write(value))
        self.assertEqual(clipboard.user.owner, 123)
        self.assertTrue(clipboard.user.closed)
        self.assertEqual(clipboard.user.destroyed, 123)
        self.assertEqual(clipboard.kernel.freed, [])
        self.assertEqual(clipboard.kernel.buffers[10].raw, value)

    def test_busy_clipboard_is_never_cleared_and_untransferred_allocation_is_freed(self):
        clipboard = self.clipboard(busy=True)
        self.assertFalse(clipboard.write(module.clipboard_bytes("synthetic")))
        self.assertFalse(clipboard.user.emptied)
        self.assertEqual(clipboard.kernel.freed, [10])
        self.assertEqual(clipboard.user.destroyed, 123)

    def test_failed_readback_cannot_report_success_or_free_transferred_memory(self):
        clipboard = self.clipboard(wrong=True)
        self.assertFalse(clipboard.write(module.clipboard_bytes("synthetic")))
        self.assertTrue(clipboard.user.closed)
        self.assertEqual(clipboard.kernel.freed, [])

    def test_text_validation_rejects_embedded_nulls_surrogates_and_size_overflow(self):
        for value in (None, "a\0b", "\ud800", "x" * 65536):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                module.clipboard_bytes(value)


class ClipboardHttpTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.target = Path(self.directory.name) / "preferences.json"
        self.service = module.Preferences(self.target, "http://127.0.0.1:42001", "test-profile")
        self.bridge = self.service.descriptor()

    def tearDown(self):
        self.service.close()
        self.directory.cleanup()

    def request(self, value=None, *, raw=None, path="/clipboard/write", origin=None, token=None, method="POST"):
        headers = {"Content-Type": "application/json", "Origin": origin or self.service.origin,
            "X-LabViz-Desktop": token or self.bridge["token"]}
        data = raw if raw is not None else json.dumps({"text": "科研资料\nline 2"} if value is None else value).encode()
        request = urllib.request.Request(self.bridge["url"].removesuffix("/preferences") + path,
            data=data if method == "POST" else None, headers=headers, method=method)
        try:
            response = urllib.request.urlopen(request, timeout=4)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, dict(response.headers), json.load(response)

    def test_only_exact_origin_token_and_post_path_can_write(self):
        with patch.object(module, "write_clipboard_text", return_value=True) as write:
            status, headers, _ = self.request(origin="https://example.invalid")
            self.assertEqual(status, 403)
            self.assertNotIn("Access-Control-Allow-Origin", headers)
            self.assertEqual(self.request(token="wrong")[0], 403)
            self.assertEqual(self.request(path="/clipboard/write/extra")[0], 403)
            self.assertEqual(self.request(method="GET")[0], 403)
            write.assert_not_called()

    def test_success_and_busy_results_never_echo_or_persist_the_copied_text(self):
        with patch.object(module, "write_clipboard_text", return_value=True) as write:
            status, headers, result = self.request()
            self.assertEqual((status, result), (200, {"copied": True, "verified": True}))
            self.assertEqual(headers["Cache-Control"], "no-store")
            write.assert_called_once_with("科研资料\nline 2")
        with patch.object(module, "write_clipboard_text", return_value=False):
            self.assertEqual(self.request()[::2], (409, {"copied": False, "verified": False}))
        self.assertFalse(self.target.exists())
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_bad_schema_text_and_oversized_bodies_never_reach_windows(self):
        with patch.object(module, "write_clipboard_text", return_value=True) as write:
            for value in ({"text": 1}, {"text": "a\0b"}, {"text": "x", "extra": True}, {"text": "x" * 65536}):
                self.assertEqual(self.request(value)[0], 400)
            self.assertEqual(self.request(raw=b"x" * (module.CLIPBOARD_BODY_LIMIT + 1))[0], 413)
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
