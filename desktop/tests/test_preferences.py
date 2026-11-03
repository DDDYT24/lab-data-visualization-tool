"""Exercise preference persistence, origin checks and invalid input over HTTP."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "runtime/preferences.py"
spec = importlib.util.spec_from_file_location("desktop_preferences", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.target = Path(self.directory.name) / "preferences.json"
        self.service = module.Preferences(self.target, "http://127.0.0.1:42001", "test-profile")
        self.descriptor = self.service.descriptor()
        self.value = {"format": "labviz-desktop-preferences-v1", "profileId": "test-profile", "locale": "zh", "storage": {"labviz:v23:interface-mode:v1": "overview"}}

    def tearDown(self):
        self.service.close()
        self.directory.cleanup()

    def request(self, value=None, *, origin=None, token=None, raw=None):
        request = urllib.request.Request(self.descriptor["url"], data=raw or json.dumps(self.value if value is None else value).encode(), headers={"Content-Type": "application/json", "Origin": origin or self.service.origin, "X-LabViz-Desktop": token or self.descriptor["token"]})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, dict(response.headers), json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), json.load(error)

    def test_confirmed_save_survives_service_restart(self):
        self.assertEqual(self.request()[0], 200)
        self.assertEqual(json.loads(self.target.read_text()), self.value)
        self.service.close()
        self.service = module.Preferences(self.target, "http://127.0.0.1:42002", "test-profile")
        self.assertEqual(self.service.value, self.value)

    def test_wrong_origin_has_no_cors_grant_and_cannot_write(self):
        status, headers, _ = self.request(origin="https://example.invalid")
        self.assertEqual(status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertFalse(self.target.exists())

    def test_wrong_token_cannot_write(self):
        self.assertEqual(self.request(token="wrong-token")[0], 403)
        self.assertFalse(self.target.exists())

    def test_oversized_and_unknown_storage_leave_saved_value_intact(self):
        self.assertEqual(self.request()[0], 200)
        before = self.target.read_bytes()
        self.assertEqual(self.request(raw=b"x" * 8193)[0], 413)
        self.assertEqual(self.request(raw=b"x" * 16384)[0], 413)
        bad = {**self.value, "storage": {"credentials": "must-not-be-stored"}}
        self.assertEqual(self.request(bad)[0], 400)
        self.assertEqual(self.target.read_bytes(), before)

    def test_profile_identity_and_bad_modes_are_rejected(self):
        self.assertEqual(self.request({**self.value, "profileId": "other-profile"})[0], 400)
        self.assertEqual(self.request({**self.value, "storage": {"labviz:v23:interface-mode:v1": "unknown"}})[0], 400)

    def test_corrupt_original_is_retained_when_replaced(self):
        self.service.close()
        self.target.write_bytes(b"invalid-old-content")
        self.service = module.Preferences(self.target, "http://127.0.0.1:42001", "test-profile")
        self.descriptor = self.service.descriptor()
        self.assertIsNotNone(self.service.warning)
        self.assertEqual(self.request()[0], 200)
        retained = list(self.target.parent.glob("preferences.json.invalid-*"))
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].read_bytes(), b"invalid-old-content")


if __name__ == "__main__":
    unittest.main()
