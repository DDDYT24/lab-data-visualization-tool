"""Shutdown failures must never become successful suspension acknowledgements."""
import json
import io
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'V2.0/api'))
import controller
import service_runner as runner


class Child:
    pid = 12345

    def __init__(self, code=0, timeout=False):
        self.code, self.timeout, self.done = code, timeout, False

    def poll(self):
        return self.code if self.done else None

    def wait(self, timeout=None):
        if self.timeout:
            self.timeout = False
            raise subprocess.TimeoutExpired('owned test process', timeout)
        self.done = True
        return self.code

    def terminate(self):
        self.done = True


class ShutdownTests(unittest.TestCase):
    def test_system_or_scheduler_gap_requires_a_fresh_quiet_interval(self):
        self.assertFalse(controller.quiet_clock_resumed(None, 1000))
        self.assertFalse(controller.quiet_clock_resumed(10, 10.2))
        self.assertTrue(controller.quiet_clock_resumed(10, 70))
        self.assertTrue(controller.quiet_clock_resumed(10, 9))
    def test_runner_main_propagates_timeout_and_writes_session_bound_receipt(self):
        from labviz_api.installed_profile import InstalledProfile, APP_ID
        import labviz_api.installed_lifecycle as lifecycle
        for timeout in [False, True]:
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve(); payload = root / 'payload'; program = root / 'program'
                payload.mkdir(); program.mkdir(); backend = root / 'appdata/LabVizV23'
                (payload / 'candidate-build.json').write_text(json.dumps({'appId': APP_ID, 'candidateId': 'c' * 64, 'dataEpoch': 1}))
                (program / 'installation.json').write_text(json.dumps({'appId': APP_ID, 'format': 'labviz-installed-v1'}))
                with patch.dict(os.environ, {'LOCALAPPDATA': str(backend.parent)}):
                    profile = InstalledProfile.open(backend, create=True)
                config = root / 'runner.json'
                config.write_text(json.dumps({'payloadRoot': str(payload), 'programRoot': str(program), 'backendProfile': str(backend), 'candidateId': 'c' * 64}))
                api, web = Child(timeout=timeout), Child(); web.pid = 12346
                job = Mock()
                def close(): api.done = web.done = True
                job.close.side_effect = close
                class Response(io.BytesIO): status = 200
                def urlopen(request, **kwargs):
                    url = request if isinstance(request, str) else request.full_url
                    if url.endswith('/instance'):
                        return Response(json.dumps({'appId': APP_ID, 'profileId': profile.identifier, 'pid': api.pid}).encode())
                    if url.endswith('/shutdown'):
                        return Response(json.dumps({'accepted': True, 'appId': APP_ID, 'profileId': profile.identifier}).encode())
                    if url.endswith('/'):
                        (backend / 'logs/stop.request').write_text(json.dumps({'profileId': profile.identifier, 'sessionId': 'a' * 32}))
                    return Response(b'{}')
                with patch.object(runner.sys, 'argv', ['runner', str(config)]), patch.object(InstalledProfile, 'open', return_value=profile), \
                     patch.object(lifecycle, 'recover_profile'), patch.object(runner, 'Mutex', return_value=Mock()), \
                     patch.object(runner, 'sid', return_value='test'), patch.object(runner, 'protect', return_value=b'test'), \
                     patch.object(runner.uuid, 'uuid4', return_value=SimpleNamespace(hex='a' * 32)), \
                     patch.object(runner.subprocess, 'Popen', side_effect=[api, web]), patch.object(runner, 'OwnedJob', return_value=job), \
                     patch.object(runner, 'port_owned', return_value=True), patch.object(runner.urllib.request, 'urlopen', side_effect=urlopen), \
                     patch.object(runner, 'record', return_value={'pid': 1}):
                    if timeout:
                        with self.assertRaises(RuntimeError): runner.main()
                    else: runner.main()
                receipt = json.loads((backend / 'logs/desktop-service-exit.json').read_text())
                self.assertEqual(receipt['normal'], not timeout)
                self.assertEqual(receipt['sessionId'], 'a' * 32)
                self.assertEqual(receipt['forced'], timeout)
                self.assertFalse((backend / 'logs/running.json').exists())

    def test_late_maintenance_cancels_suspension_and_commit_fences_submissions(self):
        from maintenance import Maintenance
        import threading
        with tempfile.TemporaryDirectory() as folder:
            owner = Path(folder).resolve()
            (owner / 'desktop-profile.json').write_text(json.dumps({'format': 'labviz-desktop-profile-v1', 'appId': controller.DESKTOP_ID, 'root': str(owner)}))
            maintenance = Maintenance(owner, owner, owner)
            maintenance.submit({'action': 'backup'})
            self.assertFalse(maintenance.commit_suspension(lambda: True, False))
            maintenance.fail('cancel synthetic backup')
            entered, release = threading.Event(), threading.Event()
            def probe(): entered.set(); release.wait(timeout=3); return True
            thread = threading.Thread(target=lambda: maintenance.commit_suspension(probe, False)); thread.start()
            self.assertTrue(entered.wait(timeout=3))
            with __import__('concurrent.futures').futures.ThreadPoolExecutor() as pool:
                future = pool.submit(maintenance.submit, {'action': 'backup'})
                self.assertFalse(future.done()); release.set(); thread.join(timeout=3)
                with self.assertRaisesRegex(ValueError, 'Suspension is pending'): future.result(timeout=3)
            maintenance.reopen_suspension()
            self.assertEqual(maintenance.submit({'action': 'backup'})['state'], 'waiting')

    def test_api_acknowledgement_precedes_web_termination_and_job_close(self):
        events = []
        api, web = Child(), Child(code=1)
        original = web.terminate
        def terminate():
            self.assertTrue(api.done)
            events.append('web')
            original()
        web.terminate = terminate
        result = runner.shutdown_services(api, web, SimpleNamespace(close=lambda: events.append('job')), lambda: events.append('api'))
        self.assertTrue(result['apiGraceful'])
        self.assertFalse(result['forced'])
        self.assertTrue(result['servicesStopped'])
        self.assertEqual(events, ['api', 'web', 'job'])

    def test_api_timeout_nonzero_lost_ownership_and_early_web_exit_fail(self):
        for case in ['timeout', 'nonzero', 'ownership', 'web-exit']:
            with self.subTest(case=case):
                api, web = Child(code=9 if case == 'nonzero' else 0, timeout=case == 'timeout'), Child()
                web.done = case == 'web-exit'
                def shutdown():
                    if case == 'ownership':
                        raise ValueError('foreign listener')
                result = runner.shutdown_services(api, web, Mock(), shutdown)
                self.assertTrue(result['forced'])
                self.assertIsNotNone(result['error'])

    def receipt(self, ready):
        return {'format': 'labviz-desktop-service-exit-v1', 'profileId': ready['profileId'],
                'sessionId': ready['sessionId'], 'normal': True, 'apiGraceful': True,
                'forced': False, 'servicesStopped': True}

    def test_controller_requires_matching_normal_runner_receipt_and_no_running_record(self):
        for case in ['normal', 'missing', 'stale', 'nonzero', 'forced', 'running', 'api-timeout']:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); logs = root / 'logs'; logs.mkdir()
                ready = {'profileId': 'profile', 'sessionId': 'session'}
                value = self.receipt(ready)
                if case == 'stale': value['sessionId'] = 'old'
                if case == 'api-timeout': value['normal'] = False; value['apiGraceful'] = False
                if case != 'missing': (logs / 'desktop-service-exit.json').write_text(json.dumps(value))
                if case == 'running': (logs / 'running.json').write_text('{}')
                result = controller.stop_services(Child(code=7 if case == 'nonzero' else 0, timeout=case == 'forced'), ready, root, Mock(), logs, None)
                self.assertEqual(result['normal'], case == 'normal')

    def test_real_controller_loop_never_reports_abnormal_stop_as_suspended(self):
        for forced in [False, True]:
            with self.subTest(forced=forced), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve(); owner = root / 'owner'; payload = root / 'payload'
                payload.mkdir(); owner.mkdir()
                (payload / 'candidate-build.json').write_text(json.dumps({'appId': controller.APP_ID, 'candidateId': 'c' * 64, 'dataEpoch': 1}))
                (owner / 'desktop-profile.json').write_text(json.dumps({'format': 'labviz-desktop-profile-v1', 'appId': controller.DESKTOP_ID, 'root': str(owner)}))
                logs = owner / 'appdata/LabVizV23/logs'; logs.mkdir(parents=True)
                ready = {'launcher': {'pid': 12345}, 'candidateId': 'c' * 64, 'appId': controller.APP_ID,
                         'profileId': 'profile', 'sessionId': 'session', 'webUrl': 'http://127.0.0.1:1234'}
                (logs / 'running.json').write_text(json.dumps(ready))
                config = root / 'config.json'
                config.write_text(json.dumps({'format': 'labviz-desktop-session-v1', 'profileRoot': str(owner), 'payloadRoot': str(payload), 'candidateId': 'c' * 64, 'idleSeconds': 1}))
                commands = queue.Queue(); commands.put('hide'); events = []; ticks = iter(n / 10 for n in range(10000))
                child = Child(timeout=forced, code=17 if forced else 0)
                original_wait = child.wait
                def wait(timeout=None):
                    result = original_wait(timeout)
                    (logs / 'running.json').unlink(missing_ok=True)
                    (logs / 'desktop-service-exit.json').write_text(json.dumps(self.receipt(ready)))
                    return result
                child.wait = wait
                class Session:
                    initial = warning = error = acknowledged = None
                    def __init__(self, *args): pass
                    def descriptor(self): return {}
                    def idle(self): return True
                    def begin(self, value): self.acknowledged = value
                    def cancel(self): self.acknowledged = None
                class Api:
                    def __init__(self, *args): self.draining = False
                    def bases(self, *args): return {}
                    def figures(self, *args): return True
                    def call(self, action, *args):
                        if action == 'drain': self.draining = True
                        return {'idle': True, 'draining': self.draining, 'generation': 1}
                def emit(kind, **values):
                    events.append(kind)
                    if kind == 'suspend-commit': commands.put('webview-destroyed')
                    if kind == 'suspended': commands.put('stop')
                with patch.object(controller.sys, 'argv', ['test', str(config)]), patch.object(controller, 'AUDIT_PATH', None), \
                     patch.object(controller.subprocess, 'Popen', return_value=child), patch.object(controller, 'OwnedJob', return_value=Mock()), \
                     patch.object(controller, 'Preferences', return_value=SimpleNamespace(descriptor=lambda: {}, close=lambda: None)), \
                     patch.object(controller, 'Session', Session), patch.object(controller, 'ApiControl', Api), \
                     patch.object(controller, 'local_key', return_value='test'), patch.object(controller.queue, 'Queue', return_value=commands), \
                     patch.object(controller.threading, 'Thread', return_value=Mock()), patch.object(runner, 'Mutex', return_value=Mock()), \
                     patch.object(runner, 'sid', return_value='test'), patch.object(controller, 'emit', side_effect=emit), \
                     patch.object(controller.time, 'sleep'), patch.object(controller.time, 'monotonic', side_effect=lambda: next(ticks)):
                    if forced:
                        with self.assertRaisesRegex(RuntimeError, 'did not stop normally'): controller.main()
                    else:
                        controller.main()
                self.assertEqual('suspended' in events, not forced)
                self.assertEqual((owner / 'logs/desktop-suspended.json').exists(), not forced)
