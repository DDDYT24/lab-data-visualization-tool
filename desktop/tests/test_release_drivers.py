"""Arguments and stage logic of the committed release drivers; the scenarios run on a candidate."""
import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acceptance = load('release_acceptance')
upgrade = load('release_upgrade') if sys.platform == 'win32' else None


class AcceptanceArguments(unittest.TestCase):
    def test_paths_come_from_the_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            args = acceptance.parse(['session', '--candidate', folder, '--label', 'p2bg2500'])
            candidate = Path(folder).resolve()
            self.assertEqual(args.node, candidate/'bundle/app/runtime/node/node.exe')
            self.assertEqual(args.trial, Path.home()/'LabVizD2/p2bg2500')
            self.assertEqual(args.logs, candidate/'p2bg2500')

    def test_labels_are_plain_names_outside_the_frozen_folders(self):
        for label in ('../x', 'a b', 'a/b', '', 'bundle', 'build', 'provenance'):
            with self.subTest(label), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                acceptance.parse(['session', '--candidate', '.', '--label', label])

    def test_a_missing_node_stops_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(SystemExit, 'Node.js not found'):
                acceptance.main(['session', '--candidate', folder, '--label', 'x'])
            self.assertFalse((Path(folder)/'x').exists())

    def test_drivers_refuse_optimized_python(self):
        for name in ('release_acceptance', 'release_upgrade') if upgrade else ('release_acceptance',):
            result = subprocess.run([sys.executable, '-B', '-O', str(SCRIPTS/(name+'.py'))], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('run without -O', result.stderr)


class SessionStage(unittest.TestCase):
    """The stage logic with a fake host and fake steps."""
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.candidate = self.root/'candidate'
        for name in ('bundle', 'build'):
            (self.candidate/name).mkdir(parents=True)
        self.write(self.candidate/'bundle/desktop-package.json', {'packageId': 'P'})
        self.write(self.candidate/'build/candidate.json', {'packageId': 'P', 'sourceCommit': 'C'})
        self.trial = self.root/'trials/t'
        self.trial.mkdir(parents=True)
        self.write(self.trial/'session.json', {'id': 'S'})
        self.write(self.trial/'exit.json', {'hostExitCode': 0, 'survivingOwnedProcesses': [], 'runId': 'R'})
        self.report(15, 17)
        self.codes, self.exit_requested = {}, False
        for patch in (mock.patch.object(acceptance, 'TRIALS', self.root/'trials'),
                      mock.patch.object(acceptance.subprocess, 'Popen', self.popen),
                      mock.patch.object(acceptance.subprocess, 'run', self.run_step),
                      contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            self.enterContext(patch)
        self.args = acceptance.parse(['session', '--candidate', str(self.candidate), '--label', 't'])

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def report(self, checks, exports, passed=True):
        self.write(self.trial/'workflow-full.json', {'passed': passed, 'checks': [0]*checks, 'exports': [0]*exports})

    def popen(self, command, stdout=None, **_):
        stdout.write(b'{"ready":true}')
        stdout.flush()

        def wait(timeout):
            self.exit_requested = json.loads((self.trial/'commands.json').read_text())['action'] == 'exit'
        host = mock.Mock(pid=1)
        host.poll.side_effect = lambda: 0 if self.exit_requested else None
        host.wait.side_effect = wait
        return host

    def run_step(self, command, **_):
        code = next((value for key, value in self.codes.items() if any(key in part for part in command)), 0)
        if code == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        return mock.Mock(returncode=code)

    @property
    def evidence(self):
        return self.candidate/'background-evidence.json'

    def test_a_passing_session_writes_the_measured_evidence(self):
        acceptance.session(self.args)
        value = json.loads(self.evidence.read_text(encoding='utf-8'))
        self.assertEqual((value['packageId'], value['sourceCommit'], value['scientificChecks'], value['frozenExports']),
                         ('P', 'C', 15, 17))
        self.assertTrue(self.exit_requested)

    def test_any_failure_leaves_no_evidence_and_ends_the_host(self):
        cases = {'few-checks': lambda: self.report(3, 17),
                 'not-passed': lambda: self.report(15, 17, False),
                 'workflow-step': lambda: self.codes.update({'cdp_workflow': 1}),
                 'workflow-timeout': lambda: self.codes.update({'cdp_workflow': 'timeout'}),
                 'headless-step': lambda: self.codes.update({'part2_headless': 1})}
        for name, arrange in cases.items():
            with self.subTest(name):
                self.report(15, 17)
                self.codes, self.exit_requested = {}, False
                arrange()
                self.args.logs = self.candidate/name
                with self.assertRaises(SystemExit):
                    acceptance.session(self.args)
                self.assertFalse(self.evidence.exists())
                self.assertTrue(self.exit_requested)

    def test_a_surviving_process_fails_the_session(self):
        self.write(self.trial/'exit.json', {'hostExitCode': 0, 'survivingOwnedProcesses': [9], 'runId': 'R'})
        with self.assertRaisesRegex(SystemExit, 'did not exit cleanly'):
            acceptance.session(self.args)
        self.assertFalse(self.evidence.exists())

    def test_existing_evidence_of_the_package_is_kept(self):
        self.write(self.evidence, {'packageId': 'P', 'kept': True})
        acceptance.session(self.args)
        self.assertTrue(json.loads(self.evidence.read_text())['kept'])

    def test_evidence_or_a_freeze_of_another_package_stops_before_the_host(self):
        self.write(self.evidence, {'packageId': 'other'})
        with self.assertRaisesRegex(SystemExit, 'is for package other'):
            acceptance.session(self.args)
        self.evidence.unlink()
        self.write(self.candidate/'build/candidate.json', {'packageId': 'other', 'sourceCommit': 'C'})
        with self.assertRaisesRegex(SystemExit, 'is for package other'):
            acceptance.session(self.args)
        self.assertFalse(self.args.logs.exists())

    def test_the_installer_needs_a_passed_session_and_a_compiled_setup(self):
        self.codes = {'part2_profile': 1}
        with self.assertRaises(SystemExit):
            acceptance.session(self.args)
        with self.assertRaisesRegex(SystemExit, 'passed session'):
            acceptance.installer(self.args)
        self.codes, self.exit_requested, self.args.logs = {}, False, self.candidate/'second'
        acceptance.session(self.args)
        with self.assertRaisesRegex(SystemExit, 'compile the Setup first'):
            acceptance.installer(self.args)
        self.write(self.candidate/'build/build-verification.json', {})
        self.assertEqual(acceptance.installer(self.args).results['installer-failure']['exitCode'], 0)


@unittest.skipUnless(upgrade, 'the upgrade driver reads the Windows registry')
class UpgradeArtifact(unittest.TestCase):
    def make(self, folder, epoch, build_epoch=None):
        build, bundle = Path(folder)/'build', Path(folder)/'bundle'
        (bundle/'app').mkdir(parents=True)
        build.mkdir()
        manifest = {'appId': upgrade.APP_ID, 'dataEpoch': epoch, 'files': {}}
        package = hashlib.sha256(upgrade.encoded(manifest)).hexdigest()
        (bundle/'desktop-package.json').write_text(json.dumps({'packageId': package, 'manifest': manifest}), encoding='utf-8')
        (bundle/'app/candidate-build.json').write_text(json.dumps({'dataEpoch': build_epoch or epoch}), encoding='utf-8')
        setup = build/'Setup.exe'
        setup.write_bytes(b'setup')
        record = {'compilerExitCode': 0, 'packageId': package,
                  'installers': {'Setup.exe': {'sha256': hashlib.sha256(b'setup').hexdigest(), 'bytes': 5}}}
        (build/'build-verification.json').write_text(json.dumps(record), encoding='utf-8')
        return build, package

    def test_the_epoch_is_read_from_the_package_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            build, package = self.make(folder, 4)
            item = upgrade.artifact(build)
            self.assertEqual(item['epoch'], 4)
            self.assertEqual(item['record']['packageId'], package)

    def test_a_bundle_whose_build_record_disagrees_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            build, _ = self.make(folder, 4, build_epoch=3)
            with self.assertRaises(AssertionError):
                upgrade.artifact(build)

    def test_a_changed_setup_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            build, _ = self.make(folder, 3)
            (build/'Setup.exe').write_bytes(b'other')
            with self.assertRaises(AssertionError):
                upgrade.artifact(build)


if __name__ == '__main__':
    unittest.main()
