"""Argument handling of the committed release drivers; the scenarios themselves run on a candidate."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acceptance = load('release_acceptance')
upgrade = load('release_upgrade') if sys.platform == 'win32' else None


class AcceptanceArguments(unittest.TestCase):
    def test_paths_come_from_the_candidate_and_trial_root(self):
        with tempfile.TemporaryDirectory() as folder:
            args = acceptance.parse(['session', '--candidate', folder, '--label', 'p2bg2500', '--trial-root', folder+'/trials'])
            candidate = Path(folder).resolve()
            self.assertEqual(args.bundle, candidate/'bundle')
            self.assertEqual(args.build, candidate/'build')
            self.assertEqual(args.node, candidate/'bundle/app/runtime/node/node.exe')
            self.assertEqual(args.trial, candidate/'trials/p2bg2500')
            self.assertEqual(args.logs, candidate/'p2bg2500')
            self.assertEqual(args.sampler_python, Path(sys.executable))

    def test_an_explicit_node_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            node = Path(folder)/'node.exe'
            args = acceptance.parse(['large', '--candidate', folder, '--label', 'a', '--node', str(node)])
            self.assertEqual(args.node, node.resolve())

    def test_labels_are_plain_names(self):
        for label in ('../x', 'a b', 'a/b', ''):
            with self.assertRaises(SystemExit):
                acceptance.parse(['session', '--candidate', '.', '--label', label])

    def test_a_missing_node_stops_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(SystemExit) as stopped:
                acceptance.main(['session', '--candidate', folder, '--label', 'x'])
            self.assertIn('Node.js not found', str(stopped.exception))
            self.assertFalse((Path(folder)/'x').exists())


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
