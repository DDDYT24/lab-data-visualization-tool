import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('release_preflight_tests',ROOT/'desktop/scripts/release_preflight.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
POLICY=json.loads((ROOT/'desktop/contracts/resource-budget-v1.json').read_text(encoding='utf-8'))


class ReleasePreflightTests(unittest.TestCase):
    def record(self,root):
        evidence=root/'observation.json';evidence.write_text('{"fixtureOnly":true,"nativeAppLaunched":true}')
        digest=module.sha(evidence)
        return {'format':'labviz-desktop-release-record-v1','packageId':'p','setupSha256':'s','machine':{'id':'unit-fixture','operator':'unit-fixture'},
                'gates':[{'id':key,'scope':scope,'outcome':'pass','utcTime':'2026-10-25T00:00:00Z','actions':['synthetic fixture action'],
                          'expected':'fixture','actual':'fixture','evidenceSha256':{'observation.json':digest}} for key,scope in module.GATES.items()],
                'metrics':{'nativeHostPrivateCommitMiB':32,'completeSuspendedPrivateCommitMiB':100,'smallEditableRestoreSamplesMs':[5000]*10},
                'reports':{item['metric']:{'path':'observation.json','sha256':digest} for item in POLICY['requiredReports']}}

    def test_complete_synthetic_record_passes_structural_checks_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.assertEqual(module.evaluate_record(self.record(root),'p','s',root,POLICY),[])

    def test_an_unfilled_or_self_declared_gate_is_not_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=self.record(root);gate=record['gates'][0]
            gate.update(evidenceSha256={},utcTime=None,actions=[])
            codes={b['code'] for b in module.evaluate_record(record,'p','s',root,POLICY)}
            self.assertTrue({'EVIDENCE_MISSING','OBSERVATION_TIME_MISSING','OBSERVATION_DETAIL_MISSING'}<=codes)

    def test_identity_duplicates_and_wrong_scope_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=self.record(root);record['packageId']='other';record['gates'][0]['scope']='headless';record['gates'].append(copy.deepcopy(record['gates'][0]))
            codes={b['code'] for b in module.evaluate_record(record,'p','s',root,POLICY)}
            self.assertTrue({'ACCEPTANCE_IDENTITY_DIFFERS','GATE_SET_DIFFERS','OBSERVATION_SCOPE_DIFFERS'}<=codes)

    def test_missing_tampered_and_escaping_evidence_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=self.record(root)
            for name,digest in [('absent.txt','0'*64),('observation.json','0'*64),('../outside.txt','0'*64)]:
                record['gates'][0]['evidenceSha256']={name:digest}
                self.assertIn('EVIDENCE_INVALID',{b['code'] for b in module.evaluate_record(record,'p','s',root,POLICY)})

    def test_headless_json_and_jsonl_cannot_accept_native_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=self.record(root)
            for suffix in ['json','jsonl']:
                path=root/('headless.'+suffix);path.write_text('{"scope":"headless production integration","nativeAppLaunched":false}\n')
                record['gates'][0]['evidenceSha256']={path.name:module.sha(path)}
                self.assertIn('HEADLESS_EVIDENCE_CANNOT_ACCEPT_NATIVE',{b['code'] for b in module.evaluate_record(record,'p','s',root,POLICY)})
            path=root/'diagnostic-only.json';path.write_text('{"nativeAcceptance":false}')
            record['gates'][0]['evidenceSha256']={path.name:module.sha(path)}
            self.assertIn('HEADLESS_EVIDENCE_CANNOT_ACCEPT_NATIVE',{b['code'] for b in module.evaluate_record(record,'p','s',root,POLICY)})

    def test_native_limits_all_slow_samples_and_invalid_numbers_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for field,value in [('nativeHostPrivateCommitMiB',32.01),('completeSuspendedPrivateCommitMiB',100.01),('nativeHostPrivateCommitMiB',True),('nativeHostPrivateCommitMiB',float('nan')),('smallEditableRestoreSamplesMs',[4000]*9+[5001]),('smallEditableRestoreSamplesMs',[4000]*9)]:
                record=self.record(root);record['metrics'][field]=value
                codes={b['code'] for b in module.evaluate_record(record,'p','s',root,POLICY)}
                self.assertTrue(codes&{'NATIVE_METRIC_MISSING','RESTORE_SAMPLES_MISSING','NATIVE_BUDGET_EXCEEDED'},(field,value))

    def test_mandatory_reports_are_not_optional_and_malformed_records_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=self.record(root);record['reports']={}
            blockers=module.evaluate_record(record,'p','s',root,POLICY)
            self.assertEqual(len([x for x in blockers if x['code']=='MANDATORY_REPORT_MISSING']),6)
            self.assertTrue(module.evaluate_record([], 'p','s',root,POLICY))
            self.assertTrue(module.evaluate_record({'gates':[None]},'p','s',root,POLICY))
            record=self.record(root);headless=root/'headless-report.json';headless.write_text('{"scope":"headless resources"}')
            record['reports']['coldStart']={'path':headless.name,'sha256':module.sha(headless)}
            self.assertIn('MANDATORY_REPORT_IS_HEADLESS',{x['code'] for x in module.evaluate_record(record,'p','s',root,POLICY)})

    def test_signature_requires_valid_intended_identity_and_timestamp(self):
        self.assertEqual(module.signature_blockers({'status':'Valid','thumbprint':'ab','timestamped':True},'AB'),[])
        self.assertIn('SIGNER_IDENTITY_UNSPECIFIED',module.signature_blockers({'status':'Valid','thumbprint':'ab','timestamped':True},None))
        self.assertIn('SIGNER_IDENTITY_DIFFERS',module.signature_blockers({'status':'Valid','thumbprint':'other','timestamped':True},'ab'))
        self.assertIn('SIGNATURE_TIMESTAMP_MISSING',module.signature_blockers({'status':'Valid','thumbprint':'ab','timestamped':False},'ab'))
        self.assertIn('SETUP_UNSIGNED_OR_UNTRUSTED',module.signature_blockers({'status':'NotSigned','thumbprint':None},'ab'))

    def artifact(self,root):
        bundle=root/'bundle';bundle.mkdir();build=root/'build';build.mkdir()
        (bundle/'payload.txt').write_text('synthetic payload');setup=build/'private.exe';setup.write_bytes(b'synthetic setup; never executed')
        manifest={'format':'labviz-desktop-package-v1','appId':'org.labviz.desktop.preview','dataEpoch':2,'checkpointSchema':1,'files':{'payload.txt':module.sha(bundle/'payload.txt')}}
        identifier=module.hashlib.sha256(module.canonical(manifest)).hexdigest()
        (bundle/'desktop-package.json').write_text(json.dumps({'packageId':identifier,'manifest':manifest}))
        (build/'build-verification.json').write_text(json.dumps({'packageId':identifier,'compilerExitCode':0,'installers':{'private.exe':{'sha256':module.sha(setup),'bytes':setup.stat().st_size}}}))
        return bundle,build

    def test_artifact_hash_checks_detect_tampering_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle,build=self.artifact(Path(folder));module.verify_artifact(bundle,build)
            original=(bundle/'payload.txt').read_bytes();(bundle/'payload.txt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Bundle member differs'):module.verify_artifact(bundle,build)
            (bundle/'payload.txt').write_bytes(original);(bundle/'extra.txt').write_text('unlisted')
            with self.assertRaisesRegex(ValueError,'Unexpected'):module.verify_artifact(bundle,build)

    def test_artifact_setup_and_manifest_identity_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle,build=self.artifact(Path(folder));setup=build/'private.exe';original=setup.read_bytes();setup.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'Setup bytes differ'):module.verify_artifact(bundle,build)
            setup.write_bytes(original);path=bundle/'desktop-package.json';value=json.loads(path.read_text());value['packageId']='0'*64;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'Package identity differs'):module.verify_artifact(bundle,build)
