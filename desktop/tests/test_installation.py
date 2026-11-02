"""Candidate identity, activation failures and repair preserve the original profile."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import test_maintenance
import installation as m


class InstallationTests(unittest.TestCase):
    setUp=test_maintenance.MaintenanceTests.setUp
    tearDown=test_maintenance.MaintenanceTests.tearDown

    def package(self, label, epoch=m.DATA_EPOCH):
        root=m.program(self.root/'program',create=True)
        content={'frozen.txt':hashlib.sha256(label.encode()).hexdigest()}
        manifest={'format':'labviz-desktop-package-v1','appId':m.DESKTOP_ID,'dataEpoch':epoch,'checkpointSchema':1,'files':content}
        identifier=hashlib.sha256(m.encoded(manifest)).hexdigest();folder=root/'versions'/identifier[:24];folder.mkdir(parents=True,exist_ok=True)
        (folder/'frozen.txt').write_bytes(label.encode());m.atomic(folder/'desktop-package.json',m.encoded({'packageId':identifier,'manifest':manifest}))
        return root,identifier,folder

    def test_changed_candidate_and_same_version_repair_activate_exact_package(self):
        root,first,_=self.package('first');m.mark(root,'current-candidate.json',first)
        _,second,_=self.package('second')
        for target in (second,second,first):
            before=(self.profile.data/'labviz.db').read_bytes();m.prepare(root,target,self.owner,self.payload);m.mark(root,'pending-candidate.json',target)
            with patch.object(m,'health',return_value={'servicesReady':True}):result=m.activate(root,self.owner,self.payload)
            self.assertEqual(result['packageId'],target);self.assertEqual((self.profile.data/'labviz.db').read_bytes(),before)
            self.assertEqual(m.pointer(root,'current-candidate.json'),target)

    def test_failed_health_restores_data_draft_and_previous_candidate(self):
        # The previous package may predate the current data epoch.
        root,first,_=self.package('first',epoch=1);m.mark(root,'current-candidate.json',first);_,second,_=self.package('second')
        checkpoint=(self.owner/'session/checkpoint.json').read_bytes();m.prepare(root,second,self.owner,self.payload);m.mark(root,'pending-candidate.json',second)
        def broken(*args):
            (self.owner/'preferences.json').write_bytes(b'failed-write')
            raise ValueError('candidate failure')
        with patch.object(m,'health',side_effect=broken):result=m.activate(root,self.owner,self.payload)
        self.assertTrue(result['rolledBack']);self.assertEqual(result['packageId'],first)
        self.assertEqual((self.owner/'session/checkpoint.json').read_bytes(),checkpoint)
        self.assertEqual(m.read(self.owner/'preferences.json'),self.preferences)
        self.assertFalse((root/'pending-candidate.json').exists())

    def test_changed_payload_triggers_previous_candidate_without_running_it(self):
        root,first,_=self.package('first');m.mark(root,'current-candidate.json',first);_,second,folder=self.package('second')
        m.prepare(root,second,self.owner,self.payload);m.mark(root,'pending-candidate.json',second);(folder/'frozen.txt').write_bytes(b'tampered')
        with patch.object(m,'health') as health:result=m.activate(root,self.owner,self.payload)
        health.assert_not_called();self.assertTrue(result['rolledBack'])

    def test_package_from_a_newer_epoch_is_refused(self):
        root,newer,_=self.package('newer',epoch=m.DATA_EPOCH+1)
        with self.assertRaisesRegex(ValueError,'format differs'):m.version(root,newer)

    def test_future_epoch_and_invalid_checkpoint_block_preparation(self):
        root,candidate,_=self.package('first')
        m.atomic(self.profile.root/'data-schema.json',m.encoded({'profileId':self.profile.identifier,'dataEpoch':3}))
        with self.assertRaisesRegex(ValueError,'newer'):m.prepare(root,candidate,self.owner,self.payload)
        self.assertFalse((root/'upgrade.json').exists())
        self.profile.set_epoch(1);(self.owner/'session/checkpoint.json').write_bytes(b'unsupported')
        with self.assertRaisesRegex(ValueError,'checkpoint'):m.prepare(root,candidate,self.owner,self.payload)

    def test_foreign_installation_or_path_escape_is_refused(self):
        foreign=self.root/'foreign';foreign.mkdir();(foreign/'keep').write_bytes(b'untouched')
        with self.assertRaises(ValueError):m.program(foreign,create=True)
        self.assertEqual((foreign/'keep').read_bytes(),b'untouched')
        root,candidate,_=self.package('first')
        with self.assertRaises(ValueError):m.version(root,'../escape')
        m.atomic(root/'current-candidate.json',m.encoded({'appId':'foreign','packageId':candidate}))
        with self.assertRaises(ValueError):m.pointer(root,'current-candidate.json')

    def test_no_pending_fast_path_does_not_touch_data_or_launch_services(self):
        root,candidate,_=self.package('first');m.mark(root,'current-candidate.json',candidate)
        before=(self.profile.data/'labviz.db').read_bytes()
        with patch.object(m,'health') as health:result=m.activate(root,self.owner,self.payload)
        health.assert_not_called();self.assertEqual(result['packageId'],candidate);self.assertEqual((self.profile.data/'labviz.db').read_bytes(),before)


if __name__=='__main__':unittest.main()
