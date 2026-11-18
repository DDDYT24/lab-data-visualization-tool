"""Candidate identity, activation failures and repair preserve the original profile."""
import hashlib
import json
import sqlite3
from contextlib import closing
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

    def upgraded_profile(self):
        root, first, _ = self.package('epoch-one', epoch=1)
        m.mark(root, 'current-candidate.json', first)
        _, second, _ = self.package('epoch-two')
        original = self.scientific_content()
        original_desktop = self.desktop_bytes()
        m.prepare(root, second, self.owner, self.payload)
        m.mark(root, 'pending-candidate.json', second)
        with patch.object(m, 'health', return_value={'servicesReady': True}):
            m.activate(root, self.owner, self.payload)
        self.profile.set_epoch(2)
        with closing(sqlite3.connect(self.profile.data/'labviz.db')) as db:
            db.execute("INSERT INTO scientific VALUES ('work-created-after-upgrade', 9.5)")
            db.commit()
        source = b'new sample,value\nafter-upgrade,9.5\n'
        digest = hashlib.sha256(source).hexdigest()
        (self.owner/'session/files'/digest).write_bytes(source)
        envelope = m.read(self.owner/'session/checkpoint.json')
        entry = envelope['payload']['entries']['home']['value']['file']
        entry.update(__labvizFile=digest, size=len(source), name='after-upgrade.csv')
        envelope['sha256'] = hashlib.sha256(m.encoded(envelope['payload'])).hexdigest()
        m.atomic(self.owner/'session/checkpoint.json', m.encoded(envelope))
        return root, first, second, original, original_desktop, digest

    def desktop_bytes(self, owner=None):
        owner = owner or self.owner
        return {str(p.relative_to(owner)): p.read_bytes()
                for name in ('session', 'preferences.json')
                for p in ([owner/name] if (owner/name).is_file()
                          else (owner/name).rglob('*')) if p.is_file()}

    def scientific_content(self, data=None):
        data = data or self.profile.data
        with closing(sqlite3.connect(data/'labviz.db')) as db:
            rows = db.execute('SELECT id, value FROM scientific ORDER BY id').fetchall()
        return rows, {str(p.relative_to(data/'objects')): p.read_bytes()
                      for p in (data/'objects').rglob('*') if p.is_file()}

    def test_epoch_rollback_restores_old_snapshot_and_retains_all_new_work(self):
        root, first, _, original, original_desktop, digest = self.upgraded_profile()
        upgraded = self.scientific_content()
        upgraded_desktop = self.desktop_bytes()
        def healthy(*_):
            self.assertEqual(self.profile.epoch(), 1)
            self.assertEqual(self.scientific_content(), original)
            self.assertEqual(self.desktop_bytes(), original_desktop)
        with patch.object(m, 'health', side_effect=healthy):
            result = m.rollback(root, self.owner, self.payload)
        self.assertTrue(result['explicitRollback'])
        self.assertEqual(m.pointer(root, 'current-candidate.json'), first)
        self.assertFalse((root/'pending-candidate.json').exists())
        self.assertEqual(self.scientific_content(Path(result['retainedDataDirectory'])), upgraded)
        self.assertEqual(self.desktop_bytes(Path(result['retainedDesktopDirectory'])), upgraded_desktop)
        self.assertTrue((Path(result['retainedDesktopDirectory'])/'session/files'/digest).is_file())
        saved = m.owned_backup(self.owner, result['retainedCurrentBackup'], self.payload)
        self.assertEqual(m.read(saved/'desktop-backup.json')['manifest']['dataEpoch'], 2)
        self.assertEqual(self.scientific_content(saved/'data'), upgraded)

    def test_failed_old_runtime_restores_upgraded_database_and_complete_desktop(self):
        root, _, second, _, _, _ = self.upgraded_profile()
        upgraded = self.scientific_content()
        desktop = self.desktop_bytes()
        with patch.object(m, 'health', side_effect=ValueError('old runtime failed')):
            result = m.rollback(root, self.owner, self.payload)
        self.assertTrue(result['rolledBack'])
        self.assertEqual(result['packageId'], second)
        self.assertEqual(self.profile.epoch(), 2)
        self.assertEqual(self.scientific_content(), upgraded)
        self.assertEqual(self.desktop_bytes(), desktop)

    def test_desktop_swap_failure_recovers_the_current_data_and_staged_sources(self):
        root, _, second, _, _, _ = self.upgraded_profile()
        upgraded = self.scientific_content()
        desktop = self.desktop_bytes()
        replace = m.os.replace
        failed = []
        def refuse_once(source, target):
            source = Path(source)
            if source.parent.name.startswith('desktop-staging-') and source.name == 'session' and not failed:
                failed.append(True)
                raise PermissionError('desktop state target is locked')
            return replace(source, target)
        with patch.object(m.os, 'replace', side_effect=refuse_once), patch.object(m, 'health') as health:
            result = m.rollback(root, self.owner, self.payload)
        health.assert_not_called()
        self.assertTrue(result['rolledBack'])
        self.assertEqual(m.pointer(root, 'current-candidate.json'), second)
        self.assertEqual(self.profile.epoch(), 2)
        self.assertEqual(self.scientific_content(), upgraded)
        self.assertEqual(self.desktop_bytes(), desktop)

    def test_rollback_refuses_foreign_owner_backup_and_wrong_epoch_before_mutation(self):
        root, _, second, _, _, _ = self.upgraded_profile()
        original_receipt = m.read(root/'last-upgrade.json')
        database = (self.profile.data/'labviz.db').read_bytes()
        desktop = self.desktop_bytes()
        cases = []
        foreign = dict(original_receipt, profileRoot=str(self.root/'another-owner'))
        cases.append(foreign)
        foreign = dict(original_receipt, backup=dict(original_receipt['backup'], backupDirectory=str(self.root/'outside')))
        cases.append(foreign)
        newer = dict(original_receipt, backup=m.backup(self.owner, self.payload))
        cases.append(newer)
        for receipt in cases:
            m.atomic(root/'last-upgrade.json', m.encoded(receipt))
            with self.assertRaises(ValueError):
                m.rollback(root, self.owner, self.payload)
            self.assertEqual((self.profile.data/'labviz.db').read_bytes(), database)
            self.assertEqual(self.desktop_bytes(), desktop)
            self.assertEqual(m.pointer(root, 'current-candidate.json'), second)
            self.assertFalse((root/'upgrade.json').exists())
        m.atomic(root/'last-upgrade.json', m.encoded(original_receipt))

    def test_rollback_refuses_tampered_snapshot_and_an_unrelated_pending_transaction(self):
        root, _, second, _, _, _ = self.upgraded_profile()
        receipt = m.read(root/'last-upgrade.json')
        saved = Path(receipt['backup']['backupDirectory'])
        original = (saved/'desktop/preferences.json').read_bytes()
        (saved/'desktop/preferences.json').write_bytes(b'changed backup')
        with self.assertRaisesRegex(ValueError, 'differ'):
            m.rollback(root, self.owner, self.payload)
        (saved/'desktop/preferences.json').write_bytes(original)
        m.mark(root, 'pending-candidate.json', second)
        with self.assertRaisesRegex(ValueError, 'existing desktop installation'):
            m.rollback(root, self.owner, self.payload)
        self.assertEqual(self.profile.epoch(), 2)

    def test_rollback_can_resume_its_prepared_transaction(self):
        root, first, _, original, _, _ = self.upgraded_profile()
        receipt = m.read(root/'last-upgrade.json')
        journal = m.prepare(root, first, self.owner, self.payload)
        journal['rollbackSource'] = receipt['backup']
        m.atomic(root/'upgrade.json', m.encoded(journal))
        with patch.object(m, 'health', return_value={'servicesReady': True}):
            result = m.rollback(root, self.owner, self.payload)
        self.assertTrue(result['explicitRollback'])
        self.assertEqual(self.profile.epoch(), 1)
        self.assertEqual(self.scientific_content(), original)

    def test_same_version_repair_preserves_the_cross_epoch_rollback_snapshot(self):
        root, first, second, original, original_desktop, _ = self.upgraded_profile()
        receipt = (root/'last-upgrade.json').read_bytes()
        m.prepare(root, second, self.owner, self.payload)
        m.mark(root, 'pending-candidate.json', second)
        with patch.object(m, 'health', return_value={'servicesReady': True}):
            m.activate(root, self.owner, self.payload)
        self.assertEqual((root/'last-upgrade.json').read_bytes(), receipt)
        with patch.object(m, 'health', return_value={'servicesReady': True}):
            result = m.rollback(root, self.owner, self.payload)
        self.assertTrue(result['explicitRollback'])
        self.assertEqual(m.pointer(root, 'current-candidate.json'), first)
        self.assertEqual(self.scientific_content(), original)
        self.assertEqual(self.desktop_bytes(), original_desktop)


if __name__=='__main__':unittest.main()
