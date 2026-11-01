"""Failure-oriented tests for durable drafts; no scientific database is written."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'runtime/session.py'
spec = importlib.util.spec_from_file_location('desktop_session', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'session'
        self.bases = {'p1': {'research': 'r1', 'figure': 'f1', 'project': 'c1'}}
        self.session = module.Session(self.root, 'profile', lambda _: self.bases)

    def tearDown(self):
        self.temp.cleanup()

    def draft(self, identifier='a'):
        return {'format': 'labviz-desktop-checkpoint-v1', 'profileId': 'profile', 'route': '/workspace/p1?workspace=research&researchTask=figures', 'checkpointId': identifier,
                'entries': {'figure:p1': {'version': 1, 'projectId': 'p1', 'base': {'research': 'r1', 'figure': 'f1'}, 'value': {'title': '未保存标题', 'undoDrafts': [{'title': 'old'}], 'redoDrafts': []}}}}

    def save(self, value=None):
        value = value or self.draft()
        self.session.begin(value['checkpointId'])
        return self.session.commit(value)

    def test_restart_preserves_draft_undo_and_exact_source_bytes(self):
        data = 'sample,value\ns1,中文\n'.encode()
        staged = self.session.stage(data)
        value = self.draft()
        value['entries']['figure:p1']['value']['file'] = {'__labvizFile': staged['id'], 'name': '科研数据.csv', 'size': len(data), 'type': 'text/csv', 'lastModified': 123}
        self.save(value)
        restarted = module.Session(self.root, 'profile', lambda _: self.bases)
        self.assertEqual(restarted.initial, value)
        self.assertEqual(restarted.file(staged['id']), data)
        self.assertFalse((self.root / 'labviz.db').exists())

    def test_disk_full_never_acknowledges_or_replaces_last_good_checkpoint(self):
        self.save()
        before = self.session.path.read_bytes()
        self.session.begin('b')
        real_replace = module.os.replace
        def full(source, destination):
            if module.io_path(Path(destination)) == module.io_path(self.session.path):
                raise OSError(28, 'Disk full')
            real_replace(source, destination)
        with patch.object(module.os, 'replace', side_effect=full), self.assertRaises(OSError):
            self.session.commit(self.draft('b'))
        self.assertIsNone(self.session.acknowledged)
        self.assertEqual(self.session.path.read_bytes(), before)

    def test_permission_failure_keeps_current_checkpoint(self):
        self.save()
        before = self.session.path.read_bytes()
        self.session.begin('b')
        with patch.object(module, 'atomic', side_effect=PermissionError('Read only')), self.assertRaises(PermissionError):
            self.session.commit(self.draft('b'))
        self.assertEqual(self.session.path.read_bytes(), before)
        self.assertIsNone(self.session.acknowledged)

    def test_revision_conflict_and_deleted_project_are_not_silently_rebased(self):
        self.save()
        before = self.session.path.read_bytes()
        for actual in ({'research': 'new', 'figure': 'f1'}, None):
            self.bases['p1'] = actual
            self.session.begin('b')
            with self.assertRaisesRegex(ValueError, 'revision changed'):
                self.session.commit(self.draft('b'))
            self.assertEqual(self.session.path.read_bytes(), before)

    def test_corrupt_checkpoint_is_retained_and_never_restored(self):
        self.session.path.write_bytes(b'{broken original')
        restarted = module.Session(self.root, 'profile', lambda _: self.bases)
        self.assertIsNone(restarted.initial)
        self.assertIsNotNone(restarted.warning)
        self.assertEqual(self.session.path.read_bytes(), b'{broken original')
        restarted.begin('a')
        restarted.commit(self.draft())
        self.assertEqual(next(self.root.glob('checkpoint.invalid-*.json')).read_bytes(), b'{broken original')

    def test_cancelled_checkpoint_cannot_acknowledge_a_later_transition(self):
        self.session.begin('a')
        self.session.cancel()
        with self.assertRaises(ValueError):
            self.session.commit(self.draft())
        self.session.begin('b')
        with self.assertRaises(ValueError):
            self.session.commit(self.draft())
        self.assertIsNone(self.session.acknowledged)

    def test_corrupt_or_unavailable_staged_file_blocks_checkpoint(self):
        staged = self.session.stage(b'source')
        (self.session.files / staged['id']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.session.file(staged['id'])
        with self.assertRaises(ValueError):
            self.session.file('../outside')

    def test_profile_version_route_and_component_boundaries(self):
        for field, bad in [('profileId', 'other'), ('format', 'v99'), ('route', 'https://example.test/'), ('route', '//example.test/'), ('route', '/workspace/p1#credential')]:
            value = self.draft(); value[field] = bad
            self.session.begin('a')
            with self.assertRaises(ValueError):
                self.session.commit(value)
        value = self.draft(); value['entries']['figure:p1']['version'] = 99
        self.session.begin('a')
        with self.assertRaises(ValueError):
            self.session.commit(value)


    def test_settings_route_can_checkpoint_and_restore(self):
        value = self.draft(); value['route'] = '/settings'
        self.save(value)
        self.assertEqual(module.Session(self.root, 'profile', lambda _: self.bases).initial['route'], '/settings')

    def source(self, data):
        stored = self.session.stage(data)
        return {'__labvizFile': stored['id'], 'name': 'source.csv', 'size': len(data), 'type': 'text/csv', 'lastModified': 0}

    def test_reclaims_full_unreferenced_staging_after_durable_checkpoint(self):
        with patch.object(module, 'MAX_STAGING', 32):
            for n in range(4): self.session.stage(bytes([n]) * 8)
            with self.assertRaisesRegex(ValueError, 'staging is full'): self.session.stage(b'new')
            self.save()
            self.assertEqual(list(self.session.files.iterdir()), [])
            self.session.stage(b'new')

    def test_current_previous_and_retained_conflict_sources_are_protected(self):
        a=self.source(b'current'); value=self.draft('a'); value['entries']['figure:p1']['value']['file']=a; self.save(value)
        b=self.source(b'next'); value=self.draft('b'); value['entries']['figure:p1']['value']['file']=b; self.save(value)
        self.assertEqual(self.session.file(a['__labvizFile']), b'current')
        c=self.source(b'conflict'); value=self.draft('conflict'); value['entries']['figure:p1']['value']['file']=c
        self.bases['p1']['research']='new'
        with self.assertRaises(ValueError): self.save(value)
        self.bases['p1']['research']='r1'; self.session.stage(b'orphan')
        self.save(self.draft('empty-1')); self.save(self.draft('empty-2'))
        self.assertFalse((self.session.files/a['__labvizFile']).exists())
        self.assertFalse((self.session.files/b['__labvizFile']).exists())
        self.assertEqual(self.session.file(c['__labvizFile']), b'conflict')

    def test_failed_checkpoint_never_reclaims_unacknowledged_sources(self):
        self.save(); stored=self.source(b'recover after failed save'); original=module.atomic
        def fail(path,data):
            if path==self.session.path: raise OSError('disk full')
            return original(path,data)
        with patch.object(module,'atomic',side_effect=fail),self.assertRaises(OSError): self.save(self.draft('failed'))
        self.assertEqual(self.session.file(stored['__labvizFile']),b'recover after failed save')

    def test_corrupt_retained_snapshot_stops_cleanup_without_revoking_save(self):
        self.save(); retained=self.root/'checkpoint.invalid-retained.json'; retained.write_bytes(b'{original broken')
        stored=self.source(b'keep for recovery'); result=self.save(self.draft('next'))
        self.assertTrue(result['saved']); self.assertEqual(retained.read_bytes(),b'{original broken')
        self.assertEqual(self.session.file(stored['__labvizFile']),b'keep for recovery')

    def test_cleanup_refuses_unfamiliar_files_and_never_touches_outside(self):
        outside=Path(self.temp.name)/'outside'; outside.write_bytes(b'outside')
        import os
        link=self.session.files/('0'*64); os.link(outside,link)
        self.save()
        self.assertTrue(link.exists()); self.assertEqual(outside.read_bytes(),b'outside')

    def test_full_old_staging_can_recover_during_new_capture_without_dropping_active_files(self):
        self.save()
        with patch.object(module,'MAX_STAGING',32), patch.object(module,'MAX_CAPTURE_RESERVE',16):
            for n in range(4):self.session.stage(bytes([n])*8)
            self.session.begin('capture')
            x=self.source(b'xxxxxxxx');y=self.source(b'yyyyyyyy')
            self.assertEqual(len(list(self.session.files.iterdir())),6)
            with self.assertRaises(ValueError):self.source(b'beyond reserve')
            value=self.draft('capture');value['entries']['figure:p1']['value']['files']=[x,y]
            self.assertTrue(self.session.commit(value)['saved'])
            self.assertEqual(self.session.file(x['__labvizFile']),b'xxxxxxxx')
            self.assertEqual(self.session.file(y['__labvizFile']),b'yyyyyyyy')

    def test_staging_without_a_verified_snapshot_never_discards_failed_recovery_material(self):
        with patch.object(module,'MAX_STAGING',8):
            ref=self.source(b'original');self.session.begin('capture')
            with self.assertRaises(ValueError):self.source(b'new file')
            self.assertEqual(self.session.file(ref['__labvizFile']),b'original')

    def test_capture_reserve_never_reclaims_before_a_failed_atomic_write(self):
        self.save()
        with patch.object(module,'MAX_STAGING',16),patch.object(module,'MAX_CAPTURE_RESERVE',8):
            old=[self.source(bytes([n])*8) for n in range(2)]
            self.session.begin('reserved');new=self.source(b'newbytes')
            value=self.draft('reserved');value['entries']['figure:p1']['value']['file']=new
            original=module.atomic
            def fail(path,data):
                if path==self.session.path:raise OSError('Disk full')
                return original(path,data)
            with patch.object(module,'atomic',side_effect=fail),self.assertRaises(OSError):self.session.commit(value)
            for n,ref in enumerate(old):self.assertEqual(self.session.file(ref['__labvizFile']),bytes([n])*8)
            self.assertEqual(self.session.file(new['__labvizFile']),b'newbytes')


if __name__ == '__main__':
    unittest.main()
