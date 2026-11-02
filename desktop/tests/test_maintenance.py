"""Desktop backups include drafts and restore into a separate authenticated profile."""
import sys
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'desktop/runtime'))
sys.path.insert(0,str(ROOT/'V2.0/api'))
import maintenance as m
import installation as install
from labviz_api.installed_profile import InstalledProfile


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.owner=self.root/'中文资料';self.payload=ROOT/'outputs/desktop-d34/bundle/app'
        install.initialise(self.owner)
        self.env=patch.dict(os.environ,{'LOCALAPPDATA':str(self.owner/'appdata')});self.env.start()
        self.profile=InstalledProfile.open(self.owner/'appdata/LabVizV23',create=True)
        self.profile.data.mkdir();self.profile.set_epoch(1)
        with closing(sqlite3.connect(self.profile.data/'labviz.db')) as db:
            db.execute('CREATE TABLE scientific(id TEXT PRIMARY KEY,value REAL)');db.execute("INSERT INTO scientific VALUES ('immutable-source-id',4.5)")
            db.execute('CREATE TABLE projects(id TEXT PRIMARY KEY)')
            db.execute('CREATE TABLE revisions(id TEXT PRIMARY KEY)')
            db.commit()
        (self.profile.data/'objects').mkdir();(self.profile.data/'objects/frozen.pdf').write_bytes(b'%PDF-frozen-exact')
        (self.owner/'session/files').mkdir(parents=True)
        raw='样本,value\na,4.5\n'.encode();digest=hashlib.sha256(raw).hexdigest();(self.owner/'session/files'/digest).write_bytes(raw)
        snapshot={'format':'labviz-desktop-checkpoint-v1','profileId':self.profile.identifier,'checkpointId':'a'*32,'route':'/', 'entries':{'home':{'version':1,'projectId':None,'base':{},'value':{'file':{'__labvizFile':digest,'size':len(raw),'name':'数据.csv','type':'text/csv','lastModified':1}}}}}
        m.atomic(self.owner/'session/checkpoint.json',m.encoded({'format':'labviz-desktop-checkpoint-envelope-v1','payload':snapshot,'sha256':hashlib.sha256(m.encoded(snapshot)).hexdigest()}))
        self.preferences={'format':'labviz-desktop-preferences-v1','profileId':self.profile.identifier,'locale':'zh','storage':{}}
        m.atomic(self.owner/'preferences.json',m.encoded(self.preferences))
        (self.profile.root/'access.dpapi').write_bytes(b'not-backed-up')
        (self.owner/'webview').mkdir();(self.owner/'webview/cookies').write_bytes(b'not-backed-up')

    def tearDown(self):
        self.env.stop();self.temp.cleanup()

    def make_backup(self):
        return Path(m.backup(self.owner,self.payload)['backupDirectory'])

    def test_full_backup_and_independent_restore_keep_frozen_bytes_and_drafts(self):
        before=(self.profile.data/'labviz.db').read_bytes();directory=self.make_backup()
        manifest=m.verify(directory,self.payload)
        self.assertEqual(manifest['checkpointSchema'],1)
        self.assertFalse(any('dpapi' in name or 'webview' in name for name in manifest['files']))
        result=m.restore(directory,self.owner,self.payload);target=Path(result['restoredProfileRoot'])
        self.assertNotEqual(result['restoredProfileId'],self.profile.identifier)
        self.assertEqual((target/'appdata/LabVizV23/data/objects/frozen.pdf').read_bytes(),b'%PDF-frozen-exact')
        self.assertEqual(m.read(target/'session/checkpoint.json')['payload']['profileId'],result['restoredProfileId'])
        self.assertEqual(m.read(target/'preferences.json')['profileId'],result['restoredProfileId'])
        self.assertEqual((self.profile.data/'labviz.db').read_bytes(),before)
        self.assertEqual(len(list((target/'session/files').iterdir())),1)

    def test_corrupt_or_extra_backup_member_is_rejected_without_mutating_data(self):
        directory=self.make_backup();before=(self.profile.data/'labviz.db').read_bytes()
        (directory/'desktop/extra').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'differ'):m.verify(directory,self.payload)
        self.assertEqual((self.profile.data/'labviz.db').read_bytes(),before)

    def test_running_writer_blocks_backup(self):
        with self.profile.lock():
            with self.assertRaises(ValueError):self.make_backup()

    def cache_fixture(self):
        cache = self.profile.root/'cache/matplotlib'
        (cache/'nested').mkdir(parents=True)
        (cache/'fontlist.json').write_bytes(b'regenerable-font-list')
        (cache/'nested/temporary').write_bytes(b'regenerable')
        (self.profile.root/'cache/unknown').mkdir()
        (self.profile.root/'cache/unknown/keep').write_bytes(b'unknown-owner')
        return cache

    def test_clear_cache_preserves_all_durable_profile_content_and_is_repeatable(self):
        cache = self.cache_fixture(); self.make_backup()
        before = {p: m.digest(p) for p in m.files(self.owner)
                  if not p.resolve().is_relative_to(cache.resolve())}
        jobs=m.Maintenance(self.owner,self.owner,self.payload);jobs.submit({'action':'clear-cache'})
        result=jobs.execute()
        self.assertEqual(result['state'],'completed');self.assertEqual(result['removedFiles'],2)
        self.assertGreater(result['removedBytes'],0);self.assertFalse(cache.exists())
        for path, expected in before.items():
            self.assertEqual(m.digest(path),expected,str(path))
        self.assertEqual(m.clear_cache(self.owner,self.payload)['removedFiles'],0)

    def test_live_writer_and_future_epoch_refuse_cache_cleanup_before_deletion(self):
        cache=self.cache_fixture()
        with self.profile.lock(),self.assertRaises(ValueError):m.clear_cache(self.owner,self.payload)
        self.assertTrue((cache/'fontlist.json').exists())
        schema=self.profile.root/'data-schema.json';original=schema.read_bytes()
        try:
            m.atomic(schema,m.encoded({'profileId':self.profile.identifier,'dataEpoch':3}))
            with self.assertRaisesRegex(ValueError,'compatible'):m.clear_cache(self.owner,self.payload)
            self.assertTrue((cache/'fontlist.json').exists())
        finally:schema.write_bytes(original)

    def test_empty_cache_junction_is_rejected_before_any_deletion(self):
        cache=self.cache_fixture();outside=self.root/'external-cache';outside.mkdir()
        (outside/'keep').write_bytes(b'foreign')
        link=cache/'empty-junction'
        __import__('subprocess').run(['cmd','/c','mklink','/J',str(link),str(outside)],capture_output=True,check=True)
        try:
            with self.assertRaisesRegex(ValueError,'Linked'):m.clear_cache(self.owner,self.payload)
            self.assertTrue((cache/'fontlist.json').exists());self.assertEqual((outside/'keep').read_bytes(),b'foreign')
        finally:link.rmdir()

    def test_cache_deletion_failure_is_not_reported_as_completed(self):
        cache=self.cache_fixture();db=(self.profile.data/'labviz.db').read_bytes()
        jobs=m.Maintenance(self.owner,self.owner,self.payload);jobs.submit({'action':'clear-cache'})
        original=Path.unlink
        def refuse(path,*args,**kwargs):
            if path.name=='fontlist.json':raise PermissionError('cache file locked')
            return original(path,*args,**kwargs)
        with patch.object(Path,'unlink',refuse):result=jobs.execute()
        self.assertEqual(result['state'],'failed');self.assertIn('locked',result['message'])
        self.assertEqual((self.profile.data/'labviz.db').read_bytes(),db)
        self.assertTrue((cache/'fontlist.json').exists())

    def test_corrupt_checkpoint_is_preserved_but_not_silently_restored(self):
        (self.owner/'session/checkpoint.json').write_bytes(b'corrupt-retained')
        directory=self.make_backup();self.assertEqual((directory/'desktop/session/checkpoint.json').read_bytes(),b'corrupt-retained')
        with self.assertRaisesRegex(ValueError,'warning'):m.restore(directory,self.owner,self.payload)

    def test_unknown_format_and_future_epoch_fail_closed(self):
        directory=self.make_backup();envelope=m.read(directory/'desktop-backup.json');envelope['manifest']['dataEpoch']=3
        envelope['sha256']=hashlib.sha256(m.encoded(envelope['manifest'])).hexdigest();m.atomic(directory/'desktop-backup.json',m.encoded(envelope))
        with self.assertRaisesRegex(ValueError,'epoch'):m.restore(directory,self.owner,self.payload)

    def test_pending_request_cannot_be_replaced_and_paths_are_not_commands(self):
        jobs=m.Maintenance(self.owner,self.owner,self.payload)
        for value in [{'action':'shell','path':'cmd.exe'},{'action':'restore','backupId':'../outside'},{'action':'backup','source':'C:/'}]:
            with self.assertRaises(ValueError):jobs.submit(value)
        first=jobs.submit({'action':'backup'})
        with self.assertRaises(ValueError):jobs.submit({'action':'backup'})
        self.assertEqual(jobs.pending['requestId'],first['requestId'])

    def test_write_failure_never_reports_completed(self):
        jobs=m.Maintenance(self.owner,self.owner,self.payload);jobs.submit({'action':'backup'})
        with patch.object(m,'backup',side_effect=OSError('disk full')):result=jobs.execute()
        self.assertEqual(result['state'],'failed');self.assertIn('disk full',result['message'])

    def test_linked_backup_directory_cannot_escape_profile(self):
        outside=self.root/'outside';outside.mkdir()
        link=self.owner/'backups';subprocess=__import__('subprocess')
        subprocess.run(['cmd','/c','mklink','/J',str(link),str(outside)],capture_output=True,check=True)
        try:
            with self.assertRaises(ValueError):self.make_backup()
            self.assertEqual(list(outside.iterdir()),[])
        finally:link.rmdir()

    def test_chinese_paths_beyond_260_characters_keep_exact_bytes(self):
        target=self.owner/('路径'*50)/('科研'*50)/'文件.json'
        m.io_path(target.parent).mkdir(parents=True)
        try:
            raw='完整数据'.encode();m.atomic(target,raw)
            self.assertGreater(len(str(target)),260)
            self.assertEqual(m.digest(target),hashlib.sha256(raw).hexdigest())
        finally:
            if m.io_path(target).exists():m.io_path(target).unlink()
            m.io_path(target.parent).rmdir()

    def test_incomplete_restore_cannot_be_selected(self):
        directory=self.make_backup();before=(self.profile.data/'labviz.db').read_bytes()
        with patch.object(m.shutil,'copy2',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):m.restore(directory,self.owner,self.payload)
        target=next((self.owner/'restored-profiles').iterdir())
        with self.assertRaises((OSError,ValueError)):m.restored_copy(self.owner,target.name)
        self.assertEqual((self.profile.data/'labviz.db').read_bytes(),before)


    def test_exit_fences_new_maintenance_and_cancel_reopens_it(self):
        jobs=m.Maintenance(self.owner,self.owner,self.payload)
        self.assertEqual(jobs.check_exit(lambda busy:'exit-waiting' if busy else 'exit-ready'),'exit-ready')
        with self.assertRaisesRegex(ValueError,'Exit is pending'):jobs.submit({'action':'backup'})
        self.assertIsNone(jobs.pending);jobs.reopen_exit();jobs.submit({'action':'backup'})
        self.assertEqual(jobs.check_exit(lambda busy:'exit-waiting' if busy else 'exit-ready'),'exit-waiting')

    def test_maintenance_submission_cannot_race_the_exit_fence(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        jobs=m.Maintenance(self.owner,self.owner,self.payload);entered=threading.Event();release=threading.Event()
        def probe(busy):entered.set();assert release.wait(5);return 'exit-ready'
        with ThreadPoolExecutor(max_workers=2) as pool:
            fenced=pool.submit(jobs.check_exit,probe);self.assertTrue(entered.wait(5))
            submission=pool.submit(jobs.submit,{'action':'backup'});release.set()
            self.assertEqual(fenced.result(timeout=5),'exit-ready')
            with self.assertRaisesRegex(ValueError,'Exit is pending'):submission.result(timeout=5)
        self.assertIsNone(jobs.pending)

if __name__=='__main__':unittest.main()
