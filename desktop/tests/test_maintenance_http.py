"""The desktop maintenance bridge rejects foreign origins, tokens and arbitrary paths."""
import json
import urllib.request
import urllib.error
import unittest
import test_maintenance
import maintenance
import installation
from preferences import Preferences

class MaintenanceHttpTests(unittest.TestCase):
    def setUp(self):
        test_maintenance.MaintenanceTests.setUp(self)
        self.service=Preferences(self.owner/'preferences.json','http://127.0.0.1:42005',self.profile.identifier)
        self.service.maintenance=maintenance.Maintenance(self.owner,self.owner,self.payload)
        self.bridge=self.service.descriptor()

    def tearDown(self):
        self.service.close();test_maintenance.MaintenanceTests.tearDown(self)

    def request(self,path,value=None,origin=None,token=None):
        url=self.bridge['url'].removesuffix('/preferences')+path
        headers={'Origin':origin or self.service.origin,'X-LabViz-Desktop':token or self.bridge['token'],'Content-Type':'application/json'}
        req=urllib.request.Request(url,headers=headers,data=None if value is None else json.dumps(value).encode())
        try:
            with urllib.request.urlopen(req,timeout=5) as response:return response.status,dict(response.headers),json.load(response)
        except urllib.error.HTTPError as response:return response.code,dict(response.headers),json.load(response)

    def test_foreign_origin_and_token_cannot_start_or_inspect_jobs(self):
        status,headers,_=self.request('/maintenance/submit',{'action':'backup'},origin='https://example.invalid')
        self.assertEqual(status,403);self.assertNotIn('Access-Control-Allow-Origin',headers)
        self.assertEqual(self.request('/maintenance/status',token='invalid')[0],403)
        self.assertIsNone(self.service.maintenance.pending)

    def test_only_fixed_operations_and_owned_backup_identifiers_are_accepted(self):
        self.assertEqual(self.request('/maintenance/submit',{'action':'restore','backupId':'C:/other'})[0],409)
        self.assertEqual(self.request('/maintenance/submit',{'action':'run','command':'cmd.exe'})[0],409)
        self.assertEqual(self.request('/maintenance/submit',{'action':'backup'})[0],202)
        self.assertEqual(self.request('/maintenance/submit',{'action':'backup'})[0],409)

    def test_partial_copy_cannot_be_selected_and_default_selection_is_explicit(self):
        self.assertEqual(self.request('/maintenance/select-copy',{'copyId':'a'*32})[0],409)
        self.assertFalse((self.owner/'profile-selection.json').exists())
        self.assertEqual(self.request('/maintenance/select-copy',{'copyId':None})[0],200)

    def test_all_completed_copies_remain_selectable_after_later_operations(self):
        directory=test_maintenance.MaintenanceTests.make_backup(self)
        first=maintenance.restore(directory,self.owner,self.payload)
        second=maintenance.restore(directory,self.owner,self.payload)
        before=(self.profile.data/'labviz.db').read_bytes()
        for item in (first,second,first):
            status,_,response=self.request('/maintenance/select-copy',{'copyId':item['copyId']})
            self.assertEqual(status,200);self.assertEqual(response['nextCopyId'],item['copyId'])
            self.assertEqual(installation.selected(self.owner),maintenance.restored_copy(self.owner,item['copyId']))
        status,_,response=self.request('/maintenance/status')
        self.assertEqual(status,200);profiles=response['profiles']
        self.assertEqual({row['copyId'] for row in profiles['copies']},{first['copyId'],second['copyId']})
        self.assertIsNone(profiles['activeCopyId']);self.assertEqual(profiles['nextCopyId'],first['copyId'])
        active=maintenance.Maintenance(maintenance.restored_copy(self.owner,first['copyId']),self.owner,self.payload)
        self.assertEqual(active.profile_status()['activeCopyId'],first['copyId'])
        self.assertEqual(self.request('/maintenance/select-copy',{'copyId':None})[0],200)
        self.assertEqual(installation.selected(self.owner),self.owner)
        self.assertEqual((self.profile.data/'labviz.db').read_bytes(),before)

    def test_selection_failure_or_busy_operation_keeps_the_previous_choice(self):
        directory=test_maintenance.MaintenanceTests.make_backup(self)
        copy=maintenance.restore(directory,self.owner,self.payload)
        self.assertEqual(self.request('/maintenance/select-copy',{'copyId':copy['copyId']})[0],200)
        before=(self.owner/'profile-selection.json').read_bytes()
        self.assertEqual(self.request('/maintenance/select-copy',{'copyId':'b'*32})[0],409)
        self.assertEqual(self.request('/maintenance/submit',{'action':'backup'})[0],202)
        self.assertEqual(self.request('/maintenance/select-copy',{'copyId':None})[0],409)
        self.assertEqual((self.owner/'profile-selection.json').read_bytes(),before)

    def test_incomplete_copy_is_preserved_and_reported_without_blocking_valid_copies(self):
        path=self.owner/'restored-profiles'/('a'*32);path.mkdir(parents=True)
        (path/'partial.txt').write_bytes(b'keep')
        status,_,response=self.request('/maintenance/status')
        self.assertEqual(status,200);self.assertEqual(response['profiles']['copies'],[])
        self.assertEqual(response['profiles']['unavailableCopies'],1)
        self.assertEqual((path/'partial.txt').read_bytes(),b'keep')

if __name__=='__main__':unittest.main()
