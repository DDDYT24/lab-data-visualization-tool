"""Malformed activity data must block suspension rather than become a stop signal."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

source=Path(__file__).resolve().parents[1]/'runtime/controller.py'
spec=importlib.util.spec_from_file_location('desktop_controller_protocol',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ControllerProtocolTests(unittest.TestCase):
    def test_installer_request_requires_fresh_profile_and_process_identity(self):
        import os,time
        from service_runner import record,SimpleProcess
        owner=Path.cwd()
        value={'format':'labviz-desktop-maintenance-request-v1','appId':module.DESKTOP_ID,'action':'prepare-update','profileRoot':str(owner),'expiresUtcSeconds':time.time()+60,'requester':record(SimpleProcess(os.getpid()))}
        self.assertTrue(module.valid_maintenance_request(value,owner))
        for changed in [dict(value,expiresUtcSeconds=time.time()-1),dict(value,profileRoot='other'),dict(value,requester={**value['requester'],'startedTicks':1}),dict(value,requester=None)]:
            self.assertFalse(module.valid_maintenance_request(changed,owner))

    def test_abandoned_installer_process_cannot_close_a_later_session(self):
        import subprocess,sys,time
        from service_runner import record,SimpleProcess
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.5)'],creationflags=subprocess.CREATE_NO_WINDOW)
        value={'format':'labviz-desktop-maintenance-request-v1','appId':module.DESKTOP_ID,'action':'prepare-update','profileRoot':str(Path.cwd()),'expiresUtcSeconds':time.time()+60,'requester':record(SimpleProcess(child.pid))}
        child.wait(timeout=10)
        self.assertFalse(module.valid_maintenance_request(value,Path.cwd()))

    def status(self):
        return {'format':'labviz-desktop-activity-v1','profileId':'profile','draining':True,'generation':2,'idle':True,'requests':0,'imports':0,'figures':0,'ownedFigureTasks':0}

    def invoke(self,value):
        control=module.ApiControl({'apiPort':12345,'profileId':'profile'},'synthetic-test-key')
        with patch.object(control,'verify_api'), patch.object(module.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(value).encode())):
            return control.call('activity')

    def test_valid_drain_status_is_accepted(self):
        self.assertTrue(self.invoke(self.status())['idle'])

    def test_missing_malformed_or_inconsistent_counters_fail_closed(self):
        for field,value in [('requests',1),('imports',-1),('draining','yes'),('format','unknown'),('profileId','other'),('generation',True)]:
            status=self.status();status[field]=value
            with self.assertRaises(ValueError):self.invoke(status)
        with self.assertRaises(ValueError):self.invoke({'profileId':'profile'})

    def test_foreign_listener_is_refused_before_sending_credentials(self):
        import os,socket
        from service_runner import record,SimpleProcess
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen()
            control=module.ApiControl({'profileId':'profile','apiPort':listener.getsockname()[1],'api':record(SimpleProcess(os.getpid()))},'test-only-key')
            control.verify_api()
            control.ready['api']['startedTicks']-=1
            with patch.object(module.urllib.request,'urlopen') as send:
                with self.assertRaises(ValueError):control.call('activity')
                send.assert_not_called()
