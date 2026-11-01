"""Exit waits for verified idle, fences new work, and can be cancelled."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

p=Path(__file__).resolve().parents[1]/'runtime/controller.py'
spec=importlib.util.spec_from_file_location('exit_controller',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Api:
    def __init__(self):self.idle=True;self.draining=False;self.generation=0;self.calls=[]
    def call(self,action,value=None):
        self.calls.append(action)
        if action=='drain':self.draining=True;self.generation+=1
        if action=='cancel-drain':self.draining=False
        return {'idle':self.idle,'draining':self.draining,'generation':self.generation}
class ExitTests(unittest.TestCase):
    def setUp(self):self.api=Api();self.ready=True;self.session=SimpleNamespace(idle=lambda:self.ready);self.request=module.ExitRequest();self.request.request()
    def test_running_work_is_not_drained_or_stopped(self):
        self.api.idle=False
        self.assertEqual(self.request.check(self.api,self.session,set()),'exit-waiting')
        self.assertEqual(self.api.calls,['activity'])
    def test_client_work_and_downloads_keep_services_running(self):
        for ready,downloads in [(False,set()),(True,{'download'})]:
            self.ready=ready;self.assertEqual(self.request.check(self.api,self.session,downloads),'exit-waiting')
        self.assertNotIn('drain',self.api.calls)
    def test_idle_is_fenced_and_cancel_reopens_admission(self):
        self.assertEqual(self.request.check(self.api,self.session,set()),'exit-ready')
        self.assertTrue(self.api.draining);self.request.cancel(self.api)
        self.assertFalse(self.api.draining);self.assertFalse(self.request.active)
    def test_confirmation_rechecks_new_download_and_generation(self):
        self.request.check(self.api,self.session,set())
        self.assertEqual(self.request.check(self.api,self.session,{'new'}),'exit-waiting')
        self.assertFalse(self.api.draining)
        self.request.check(self.api,self.session,set());self.api.generation+=1
        self.assertEqual(self.request.check(self.api,self.session,set()),'exit-waiting')
        self.assertFalse(self.api.draining)
    def test_unknown_activity_cannot_approve_exit(self):
        def unavailable(*args):raise OSError('Unavailable')
        self.api.call=unavailable
        with self.assertRaises(OSError):self.request.check(self.api,self.session,set())
        self.assertIsNone(self.request.generation)
    def test_new_maintenance_between_ready_and_confirm_blocks_exit(self):
        self.request.check(self.api,self.session,set())
        self.assertEqual(self.request.check(self.api,self.session,set(),maintenance_busy=True),'exit-waiting')
        self.assertFalse(self.api.draining)
if __name__=='__main__':unittest.main()
