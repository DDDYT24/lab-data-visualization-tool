"""Windows process, port, mutex and credential boundaries of the direct desktop runner."""
import importlib.util
import os
import socket
import tempfile
import threading
from pathlib import Path
import unittest
import uuid
import base64
import ctypes
import gc
import json
from unittest.mock import patch

source=Path(__file__).resolve().parents[1]/'runtime/service_runner.py'
spec=importlib.util.spec_from_file_location('desktop_service_runner',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
spec2=importlib.util.spec_from_file_location('runner_test_controller',source.with_name('controller.py'))
controller=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(controller)

@unittest.skipUnless(os.name=='nt','Windows ownership APIs')
class ServiceRunnerTests(unittest.TestCase):
    def test_foreign_installation_marker_is_refused_before_profile_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);payload=root/'payload';payload.mkdir();program=root/'program';program.mkdir()
            (payload/'candidate-build.json').write_text(json.dumps({'appId':'55B46BB7-101E-46DC-951A-3DE7E402EDD0','candidateId':'c'*64,'dataEpoch':1}))
            (program/'installation.json').write_text(json.dumps({'appId':'unrelated','format':'labviz-installed-v1'}))
            (program/'keep.txt').write_bytes(b'foreign data')
            config=root/'runner.json';config.write_text(json.dumps({'payloadRoot':str(payload),'programRoot':str(program),'backendProfile':str(root/'backend'),'candidateId':'c'*64}))
            with patch.object(module.sys,'argv',['runner',str(config)]),patch.object(module.sys,'path',list(module.sys.path)):
                with self.assertRaisesRegex(ValueError,'installation identity'):
                    module.main()
            self.assertFalse((root/'backend').exists());self.assertEqual((program/'keep.txt').read_bytes(),b'foreign data')

    def test_only_the_actual_loopback_listener_is_owned(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen()
            port=listener.getsockname()[1]
            self.assertTrue(module.port_owned(port,os.getpid()))
            self.assertFalse(module.port_owned(port,os.getpid()+1))

    def test_current_user_dpapi_round_trip_and_private_process_record(self):
        raw=os.urandom(32)
        protected=module.protect(raw)
        self.assertNotEqual(protected,raw)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'access.dpapi';path.write_bytes(protected)
            self.assertEqual(controller.local_key(path),base64.urlsafe_b64encode(raw).rstrip(b'=').decode())
        record=module.record(module.SimpleProcess(os.getpid()))
        self.assertEqual(record['pid'],os.getpid());self.assertGreater(record['startedTicks'],0)
        self.assertTrue(Path(record['executable']).is_file());self.assertTrue(module.sid().startswith('S-1-'))

    def test_other_threads_cannot_acquire_an_owned_profile_mutex(self):
        name='Local\\LabViz-Runner-Test-'+uuid.uuid4().hex
        owned=module.Mutex(name);results=[]
        def contender():
            try:
                candidate=module.Mutex(name);candidate.close();results.append('unexpected')
            except ValueError:results.append('refused')
        worker=threading.Thread(target=contender);worker.start();worker.join();owned.close()
        self.assertEqual(results,['refused'])
        reopened=module.Mutex(name);reopened.close()

    def test_repeated_dpapi_reads_do_not_grow_the_pointer_type_cache(self):
        raw=os.urandom(32)
        expected=base64.urlsafe_b64encode(raw).rstrip(b'=').decode()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'access.dpapi'
            path.write_bytes(module.protect(raw))
            self.assertEqual(controller.local_key(path),expected)
            gc.collect()
            warmed=len(ctypes._pointer_type_cache)
            for _ in range(400):
                self.assertEqual(controller.local_key(path),expected)
            gc.collect()
            self.assertEqual(len(ctypes._pointer_type_cache),warmed)
