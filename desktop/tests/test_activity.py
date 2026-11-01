"""Exercise the ASGI gate through actual streaming and background-task lifetimes."""
import asyncio
import importlib.util
import importlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager, closing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/desktop-d01/payload/runtime/python/Lib/site-packages'))
sys.path.insert(0, str(ROOT/'V2.0/api'))
module = importlib.import_module('labviz_api.desktop_lifecycle')
from starlette.background import BackgroundTask
from starlette.responses import Response


class ActivityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / 'app.db'
        with self.connect() as db:
            db.executescript('CREATE TABLE jobs(stage TEXT); CREATE TABLE research_figure_jobs(state TEXT);')
        self.activity = module.Activity(self.database, SimpleNamespace(state=SimpleNamespace()))

    def tearDown(self):
        self.temp.cleanup()

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.database)) as db:
            with db:
                yield db

    def scope(self):
        return {'type': 'http', 'path': '/api/v1/projects', 'method': 'POST', 'headers': [], 'query_string': b''}

    async def receive(self):
        return {'type': 'http.request', 'body': b''}

    async def test_drain_cannot_pass_before_last_response_send_finishes(self):
        sending, release = asyncio.Event(), asyncio.Event()
        async def app(scope, receive, send):
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'file', 'more_body': False})
        async def send(message):
            if message['type'] == 'http.response.body':
                sending.set(); await release.wait()
        task = asyncio.create_task(module.ActivityMiddleware(app, self.activity)(self.scope(), self.receive, send))
        await sending.wait()
        self.assertFalse(self.activity.drain()['idle'])
        release.set(); await task
        self.assertTrue(self.activity.status()['idle'])

    async def test_background_import_remains_active_after_response_headers_and_body(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def background():
            started.set(); await release.wait()
        response = Response('accepted', background=BackgroundTask(background))
        async def send(_):
            pass
        task = asyncio.create_task(module.ActivityMiddleware(response, self.activity)(self.scope(), self.receive, send))
        await started.wait()
        self.assertFalse(self.activity.drain()['idle'])
        release.set(); await task
        self.assertTrue(self.activity.status()['idle'])

    async def test_new_work_is_rejected_while_draining_and_cancellation_reopens_admission(self):
        reached, sent = [], []
        async def app(scope, receive, send):
            reached.append(True)
        async def send(message):
            sent.append(message)
        gate = module.ActivityMiddleware(app, self.activity)
        self.activity.drain()
        await gate(self.scope(), self.receive, send)
        self.assertFalse(reached)
        self.assertEqual(sent[0]['status'], 503)
        self.activity.cancel()
        await gate(self.scope(), self.receive, send)
        self.assertEqual(reached, [True])

    async def test_queued_jobs_and_terminal_artifact_callbacks_are_busy_without_http_requests(self):
        with self.connect() as db:
            db.execute("INSERT INTO research_figure_jobs VALUES ('queued')")
        self.assertFalse(self.activity.status()['idle'])
        with self.connect() as db:
            db.execute("UPDATE research_figure_jobs SET state='ready'")
        import threading
        self.activity.app.state.figure_jobs = SimpleNamespace(lock=threading.Lock(), owned={'writing-cache': None})
        self.assertFalse(self.activity.status()['idle'])
        self.activity.app.state.figure_jobs.owned.clear()
        self.assertTrue(self.activity.status()['idle'])

    async def test_maintenance_cannot_enter_during_drain(self):
        self.activity.drain()
        with self.activity.operation() as admitted:
            self.assertFalse(admitted)
        self.assertEqual(self.activity.active, 0)
        self.activity.cancel()
        with self.activity.operation() as admitted:
            self.assertTrue(admitted)
            self.assertEqual(self.activity.active, 1)
        self.assertEqual(self.activity.active, 0)

    async def test_control_baseline_matches_the_public_uuid_serialization(self):
        from fastapi import FastAPI
        app=FastAPI();app.state.settings=SimpleNamespace(database_path=self.database)
        with self.connect() as db:
            db.executescript('CREATE TABLE projects(id TEXT,current_revision_id TEXT); CREATE TABLE research_heads(project_id TEXT,revision_id TEXT); CREATE TABLE research_figure_boards(id TEXT,project_id TEXT,created_at TEXT);')
            db.execute('INSERT INTO projects VALUES (?,?)',('p1','4b702969f64c4741ad11d34dc11236c2'))
        module.install(app,SimpleNamespace(identifier='profile'),'test-local-key')
        endpoint=next(route.endpoint for route in app.routes if getattr(route,'path','')=='/api/v1/local/draft-bases')
        self.assertEqual(endpoint(['p1'])['projects']['p1']['project'],'4b702969-f64c-4741-ad11-d34dc11236c2')
