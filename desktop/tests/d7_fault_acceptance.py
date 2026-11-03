"""Run real packaged-service fault checks on independent synthetic profiles.

No native executable, desktop control, system component or default profile is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from unittest.mock import patch
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Controller:
    def __init__(self, bundle, owner, base, output, label):
        self.owner, self.output, self.label = owner, output, label
        self.events, self.messages, self.known = [], queue.Queue(), {}
        self.pulse_stop=threading.Event()
        config = output/(label+'-config.json')
        config.write_text(json.dumps({'format':'labviz-desktop-session-v1',
            'profileRoot':str(owner), 'profileBase':str(base), 'payloadRoot':str(bundle/'app'),
            'candidateId':read(bundle/'desktop-runtime.json')['candidateId'],
            'startupTimeoutSeconds':120, 'idleSeconds':1}), encoding='utf-8')
        self.error = (output/(label+'-stderr.log')).open('w', encoding='utf-8')
        self.child = subprocess.Popen([str(bundle/'app/runtime/python/python.exe'), '-B', '-X', 'utf8',
            str(bundle/'runtime/controller.py'), str(config)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.error, text=True, encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW)
        self.known[self.child.pid] = psutil.Process(self.child.pid).create_time()
        def consume():
            for line in self.child.stdout:
                value=json.loads(line)
                # Ready messages contain credentials; only event names are persisted.
                self.events.append({'type':value['type'], 'monotonic':time.monotonic()})
                self.messages.put(value)
        self.reader=threading.Thread(target=consume,daemon=True);self.reader.start()

    def remember(self):
        try:
            for member in psutil.Process(self.child.pid).children(recursive=True):
                self.known[member.pid]=member.create_time()
        except psutil.NoSuchProcess:pass

    def event(self, kind, timeout=130):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            self.remember()
            try:
                value=self.messages.get(timeout=.1)
                if value['type']=='failed':raise AssertionError(value['message'])
                if value['type']==kind:
                    if kind=='ready':self.client(value)
                    return value
            except queue.Empty:
                if self.child.poll() is not None:raise AssertionError('Controller exited before '+kind)
        raise TimeoutError('No controller event: '+kind)

    def client(self, ready):
        """A protocol adapter, not a native or headless browser acceptance claim."""
        origin='/'.join(ready['url'].split('/')[:3])
        descriptor=ready['preferences']
        def pulse():
            while not self.pulse_stop.is_set() and self.child.poll() is None:
                try:
                    req=urllib.request.Request(descriptor['url'].removesuffix('/preferences')+'/session/activity',
                        data=json.dumps({'ready':True,'busy':False,'pendingRequests':0}).encode(),
                        headers={'Origin':origin,'X-LabViz-Desktop':descriptor['token'],'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=2) as response:assert response.status==200
                    self.remember()
                except (OSError,AssertionError):pass  # Closing the service also closes this adapter.
                self.pulse_stop.wait(.5)
        self.pulser=threading.Thread(target=pulse,daemon=True);self.pulser.start()

    def send(self, action):
        self.child.stdin.write(json.dumps({'action':action})+'\n');self.child.stdin.flush()

    def exited(self, expected):
        self.child.wait(timeout=100);self.reader.join(timeout=5);self.error.close()
        self.pulse_stop.set()
        assert self.child.returncode==expected, (self.label,self.child.returncode,expected)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            survivors=[]
            for pid,created in self.known.items():
                try:
                    if psutil.Process(pid).create_time()==created:survivors.append(pid)
                except psutil.NoSuchProcess:pass
            if not survivors:break
            time.sleep(.1)
        assert not survivors,survivors
        (self.output/(self.label+'-events.json')).write_text(json.dumps(self.events,indent=2))
        return {'exitCode':self.child.returncode,'ownedSurvivors':survivors}

    def close(self):
        if self.child.poll() is None:
            self.send('exit-wait');self.event('exit-ready');self.send('exit-confirm')
        result=self.exited(0)
        receipt=read(self.owner/'logs/desktop-controller-exit.json')
        assert receipt['normal'] and receipt['runningRecordRemoved'],receipt
        return result

    def emergency_close(self):
        if self.child.poll() is None:
            try:self.send('stop');self.child.wait(timeout=100)
            except (OSError,subprocess.TimeoutExpired):self.child.kill();self.child.wait(timeout=10)
        self.reader.join(timeout=5)
        self.pulse_stop.set()
        if not self.error.closed:self.error.close()


def database(owner):
    return owner/'appdata/LabVizV23/data/labviz.db'


def rows(owner, sql, args=()):
    with closing(sqlite3.connect(database(owner).resolve().as_uri()+'?mode=ro',uri=True,timeout=5)) as db:
        db.row_factory=sqlite3.Row
        return [dict(row) for row in db.execute(sql,args)]


def frozen(owner):
    result={}
    for row in rows(owner,'SELECT * FROM research_figure_artifacts ORDER BY board_id'):
        result[row['board_id']]={name:hashlib.sha256(value if isinstance(value,bytes) else str(value).encode()).hexdigest()
                                  for name,value in row.items()}
    return result


def request(ready, key, path, value=None, expected=200):
    req=urllib.request.Request(ready['webUrl']+path,
        data=None if value is None else json.dumps(value).encode(),
        headers={'Content-Type':'application/json','X-LabViz-Local-Key':key,
                 'Cookie':f"labviz_local_session_v23_installed_{ready['profileId']}={key}"})
    try:
        with urllib.request.urlopen(req,timeout=30) as response:status,data=response.status,response.read()
    except urllib.error.HTTPError as error:status,data=error.code,error.read()
    assert status==expected,(path,status,data[:500])
    return data


def await_job(owner, board, state, timeout=120):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        row=rows(owner,'SELECT * FROM research_figure_jobs WHERE board_id=?',(board,))[0]
        if row['state']==state:return row
        assert row['state']!='failed',row
        time.sleep(.01 if state=='running' else .1)
    raise TimeoutError('Job did not reach '+state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--backup',type=Path,required=True,help='An existing synthetic desktop backup')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();bundle=args.bundle.resolve();output=args.output.resolve()
    assert output.is_relative_to(ROOT/'outputs') and output!=ROOT/'outputs'
    output.mkdir(parents=True,exist_ok=False)
    package=read(bundle/'desktop-package.json');manifest=package['manifest']
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes=list(pool.map(lambda name:sha(bundle/name),manifest['files']))
    assert dict(zip(manifest['files'],hashes))==manifest['files'],'Frozen candidate changed'
    sys.path[:0]=[str(bundle/'runtime'),str(bundle/'app/V2.0/api')]
    import maintenance as maintenance
    import installation as installation
    import controller as control
    import service_runner as runner
    from labviz_api.installed_profile import InstalledProfile
    trial=Path.home()/'LabVizD2'/('d7-'+uuid.uuid4().hex[:12]);trial.mkdir()
    base=trial/'profile';installation.initialise(base)
    original_local=os.environ.get('LOCALAPPDATA')
    os.environ['LOCALAPPDATA']=str(base/'appdata')
    InstalledProfile.open(base/'appdata/LabVizV23',create=True)
    report={'format':'labviz-desktop-d7-fault-verification-v1','passed':False,'inProgress':True,
            'packageId':package['packageId'],'trial':str(trial),'checks':[],
            'nativeAppLaunched':False,'publicReleaseAccepted':False}
    def save():
        (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def check(name,**detail):
        report['checks'].append({'name':name,**detail});save();print(name,flush=True)
    active=[]
    try:
        with maintenance.backend(base,bundle/'app'):
            restored=maintenance.restore(args.backup.resolve(),base,bundle/'app')
        owner=Path(restored['restoredProfileRoot']);before=frozen(owner)
        assert before,'Synthetic backup has no frozen boards'
        session=Controller(bundle,owner,base,output,'crash');active.append(session)
        session.event('ready');session.remember()
        ready=read(owner/'logs/desktop-services.json')
        key=control.local_key(owner/'appdata/LabVizV23/access.dpapi')
        # The largest saved scene gives a deterministic window to interrupt actual rendering.
        board=rows(owner,'SELECT b.* FROM research_figure_boards b JOIN research_figure_artifacts a ON a.board_id=b.id ORDER BY length(a.scene_json) DESC LIMIT 1')[0]
        spec=json.loads(board['spec_json']);spec['title']='Crash recovery acceptance'
        url=f"/api/v1/projects/{board['project_id']}/research/figure-boards"
        created=json.loads(request(ready,key,url+'?deferArtifacts=true',spec,202));identifier=created['id']
        running=await_job(owner,identifier,'running')
        assert not rows(owner,'SELECT board_id FROM research_figure_artifacts WHERE board_id=?',(identifier,))
        api=control.ApiControl(ready,key)
        session.send('hide');observed=[];deadline=time.monotonic()+1.3
        while time.monotonic()<deadline:
            status=api.call('activity');observed.append(status['idle'])
            assert not status['idle'],'The large render finished before the busy guard observation'
            assert not any(e['type'] in {'checkpoint','suspend-commit','suspended'} for e in session.events)
            time.sleep(.1)
        session.send('show')
        check('accepted-render-blocks-suspension',busySamples=len(observed),boardId=identifier)
        # Verify the raw Windows process identity immediately before touching this owned API.
        actual=runner.record(runner.SimpleProcess(ready['api']['pid']))
        assert actual==ready['api']
        psutil.Process(actual['pid']).kill()
        result=session.exited(1)
        receipt=read(owner/'logs/desktop-controller-exit.json')
        assert not receipt['normal'] and not any(e['type']=='suspended' for e in session.events)
        assert frozen(owner)==before,'Existing frozen artifacts changed during crash'
        assert rows(owner,'SELECT state FROM research_figure_jobs WHERE board_id=?',(identifier,))[0]['state']=='running'
        assert not rows(owner,'SELECT board_id FROM research_figure_artifacts WHERE board_id=?',(identifier,))
        check('actual-api-crash-preserves-running-revision-and-frozen-sets',boardId=identifier,**result)
        recovered=Controller(bundle,owner,base,output,'recovered');active.append(recovered)
        recovered.event('ready');recovered.remember()
        current=read(owner/'logs/desktop-services.json');new_key=control.local_key(owner/'appdata/LabVizV23/access.dpapi')
        completed=await_job(owner,identifier,'ready')
        assert completed['attempt']!=running['attempt'],'Abandoned process attempt reused'
        assert len(rows(owner,'SELECT board_id FROM research_figure_artifacts WHERE board_id=?',(identifier,)))==1
        for old_id,hashes in before.items():assert frozen(owner)[old_id]==hashes
        artifact_hashes={}
        for extension,signature in [('svg',b'<?xml'),('png',b'\x89PNG'),('pdf',b'%PDF')]:
            raw=request(current,new_key,url+'/'+identifier+'/artifact/'+extension)
            assert raw.startswith(signature),(extension,raw[:30])
            artifact_hashes[extension]=hashlib.sha256(raw).hexdigest()
        request(current,new_key,url+'/'+identifier+'/retry',{},409)
        check('reopen-restarts-one-abandoned-attempt-and-publishes-one-complete-set',boardId=identifier,artifacts=artifact_hashes)
        next_board=json.loads(request(current,new_key,url+'?deferArtifacts=true',spec,202))['id']
        assert next_board!=identifier
        await_job(owner,next_board,'ready')
        for table in ['research_figure_boards','research_figure_jobs','research_figure_artifacts']:
            column='id' if table=='research_figure_boards' else 'board_id'
            assert rows(owner,f'SELECT count(*) AS n FROM {table} WHERE {column} IN (?,?)',(identifier,next_board))[0]['n']==2
        check('immediate-next-submission-has-two-revisions-without-duplicate-publication',boards=[identifier,next_board])
        final=frozen(owner);recovered.close()
        reopened=Controller(bundle,owner,base,output,'second-reopen');active.append(reopened)
        reopened.event('ready');reopened.remember();assert frozen(owner)==final
        check('second-reopen-does-not-requeue-ready-revisions',**reopened.close())
        # Steal the actual reserved API port at the runner's spawn boundary.
        # Actual Python/Node services still start; only this controlled timing seam is injected.
        race_owner=trial/'port-race';installation.initialise(race_owner)
        program=race_owner/'program';program.mkdir(exist_ok=True)
        (program/'installation.json').write_text(json.dumps({'appId':control.APP_ID,'format':'labviz-installed-v1'}))
        config=output/'port-race-config.json'
        config.write_text(json.dumps({'payloadRoot':str(bundle/'app'),'programRoot':str(program),
            'backendProfile':str(race_owner/'appdata/LabVizV23'),'candidateId':manifest['payloadCandidateId']}))
        foreign=socket.socket();children=[];spawn=subprocess.Popen;intercepted=[]
        def contested(command,*positional,**keyword):
            if '--port' in command:
                port=int(command[command.index('--port')+1]);foreign.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
                foreign.bind(('127.0.0.1',port));foreign.listen();foreign.settimeout(.1);intercepted.append(port)
            child=spawn(command,*positional,**keyword);children.append(child);return child
        os.environ['LOCALAPPDATA']=str(race_owner/'appdata')
        failure=None
        try:
            with patch.object(runner.sys,'argv',['service_runner',str(config)]),patch.object(runner.subprocess,'Popen',side_effect=contested):
                try:runner.main()
                except RuntimeError as error:failure=str(error)
            assert failure and len(intercepted)==1
            assert all(child.poll() is not None for child in children)
            assert not (race_owner/'appdata/LabVizV23/logs/running.json').exists()
            outcome=read(race_owner/'appdata/LabVizV23/logs/desktop-service-exit.json');assert not outcome['normal']
            # The foreign listener remains alive and received no authenticated request.
            assert runner.port_owned(intercepted[0],os.getpid())
            try:connection,_=foreign.accept();connection.close();raise AssertionError('Credentials may have reached the stolen listener')
            except socket.timeout:pass
            check('real-reservation-to-bind-port-theft-fails-closed',port=intercepted[0],failure=failure,
                  ownedChildrenStopped=True,foreignListenerPreserved=True,foreignRequests=0)
        finally:
            foreign.close()
            for child in children:
                if child.poll() is None:child.kill();child.wait(timeout=10)
        # Clean cache on the populated stopped profile and verify every other file.
        os.environ['LOCALAPPDATA']=str(owner/'appdata')
        cache=owner/'appdata/LabVizV23/cache/matplotlib';cache.mkdir(parents=True,exist_ok=True)
        (cache/'acceptance-regenerable.json').write_bytes(b'regenerable')
        baseline={str(p):sha(p) for p in maintenance.files(owner) if not p.resolve().is_relative_to(cache.resolve()) and p.name!='.writer.lock'}
        result=maintenance.clear_cache(owner,bundle/'app')
        assert result['removedFiles']>0 and not cache.exists()
        for name,expected in baseline.items():assert sha(Path(name))==expected,name
        check('populated-profile-cache-cleanup-keeps-all-other-bytes',**result)
        last=Controller(bundle,owner,base,output,'cache-reopen');active.append(last)
        last.event('ready');last.remember();assert frozen(owner)==final
        check('cache-cleanup-reopens-with-identical-frozen-artifacts',**last.close())
        report.update(passed=True,inProgress=False)
    except BaseException as error:
        report.update(error=repr(error),inProgress=False);raise
    finally:
        for session in active:session.emergency_close()
        if original_local is None:os.environ.pop('LOCALAPPDATA',None)
        else:os.environ['LOCALAPPDATA']=original_local
        save()


if __name__=='__main__':main()
