"""Import new data after an upgrade and prove compatible rollback keeps it."""
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'desktop/runtime'))
TEST_BUNDLE=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'outputs/desktop-d06/bundle'
assert TEST_BUNDLE.is_relative_to(ROOT/'outputs/desktop-d06')
sys.path.insert(0,str(TEST_BUNDLE/'app/V2.0/api'))
import installation as install
from controller import ApiControl,local_key
from session import atomic,encoded,unlinked
from process_job import OwnedJob
from labviz_api.source_data import safe_files
from retention_checks import assert_retained

trial=unlinked(Path(sys.argv[1]).resolve())
assert trial.is_relative_to(Path.home()/'LabVizD2')
prior=json.loads((trial/'installer/verification.json').read_text(encoding='utf-8-sig'));assert prior['passed'] and prior['desktopRegistrationRemoved']
profile=trial/'profile';root=install.program(trial/'retention-program',create=True)
bundle=unlinked(TEST_BUNDLE)
package=json.loads((bundle/'desktop-package.json').read_text(encoding='utf-8-sig'))['packageId']
compatible=trial/'retained-compatible-generation';second=json.loads((compatible/'desktop-package.json').read_text(encoding='utf-8-sig'))['packageId']
for source,identifier in [(bundle,package),(compatible,second)]:
    target=root/'versions'/identifier[:24]
    if not target.exists():
        assert target.resolve().is_relative_to(root/'versions')
        target.mkdir(parents=True,exist_ok=False)
        members=safe_files(source)
        def copy_member(path):
            destination=target/path.relative_to(source)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(copy_member,members))
    assert install.version(root,identifier,True)==target
first_folder=install.version(root,package);second_folder=install.version(root,second)
install.mark(root,'current-candidate.json',package)

def ids():
    with closing(sqlite3.connect((profile/'appdata/LabVizV23/data/labviz.db').as_uri()+'?mode=ro',uri=True)) as db:
        return {row[0] for row in db.execute('SELECT id FROM projects')}

def project_rows():
    with closing(sqlite3.connect((profile/'appdata/LabVizV23/data/labviz.db').as_uri()+'?mode=ro',uri=True)) as db:
        return {row[0]:{'storageMode':row[1],'expiresAt':row[2]} for row in db.execute('SELECT id,storage_mode,expires_at FROM projects')}

def imported_fingerprints(identifiers):
    result={}
    with closing(sqlite3.connect((profile/'appdata/LabVizV23/data/labviz.db').as_uri()+'?mode=ro',uri=True)) as db:
        for identifier in identifiers:
            cursor=db.execute('SELECT * FROM projects WHERE id=?',(identifier,))
            columns=[column[0] for column in cursor.description]
            row=cursor.fetchone();assert row is not None
            values=dict(zip(columns,row))
            assert values['dataset_version_id'] and values['current_revision_id']
            content={key:{'binarySha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)} if isinstance(value,bytes) else value for key,value in values.items()}
            result[identifier]=hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result

def activate(identifier):
    install.prepare(root,identifier,profile,bundle/'app')
    install.mark(root,'pending-candidate.json',identifier)
    result=install.activate(root,profile,bundle/'app')
    assert not result.get('rolledBack') and result['packageId']==identifier

old_rows=project_rows();old=set(old_rows);activate(second);assert_retained(old_rows,project_rows())
config=profile/'retention-import-service.json'
metadata=json.loads((second_folder/'app/candidate-build.json').read_text(encoding='utf-8-sig'))
atomic(config,encoded({'payloadRoot':str(second_folder/'app'),'backendProfile':str(profile/'appdata/LabVizV23'),'programRoot':str(profile/'program'),'candidateId':metadata['candidateId']}))
env={**os.environ,'LOCALAPPDATA':str(profile/'appdata'),'LABVIZ_DESKTOP_MODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
for name in ['PYTHONPATH','PYTHONHOME','PYTHONPYCACHEPREFIX','NODE_PATH']:env.pop(name,None)
log=(trial/'retention-import.log').open('wb')
child=subprocess.Popen([str(second_folder/'app/runtime/python/python.exe'),'-B','-X','utf8',str(second_folder/'runtime/service_runner.py'),str(config)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
job=OwnedJob(child);ready=None
try:
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        assert child.poll() is None
        path=profile/'appdata/LabVizV23/logs/running.json'
        value=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
        if value.get('launcher',{}).get('pid')==child.pid:ready=value;break
        time.sleep(.1)
    assert ready
    key=local_key(profile/'appdata/LabVizV23/access.dpapi');api=ApiControl(ready,key);api.verify_api()
    boundary='LabVizRetention'+uuid.uuid4().hex
    content=(ROOT/'outputs/desktop-d01/d0/fixtures/desktop-small.csv').read_bytes()
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="after-upgrade.csv"\r\nContent-Type: text/csv\r\n\r\n').encode()+content+f'\r\n--{boundary}--\r\n'.encode()
    request=urllib.request.Request(f"http://127.0.0.1:{ready['apiPort']}/api/v1/projects",data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary,'Cookie':f"labviz_local_session_v23_installed_{ready['profileId']}={key}"})
    with urllib.request.urlopen(request,timeout=60) as response:assert response.status==202;response.read()
    deadline=time.monotonic()+120
    while not api.call('activity')['idle'] and time.monotonic()<deadline:time.sleep(.1)
    assert api.call('activity')['idle']
finally:
    if ready:atomic(profile/'appdata/LabVizV23/logs/stop.request',encoded({'profileId':ready['profileId'],'sessionId':ready['sessionId']}))
    try:child.wait(timeout=65)
    finally:job.close();log.close()

after_import=project_rows();new=set(after_import)-old;assert len(new)==1
assert all(after_import[identifier]['storageMode']=='local' and after_import[identifier]['expiresAt'] is None for identifier in new)
expected={**old_rows,**{identifier:after_import[identifier] for identifier in new}}
assert_retained(expected,after_import)
imported_before=imported_fingerprints(new)
activate(package)
expired_removed=assert_retained(expected,project_rows())
assert imported_fingerprints(new)==imported_before
result={'passed':True,'packageId':package,'upgradedPackageId':second,'oldProjectCount':len(old),'newProjectIds':sorted(new),'retainedProjectCount':len(ids()),'expiredTemporaryProjectIdsRemoved':expired_removed,'allDurableProjectsRetained':True,'importedProjectFingerprints':imported_before,'importedDataAndMetadataUnchangedAcrossRollback':True,'scope':'Actual authenticated CSV import after compatible upgrade; successful rollback keeps all durable projects, including the new data blob, metadata and current revision marker. Only temporary projects past their recorded expiry may disappear. No native window or installer registration created.'}
(trial/'retention-after-upgrade.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
