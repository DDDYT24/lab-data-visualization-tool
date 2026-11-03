"""Exercise long Unicode input/output paths with the actual bundled runtime and API."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'desktop/runtime'))
from session import atomic,encoded,io_path,unlinked
from controller import ApiControl,local_key
from process_job import OwnedJob

trial=Path(sys.argv[1]).resolve();assert trial.is_relative_to(Path.home()/'LabVizD2')
assert json.loads((trial/'retention-after-upgrade.json').read_text(encoding='utf-8-sig'))['passed']
assert not (trial/'profile/appdata/LabVizV23/logs/running.json').exists()
profile=trial/'profile';bundle=unlinked(Path(sys.argv[2]).resolve())
assert bundle.is_relative_to(ROOT/'outputs/desktop-d06')
workflow=unlinked(Path(sys.argv[3]).resolve());assert workflow.is_relative_to(ROOT/'outputs')
package=json.loads((bundle/'desktop-package.json').read_text(encoding='utf-8-sig'))['packageId']
folder=trial/'长路径输入输出'/('科研路径'*25)/('源文件路径'*20)
io_path(folder).mkdir(parents=True,exist_ok=True)
source=folder/'科研数据.csv';target=folder/'冻结结果.pdf'
raw=(ROOT/'outputs/desktop-d01/d0/fixtures/desktop-small.csv').read_bytes();io_path(source).write_bytes(raw)
assert len(str(source))>260 and len(str(target))>260
metadata=json.loads((bundle/'app/candidate-build.json').read_text(encoding='utf-8-sig'));config=profile/'long-path-service.json'
atomic(config,encoded({'payloadRoot':str(bundle/'app'),'backendProfile':str(profile/'appdata/LabVizV23'),'programRoot':str(profile/'program'),'candidateId':metadata['candidateId']}))
env={**os.environ,'LOCALAPPDATA':str(profile/'appdata'),'LABVIZ_DESKTOP_MODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
for name in ['PYTHONPATH','PYTHONHOME','PYTHONPYCACHEPREFIX','NODE_PATH']:env.pop(name,None)
log=(trial/'long-path-service.log').open('wb')
child=subprocess.Popen([str(bundle/'app/runtime/python/python.exe'),'-B','-X','utf8',str(bundle/'runtime/service_runner.py'),str(config)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
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
 old=json.loads(workflow.read_text(encoding='utf-8-sig'));assert old['passed'];pdf=next(item for item in old['exports'] if item['format']=='pdf')
 private={'api':f"http://127.0.0.1:{ready['apiPort']}",'cookie':f"labviz_local_session_v23_installed_{ready['profileId']}={key}",'input':str(source),'output':str(target),'inputSha256':hashlib.sha256(raw).hexdigest(),'pdfHref':pdf['href'],'pdfSha256':pdf['sha256']}
 result=subprocess.run([str(bundle/'app/runtime/node/node.exe'),str(ROOT/'desktop/tests/d6_long_path_client.mjs')],input=json.dumps(private).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW,timeout=180)
 (trial/'long-path-client.log').write_bytes(result.stdout+result.stderr);assert result.returncode==0
 record=json.loads(result.stdout);assert record['passed'];record['packageId']=package
 (trial/'long-path-verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))
finally:
 if ready:atomic(profile/'appdata/LabVizV23/logs/stop.request',encoded({'profileId':ready['profileId'],'sessionId':ready['sessionId']}))
 try:child.wait(timeout=65)
 finally:job.close();log.close()
 assert child.returncode==0
 assert json.loads((profile/'appdata/LabVizV23/logs/desktop-service-exit.json').read_text(encoding='utf-8-sig'))['normal']
