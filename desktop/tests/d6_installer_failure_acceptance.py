"""Verify a real silent Setup failure reports failure and retains the old program/data."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import winreg

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'desktop/runtime'))
TEST_BUNDLE=Path(sys.argv[2]).resolve().parent/'bundle'
assert TEST_BUNDLE.is_relative_to(ROOT/'outputs/desktop-d06')
sys.path.insert(0,str(TEST_BUNDLE/'app/V2.0/api'))
import installation as install
from session import unlinked

trial=unlinked(Path(sys.argv[1]).resolve());build=unlinked(Path(sys.argv[2]).resolve())
assert trial.is_relative_to(Path.home()/'LabVizD2')
assert json.loads((trial/'retention-after-upgrade.json').read_text(encoding='utf-8-sig'))['passed']
program=trial/'retention-program';profile=trial/'profile';bundle=unlinked(TEST_BUNDLE)
record=json.loads((build/'build-verification.json').read_text(encoding='utf-8-sig'));identifier=record['packageId']
exe=build/next(iter(record['installers']))
assert install.file_sha(exe)==next(iter(record['installers'].values()))['sha256']
out=trial/(sys.argv[3] if len(sys.argv)>3 else 'installer-wrapper-failure');assert out.parent==trial;out.mkdir(exist_ok=False)
def ids():
    with closing(sqlite3.connect((profile/'appdata/LabVizV23/data/labviz.db').as_uri()+'?mode=ro',uri=True)) as db:return {row[0] for row in db.execute('SELECT id FROM projects')}
def execute(command,label):
    with (out/(label+'.log')).open('wb') as stream:
        return subprocess.run([str(x) for x in command],stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW,timeout=600).returncode
def setup(label):
    return execute([exe,'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-','/NOICONS','/TASKS=!desktopicon','/DIR='+str(program),'/ACCEPTANCEROOT='+str(trial),'/LOG='+str(out/(label+'-inno.log'))],label)
before=ids();checkpoint=(profile/'session/checkpoint.json').read_bytes()
assert setup('baseline-install')==0
assert install.pointer(program,'current-candidate.json')==identifier
previous_id=json.loads((trial/'retained-compatible-generation/desktop-package.json').read_text(encoding='utf-8-sig'))['packageId']
previous=install.version(program,previous_id,True)
install.prepare(program,previous_id,profile,bundle/'app');install.mark(program,'pending-candidate.json',previous_id)
assert install.activate(program,profile,bundle/'app')['packageId']==previous_id
assert not (program/'pending-candidate.json').exists()
target=unlinked(program/'versions'/identifier[:24]/'runtime/controller.py')
assert target.is_relative_to(program/'versions')
injected=threading.Event();finished=threading.Event();faults=[]
def inject():
    try:
        deadline=time.monotonic()+500
        while not finished.is_set() and time.monotonic()<deadline:
            path=program/'pending-candidate.json'
            if path.exists():
                try:value=json.loads(path.read_text(encoding='utf-8-sig'))
                except (OSError,ValueError):time.sleep(.01);continue
                if value.get('packageId')==identifier:
                    target.write_bytes(b"raise SystemExit('Acceptance: corrupt deployed candidate')\n")
                    injected.set();return
            time.sleep(.01)
    except BaseException as error:faults.append(repr(error))
watcher=threading.Thread(target=inject);watcher.start()
try:code=setup('injected-failure')
finally:finished.set();watcher.join(timeout=5)
assert injected.is_set() and not faults,faults
assert install.pointer(program,'current-candidate.json')==previous_id
assert ids()==before and (profile/'session/checkpoint.json').read_bytes()==checkpoint
install.health(previous,profile)
assert (program/'LabViz.exe').is_file() and install.file_sha(program/'LabViz.exe')==record['hostSha256']
result={'passed':code!=0,'installerExitCode':code,'injected':True,'previousProgramRetained':True,'scientificDataRetained':True,'checkpointRetained':True,'previousServicesHealthy':True,'packageId':identifier,'scope':'Exact silent Setup, fault injected into the newly deployed candidate after its pending marker. No LabViz GUI launched.'}
(out/'verification.json').write_text(json.dumps(result,indent=2))
assert code!=0,'Silent Setup reported success after activation rolled back'
uninstaller=program/'unins000.exe'
assert execute([uninstaller,'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/ACCEPTANCEROOT='+str(trial)],'cleanup-uninstall')==0
key='Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{'+install.INSTALL_ID+'}_is1'
try:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,key):raise AssertionError('Private uninstall registration remained')
except FileNotFoundError:pass
result['registrationRemoved']=True;(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
