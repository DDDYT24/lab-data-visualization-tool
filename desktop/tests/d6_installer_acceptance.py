"""Exercise the exact private installer silently in an owned fixture, without GUI launch."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import winreg

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'desktop/runtime'))
import installation as lifecycle
from session import encoded, unlinked, io_path

trial=unlinked(Path(sys.argv[1]).resolve())
build=unlinked(Path(sys.argv[2]).resolve())
assert build.is_relative_to(ROOT/'outputs/desktop-d06')
sys.path.insert(0,str(build.parent/'bundle/app/V2.0/api'))
assert trial.is_relative_to(Path.home()/'LabVizD2') or trial.is_relative_to(Path.home()/'LabVizD6')
program=trial/'installed';profile=trial/'profile';output=trial/'installer'
output.mkdir(exist_ok=False)
record=json.loads((build/'build-verification.json').read_text(encoding='utf-8-sig'))
exe_name,receipt=next(iter(record['installers'].items()));exe=build/exe_name
assert lifecycle.file_sha(exe)==receipt['sha256'] and record['compilerExitCode']==0
exit_record=json.loads((trial/'exit.json').read_text(encoding='utf-8-sig'));session=json.loads((trial/'session.json').read_text(encoding='utf-8-sig'))
assert exit_record['runId']==session['id'] and not exit_record['survivingOwnedProcesses'] and exit_record['hostExitCode']==0
(trial/'acceptance.json').write_bytes(encoded({'format':'labviz-desktop-installer-acceptance-v1','appId':lifecycle.DESKTOP_ID,'root':str(trial)}))
registry='Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{'+lifecycle.INSTALL_ID+'}_is1'
report={'passed':False,'inProgress':True,'packageId':record['packageId'],'installerSha256':receipt['sha256'],'scope':'Actual silent installer and bundled-service activation; no native app launch, desktop shortcut or system component installation','checks':[]}
def save(): (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
def check(name,**values):report['checks'].append({'name':name,**values});save();print(name,flush=True)
def registrations():
    found=[]
    for hive,label in [(winreg.HKEY_CURRENT_USER,'HKCU'),(winreg.HKEY_LOCAL_MACHINE,'HKLM')]:
        for view in [winreg.KEY_WOW64_64KEY,winreg.KEY_WOW64_32KEY]:
            try:
                with winreg.OpenKey(hive,registry,0,winreg.KEY_READ|view) as key:found.append((label,winreg.QueryValueEx(key,'InstallLocation')[0]))
            except FileNotFoundError:pass
    return found
assert not registrations(),'An existing desktop uninstall identity must not be adopted'
def execute(command,label,timeout=600,expect=0):
    env={**os.environ,'PATH':os.pathsep.join([os.environ['SystemRoot']+'\\System32',os.environ['SystemRoot'],os.environ['SystemRoot']+'\\System32\\WindowsPowerShell\\v1.0'])}
    with (output/(label+'.log')).open('wb') as log:
        result=subprocess.run([str(x) for x in command],stdout=log,stderr=log,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW,env=env,timeout=timeout,check=False)
    assert result.returncode==expect,(label,result.returncode)
def install(label):
    execute([exe,'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-','/NOICONS','/TASKS=!desktopicon','/DIR='+str(program),'/ACCEPTANCEROOT='+str(trial),'/LOG='+str(output/(label+'-inno.log'))],label)
    assert lifecycle.pointer(program,'current-candidate.json')==record['packageId']
    assert registrations() and all(hive=='HKCU' and Path(path).resolve()==program for hive,path in registrations())
def science():
    with closing(sqlite3.connect((profile/'appdata/LabVizV23/data/labviz.db').as_uri()+'?mode=ro',uri=True)) as db:
        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'projects','research_heads','research_revisions','research_charts','research_figure_boards','research_outputs'} <= tables
        return {'counts':{name:db.execute('SELECT count(*) FROM '+name).fetchone()[0] for name in ['projects','project_revisions','research_revisions','research_charts','research_figure_boards','research_outputs'] if name in tables},'heads':db.execute('SELECT project_id,revision_id FROM research_heads ORDER BY project_id').fetchall()}
def profile_hashes():
    from labviz_api.source_data import safe_files
    members=safe_files(profile)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes=list(pool.map(lifecycle.file_sha,members))
    return dict(zip((p.relative_to(profile).as_posix() for p in members),hashes))

def footprint():
    members=[p for p in program.rglob('*') if p.is_file()]
    return {'logicalBytes':sum(p.stat().st_size for p in members),'files':len(members),
            'scope':'Complete private program root, including launcher, retained versions, support and uninstall files; excludes profile and system WebView2'}
before=science();checkpoint=(profile/'session/checkpoint.json').read_bytes()
try:
    install('first-install')
    assert science()==before and (profile/'session/checkpoint.json').read_bytes()==checkpoint
    folder=lifecycle.version(program,record['packageId'])
    package=json.loads((folder/'desktop-package.json').read_text(encoding='utf-8-sig'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda item:lifecycle.file_sha(folder/item[0])==item[1],package['manifest']['files'].items()))
    assert all(results);assert lifecycle.file_sha(program/'LabViz.exe')==record['hostSha256']
    check('actual-per-user-installation-exact-files-and-authenticated-activation',files=len(results),registry=registrations(),footprint=footprint())
    # A real .lnk is created only inside the private fixture, never on the desktop.
    powershell=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    shortcut_script=output/'shortcut.ps1'
    def quote(value):return "'"+str(value).replace("'","''")+"'"
    shortcut_script.write_text("$s=New-Object -ComObject WScript.Shell\n$l=$s.CreateShortcut("+quote(output/'Preview.lnk')+")\n$l.TargetPath="+quote(program/'LabViz.exe')+"\n$l.WorkingDirectory="+quote(program)+"\n$l.Save()\n$r=$s.CreateShortcut("+quote(output/'Preview.lnk')+")\nif($r.TargetPath -ne "+quote(program/'LabViz.exe')+"){throw 'Shortcut differs'}\n",encoding='utf-8-sig')
    execute([powershell,'-NoProfile','-ExecutionPolicy','Bypass','-File',shortcut_script],'private-shortcut')
    check('private-windows-shortcut-targets-gui-executable',desktopShortcutChanged=False)
    install('same-version-repair');assert science()==before
    check('actual-same-version-repair-keeps-scientific-data-and-drafts')
    def clone(label,broken=False):
        from labviz_api.source_data import safe_files
        staging=trial/('clone-'+label);assert not staging.exists();staging.mkdir()
        def copy(member):
            target=unlinked(staging/member.relative_to(folder));io_path(target.parent).mkdir(parents=True,exist_ok=True)
            shutil.copy2(io_path(member),io_path(target))
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(copy,safe_files(folder)))
        manifest=json.loads((staging/'desktop-package.json').read_text(encoding='utf-8-sig'))['manifest'];manifest['acceptanceScenario']=label
        if broken:
            runner=staging/'runtime/service_runner.py';runner.write_text("raise SystemExit('Acceptance: deliberate activation failure')\n")
            binding=json.loads((staging/'desktop-runtime.json').read_text(encoding='utf-8-sig'));binding['serviceRunnerSha256']=lifecycle.file_sha(runner);(staging/'desktop-runtime.json').write_text(json.dumps(binding))
        members=[p for p in safe_files(staging) if p!=staging/'desktop-package.json']
        with ThreadPoolExecutor(max_workers=8) as pool:hashes=list(pool.map(lifecycle.file_sha,members))
        manifest['files']={p.relative_to(staging).as_posix():digest for p,digest in zip(members,hashes)}
        identifier=hashlib.sha256(encoded(manifest)).hexdigest();(staging/'desktop-package.json').write_bytes(encoded({'packageId':identifier,'manifest':manifest}))
        target=program/'versions'/identifier[:24]
        assert staging.resolve().is_relative_to(trial) and target.resolve().is_relative_to(program/'versions') and not target.exists()
        shutil.move(staging,target)
        return identifier,target
    def activate(identifier,label):
        tool=folder/'runtime/installation.py';python=folder/'app/runtime/python/python.exe';payload=folder/'app'
        execute([python,'-B','-X','utf8',tool,'prepare','--install-root',program,'--profile-base',profile,'--payload',payload,'--package',identifier],label+'-prepare')
        lifecycle.mark(program,'pending-candidate.json',identifier)
        execute([python,'-B','-X','utf8',tool,'activate','--install-root',program,'--profile-base',profile,'--payload',payload],label+'-activate')
    second,second_folder=clone('compatible-generation')
    activate(second,'compatible-update');assert lifecycle.pointer(program,'current-candidate.json')==second and science()==before
    activate(record['packageId'],'compatible-rollback');assert science()==before
    check('actual-compatible-generation-upgrade-and-rollback-preserve-data',footprint=footprint())
    failed,failed_folder=clone('failed-generation',True)
    activate(failed,'failed-update');assert lifecycle.pointer(program,'current-candidate.json')==record['packageId'] and science()==before
    assert (profile/'session/checkpoint.json').read_bytes()==checkpoint
    check('actual-failed-service-activation-restores-previous-program-data-and-draft')
    for member,label in [(second_folder,'compatible-generation'),(failed_folder,'failed-generation')]:
        destination=trial/('retained-'+label);assert member.resolve().is_relative_to(program/'versions') and destination.resolve().is_relative_to(trial) and not destination.exists();shutil.move(member,destination)
    schema=profile/'appdata/LabVizV23/data-schema.json';original=schema.read_bytes();future=json.loads(original);future['dataEpoch']=2;schema.write_bytes(encoded(future))
    execute([folder/'app/runtime/python/python.exe','-B','-X','utf8',folder/'runtime/installation.py','prepare','--install-root',program,'--profile-base',profile,'--payload',folder/'app','--package',record['packageId']],'future-epoch',expect=1)
    assert schema.read_bytes()==encoded(future) and science()==before;schema.write_bytes(original)
    check('future-epoch-refusal-keeps-current-data-and-candidate')
    snapshot=profile_hashes()
    execute([program/'unins000.exe','/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/ACCEPTANCEROOT='+str(trial),'/LOG='+str(output/'uninstall-inno.log')],'uninstall')
    assert not registrations() and not (program/'LabViz.exe').exists();assert profile_hashes()==snapshot
    check('actual-uninstall-retains-every-profile-byte-and-clears-only-desktop-registration')
    install('reinstall');assert science()==before and (profile/'session/checkpoint.json').read_bytes()==checkpoint
    check('actual-reinstall-reopens-retained-data-with-bundled-runtimes')
    # Leave the machine without this private test registration or installed program.
    snapshot=profile_hashes()
    execute([program/'unins000.exe','/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/ACCEPTANCEROOT='+str(trial)],'final-uninstall')
    assert not registrations() and profile_hashes()==snapshot
    report.update(passed=True,inProgress=False,desktopRegistrationRemoved=True,nativeAppLaunched=False)
    save();print(json.dumps({'passed':True,'checks':len(report['checks'])}),flush=True)
except BaseException as error:
    report.update(error=repr(error),inProgress=False);save();raise
