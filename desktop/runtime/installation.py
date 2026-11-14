"""Private desktop installation activation and rollback, without opening a window."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maintenance import DATA_EPOCH, DESKTOP_ID, backend, backup, checkpoint_schema, identity, read, verify, restored_copy, files
from session import atomic, encoded, unlinked
from service_runner import Mutex, sid, SimpleProcess, record as process_record
from process_job import OwnedJob

INSTALL_ID = 'CA2B3AF4-BFB0-4541-8EA5-2E978F105D63'
INSTALLATION = {'format': 'labviz-desktop-installation-v1', 'appId': DESKTOP_ID, 'installerId': INSTALL_ID}
PACKAGE_ID = re.compile(r'^[a-f0-9]{64}$')


def lock_name(prefix, path):
    return 'Local\\' + prefix + '-' + hashlib.sha256((sid() + '|' + str(path).lower()).encode()).hexdigest().upper()


def selected(base):
    base = unlinked(base)
    selection = base / 'profile-selection.json'
    if not selection.exists():
        return base
    value = read(selection)
    if set(value) != {'format', 'appId', 'copyId'} or value['format'] != 'labviz-desktop-selection-v1' or value['appId'] != DESKTOP_ID or not re.fullmatch('[a-f0-9]{32}', value['copyId']):
        raise ValueError('The selected desktop profile identity differs')
    return restored_copy(base, value['copyId'])


def program(root, create=False):
    root = unlinked(root.absolute())
    local = Path(os.environ['LOCALAPPDATA']).resolve()
    for reserved in (local/'Programs/LabViz', local/'LabViz', local/'Programs/LabVizV23', local/'LabVizV23', local/'LabVizDesktopPreview'):
        if root == reserved or root.is_relative_to(reserved) or reserved.is_relative_to(root):
            raise ValueError('Existing program and data roots are protected')
    marker = root / 'desktop-installation.json'
    if not marker.exists() and create:
        if root.exists() and any(root.iterdir()):
            raise ValueError('An unrelated installation directory cannot be adopted')
        if create == 'check':
            return root
        root.mkdir(parents=True, exist_ok=True)
        atomic(marker, encoded(INSTALLATION))
    if read(marker) != INSTALLATION:
        raise ValueError('The desktop installation identity differs')
    return root


def version(root, candidate, verify_files=False):
    if not isinstance(candidate, str) or not PACKAGE_ID.fullmatch(candidate):
        raise ValueError('Invalid desktop package identity')
    folder = unlinked(root/'versions'/candidate[:24])
    envelope = read(folder/'desktop-package.json')
    record = envelope['manifest']
    if record['format'] != 'labviz-desktop-package-v1' or envelope['packageId'] != candidate or hashlib.sha256(encoded(record)).hexdigest() != candidate or record['appId'] != DESKTOP_ID or type(record['dataEpoch']) is not int or not 1 <= record['dataEpoch'] <= DATA_EPOCH or record['checkpointSchema'] != 1:
        raise ValueError('Desktop package identity or format differs')
    if verify_files:
        from labviz_api.source_data import safe_files
        members = [f for f in safe_files(folder) if f != folder/'desktop-package.json']
        with ThreadPoolExecutor(max_workers=8) as pool:
            hashes = list(pool.map(file_sha, members))
        actual = {f.relative_to(folder).as_posix(): digest for f, digest in zip(members, hashes)}
        if actual != record['files']:
            raise ValueError('Installed desktop files differ from the exact candidate')
    return folder


def file_sha(path):
    with unlinked(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pointer(root, name):
    path = unlinked(root/name)
    if not path.exists():
        return None
    value = read(path)
    if set(value) != {'appId','packageId'} or value['appId'] != DESKTOP_ID or not PACKAGE_ID.fullmatch(value['packageId']):
        raise ValueError('Invalid desktop candidate pointer')
    return value['packageId']


def mark(root, name, candidate):
    atomic(root/name, encoded({'appId': DESKTOP_ID, 'packageId': candidate}))


def owned_backup(owner, receipt, payload, maximum_epoch=DATA_EPOCH):
    """Bind an immutable backup to the selected owner before replacing any state."""
    if (not isinstance(receipt, dict) or set(receipt) != {'backupId', 'backupDirectory'}
            or not re.fullmatch('[a-f0-9]{32}', receipt.get('backupId', ''))):
        raise ValueError('Invalid desktop rollback backup receipt')
    directory = unlinked(Path(receipt['backupDirectory']))
    if directory != unlinked(owner/'backups'/receipt['backupId']):
        raise ValueError('The desktop rollback backup is outside this profile')
    record = verify(directory, payload)
    with backend(owner, payload) as profile:
        if (record['backupId'] != receipt['backupId'] or record['profileId'] != profile.identifier
                or record['checkpointSchema'] != 1 or record['dataEpoch'] > maximum_epoch):
            raise ValueError('The desktop rollback backup identity or epoch differs')
    return directory


def restore_desktop(owner, source):
    """Swap a complete saved desktop state, retaining every newer draft and staged file."""
    staging = unlinked(owner/('desktop-staging-'+uuid.uuid4().hex))
    staging.mkdir()
    for member in files(source) if source.exists() else []:
        relative = member.relative_to(source)
        if relative.parts[0] not in {'session', 'preferences.json'}:
            raise ValueError('Unexpected desktop rollback backup member')
        target = unlinked(staging/relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(member, target)
    retained = unlinked(owner/('desktop.retained-'+uuid.uuid4().hex))
    retained.mkdir()
    moved = []
    installed = []
    try:
        for name in ('session', 'preferences.json'):
            current = unlinked(owner/name)
            if current.exists():
                if current.is_dir():
                    files(current)
                os.replace(current, retained/name)
                moved.append(name)
        for name in ('session', 'preferences.json'):
            current = staging/name
            if current.exists():
                os.replace(current, owner/name)
                installed.append(name)
    except BaseException:
        for name in reversed(installed):
            os.replace(owner/name, retained/('failed-'+name))
        for name in reversed(moved):
            os.replace(retained/name, owner/name)
        raise
    if not any(staging.iterdir()):
        staging.rmdir()
    return retained


def restore_snapshot(owner, receipt, payload):
    directory = owned_backup(owner, receipt, payload)
    with backend(owner, payload) as profile:
        from labviz_api.installed_lifecycle import replace_active
        retained = replace_active(profile, directory/'data')
    desktop = restore_desktop(owner, directory/'desktop')
    checkpoint_schema(owner, strict=True)
    return {'retainedDataDirectory': str(retained), 'retainedDesktopDirectory': str(desktop)}


def initialise(owner):
    owner = unlinked(owner)
    if not (owner/'desktop-profile.json').exists():
        if owner.exists() and any(owner.iterdir()):
            raise ValueError('An unrelated profile cannot be adopted')
        owner.mkdir(parents=True, exist_ok=True)
        atomic(owner/'desktop-profile.json', encoded({'format': 'labviz-desktop-profile-v1', 'appId': DESKTOP_ID, 'root': str(owner)}))
    identity(owner)
    (owner/'program').mkdir(exist_ok=True)
    marker = owner/'program/installation.json'
    value = {'appId': '55B46BB7-101E-46DC-951A-3DE7E402EDD0', 'format': 'labviz-installed-v1'}
    if not marker.exists(): atomic(marker, encoded(value))
    if read(marker) != value: raise ValueError('Backend installation identity differs')


def quiesce(owner, timeout=200):
    """Ask the live controller to checkpoint, stop, back up and exit. Never kill it."""
    name = lock_name('LabVizDesktopPreview', owner)
    try:
        held = Mutex(name)
        held.close()
        return
    except ValueError:
        request_id = uuid.uuid4().hex
        request_file = unlinked(owner/'maintenance-request.json')
        atomic(request_file, encoded({'format':'labviz-desktop-maintenance-request-v1','appId':DESKTOP_ID,'action':'prepare-update','requestId':request_id,'profileRoot':str(owner),'expiresUtcSeconds':time.time()+timeout,'requester':process_record(SimpleProcess(os.getpid()))}))
        deadline = time.monotonic()+timeout
        try:
            while time.monotonic()<deadline:
                path = owner/'maintenance.json'
                result = read(path) if path.exists() else {}
                if result.get('requestId') == request_id and result.get('state') == 'failed':
                    raise ValueError('Workspace retained: '+result.get('message','maintenance failed'))
                if result.get('requestId') == request_id and result.get('state') == 'completed':
                    try:
                        held=Mutex(name);held.close();return
                    except ValueError: pass
                time.sleep(.2)
            raise TimeoutError('The workspace is busy or unavailable. Finish work, exit LabViz and retry setup.')
        finally:
            if request_file.exists() and read(request_file).get('requestId') == request_id:
                request_file.unlink()


def prepare(root, candidate, base, payload):
    root = program(root, create=True)
    owner = selected(base)
    if (owner/'desktop-profile.json').exists():
        quiesce(owner)
        held = Mutex(lock_name('LabVizDesktopPreview',owner))
        try:
            with backend(owner,payload) as profile:
                if profile.epoch()>DATA_EPOCH: raise ValueError('A newer data schema requires a compatible application')
                checkpoint_schema(owner,strict=True)
            saved = backup(owner,payload)
        finally: held.close()
    else: saved = None
    document={'format':'labviz-desktop-upgrade-v1','appId':DESKTOP_ID,'packageId':candidate,'previous':pointer(root,'current-candidate.json'),'profileRoot':str(owner),'backup':saved,'state':'prepared'}
    atomic(root/'upgrade.json',encoded(document))
    return document


def health(folder, owner, timeout=120):
    initialise(owner)
    config=owner/('health-'+uuid.uuid4().hex+'.json')
    metadata=read(folder/'app/candidate-build.json')
    atomic(config,encoded({'payloadRoot':str(folder/'app'),'backendProfile':str(owner/'appdata/LabVizV23'),'programRoot':str(owner/'program'),'candidateId':metadata['candidateId']}))
    env={**os.environ,'LOCALAPPDATA':str(owner/'appdata'),'LABVIZ_DESKTOP_MODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    for name in ('PYTHONPATH','PYTHONHOME','PYTHONPYCACHEPREFIX','NODE_PATH'):env.pop(name,None)
    running=owner/'appdata/LabVizV23/logs/running.json'
    log=(owner/'activation-health.log').open('ab')
    child=subprocess.Popen([str(folder/'app/runtime/python/python.exe'),'-B','-X','utf8',str(folder/'runtime/service_runner.py'),str(config)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    job=OwnedJob(child)
    ready=None
    try:
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if child.poll() is not None:raise ValueError('Candidate services failed the activation health check')
            if running.exists():
                value=read(running)
                if value['launcher']['pid']==child.pid and value['candidateId']==metadata['candidateId']:
                    ready=value;return {'servicesReady':True}
            time.sleep(.1)
        raise TimeoutError('Candidate activation health check timed out')
    finally:
        if ready:
            atomic(owner/'appdata/LabVizV23/logs/stop.request',encoded({'profileId':ready['profileId'],'sessionId':ready['sessionId']}))
            try:child.wait(timeout=65)
            except subprocess.TimeoutExpired:pass
        job.close()
        child.wait(timeout=20)
        log.close()


def activate(root, base, payload):
    root=program(root)
    candidate=pointer(root,'pending-candidate.json')
    if candidate is None:
        current=pointer(root,'current-candidate.json')
        if not current:raise ValueError('Setup has no activated desktop candidate. Run setup again.')
        return {'runtimeRoot':str(version(root,current)),'packageId':current}
    owner=selected(base)
    journal=read(root/'upgrade.json')
    if journal['format']!='labviz-desktop-upgrade-v1' or journal['appId']!=DESKTOP_ID or journal['packageId']!=candidate or journal['profileRoot']!=str(owner):
        raise ValueError('The prepared upgrade identity differs')
    held=Mutex(lock_name('LabVizDesktopPreview',owner))
    try:
        if journal['backup']:
            owned_backup(owner, journal['backup'], payload)
        try:
            folder=version(root,candidate,True)
            restored = {}
            if journal.get('rollbackSource'):
                maximum = read(folder/'desktop-package.json')['manifest']['dataEpoch']
                owned_backup(owner, journal['rollbackSource'], payload, maximum)
                restored = restore_snapshot(owner, journal['rollbackSource'], payload)
                journal['restored'] = restored
                atomic(root/'upgrade.json', encoded(journal))
            health(folder,owner)
            mark(root,'current-candidate.json',candidate)
            journal['state']='activated'
            previous_receipt = read(root/'last-upgrade.json') if (root/'last-upgrade.json').exists() else {}
            preserve_rollback = (
                journal['previous'] == candidate
                and previous_receipt.get('format') == 'labviz-desktop-upgrade-v1'
                and previous_receipt.get('appId') == DESKTOP_ID
                and previous_receipt.get('state') == 'activated'
                and previous_receipt.get('packageId') == candidate
                and previous_receipt.get('profileRoot') == str(owner)
                and previous_receipt.get('previous') not in (None, candidate)
                and previous_receipt.get('backup')
            )
            if not preserve_rollback:
                atomic(root/'last-upgrade.json',encoded(journal))
            (root/'pending-candidate.json').unlink()
            (root/'upgrade.json').unlink()
            return {'runtimeRoot':str(folder),'packageId':candidate,'activated':True,
                    **({'explicitRollback': True, 'retainedCurrentBackup': journal['backup'], **restored}
                       if journal.get('rollbackSource') else {})}
        except Exception as error:
            if journal['backup']:
                journal['failureRecovery'] = restore_snapshot(owner, journal['backup'], payload)
            journal.update(state='failed',message=str(error))
            atomic(root/'failed-upgrade.json',encoded(journal))
            (root/'pending-candidate.json').unlink()
            (root/'upgrade.json').unlink()
            if journal['previous']:
                previous=version(root,journal['previous'],True)
                mark(root,'current-candidate.json',journal['previous'])
                return {'runtimeRoot':str(previous),'packageId':journal['previous'],'rolledBack':True,'message':str(error)}
            raise
    finally:held.close()


def rollback(root, base, payload):
    """Use the current helper to restore the preceding candidate's pre-upgrade snapshot."""
    root = program(root)
    owner = selected(base)
    current = pointer(root, 'current-candidate.json')
    receipt = read(root/'last-upgrade.json')
    previous = receipt.get('previous')
    if (receipt.get('format') != 'labviz-desktop-upgrade-v1'
            or receipt.get('appId') != DESKTOP_ID or receipt.get('state') != 'activated'
            or receipt.get('packageId') != current or receipt.get('profileRoot') != str(owner)
            or not previous or previous == current):
        raise ValueError('No matching preceding desktop candidate and rollback snapshot are available')
    version(root, current, True)
    target = version(root, previous, True)
    maximum = read(target/'desktop-package.json')['manifest']['dataEpoch']
    saved = receipt.get('backup')
    owned_backup(owner, saved, payload, maximum)
    journal_path = root/'upgrade.json'
    pending = pointer(root, 'pending-candidate.json')
    if journal_path.exists() or pending:
        journal = read(journal_path) if journal_path.exists() else {}
        if (journal.get('rollbackSource') != saved or journal.get('previous') != current
                or journal.get('packageId') != previous or journal.get('profileRoot') != str(owner)
                or pending not in (None, previous)):
            raise ValueError('Complete the existing desktop installation transaction before rolling back')
    else:
        journal = prepare(root, previous, base, payload)
        journal['rollbackSource'] = saved
        atomic(journal_path, encoded(journal))
    mark(root, 'pending-candidate.json', previous)
    return activate(root, base, payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','prepare','activate','rollback','abort','uninstall'])
    parser.add_argument('--install-root',type=Path,required=True)
    parser.add_argument('--profile-base',type=Path,required=True)
    parser.add_argument('--payload',type=Path,required=True)
    parser.add_argument('--package')
    args=parser.parse_args()
    sys.path.insert(0,str(args.payload/'V2.0/api'))
    root=unlinked(args.install_root.absolute());base=unlinked(args.profile_base.absolute())
    if root == base or root.is_relative_to(base) or base.is_relative_to(root):
        raise ValueError('Program files and desktop profiles must be separate')
    held=Mutex(lock_name('LabVizDesktopInstall',root))
    try:
        if args.action=='check':
            program(root, create='check')
            if base.exists() and any(base.iterdir()): identity(base)
            result={'destinationVerified':True,'dataUnchanged':True}
        elif args.action=='prepare':
            if not args.package or not PACKAGE_ID.fullmatch(args.package):raise ValueError('Invalid desktop package')
            result=prepare(root,args.package,base,args.payload)
        elif args.action=='activate':result=activate(root,base,args.payload)
        elif args.action=='rollback':result=rollback(root,base,args.payload)
        elif args.action=='uninstall':
            program(root)
            for path in root.rglob('*'):unlinked(path)
            quiesce(selected(base));result={'profileRetained':True}
        else:
            program(root)
            path=root/'upgrade.json'
            if path.exists() and not (root/'pending-candidate.json').exists() and read(path)['packageId']==args.package:path.unlink()
            result={'aborted':True,'dataRetained':True}
        print(json.dumps(result))
        if args.action == 'rollback' and result.get('rolledBack'):
            raise SystemExit(20)
    finally:held.close()


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'failed':True,'message':str(error)}),file=sys.stderr)
        raise SystemExit(1)
