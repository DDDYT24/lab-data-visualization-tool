"""Install an older Setup, upgrade it with a newer one, repair, roll back and uninstall.

Usage, from the repository root:
    python -B -X utf8 desktop/scripts/release_upgrade.py --old-build <old candidate>/build
        --new-build <new candidate>/build --old-package <package id> --old-setup-sha <sha256>
        --output <new folder> [--label <label>] [--trial-root <folder>]

The old Setup is named by its package id and Setup SHA-256, so a wrong file is refused. Each
build's data epoch is read from its own package manifest. Everything runs silently in a fresh
synthetic installation under <trial root>/<label>, and the checks go to <output>/verification.json.
This does not open a native window or sign off any native interaction. The newer installed helper
performs the rollback; the older Setup is used unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import urllib.request
import uuid
import winreg

APP_ID = 'org.labviz.desktop.preview'
INSTALL_ID = 'CA2B3AF4-BFB0-4541-8EA5-2E978F105D63'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def write(path, value):
    temporary = path.with_name(path.name+'.new-'+uuid.uuid4().hex)
    temporary.write_bytes(encoded(value))
    os.replace(temporary, path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unlinked(path):
    path = path.absolute()
    for item in (path, *path.parents):
        assert not item.is_symlink() and not item.is_junction(), str(item)
    return path.resolve()


def registrations():
    key = 'Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{'+INSTALL_ID+'}_is1'
    found = []
    for hive, label in ((winreg.HKEY_CURRENT_USER, 'HKCU'), (winreg.HKEY_LOCAL_MACHINE, 'HKLM')):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(hive, key, 0, winreg.KEY_READ|view) as entry:
                    found.append({'hive': label, 'view': view, 'root': winreg.QueryValueEx(entry, 'InstallLocation')[0]})
            except FileNotFoundError:
                pass
    return found


def artifact(build):
    build = unlinked(build)
    bundle = build.parent/'bundle'
    record = read(build/'build-verification.json')
    envelope = read(bundle/'desktop-package.json')
    manifest = envelope['manifest']
    assert record['compilerExitCode'] == 0 and len(record['installers']) == 1
    assert envelope['packageId'] == record['packageId'] == hashlib.sha256(encoded(manifest)).hexdigest()
    epoch = manifest['dataEpoch']
    assert manifest['appId'] == APP_ID and read(bundle/'app/candidate-build.json')['dataEpoch'] == epoch
    name, digest = next(iter(record['installers'].items()))
    executable = build/name
    assert sha(executable) == digest['sha256'] and executable.stat().st_size == digest['bytes']
    return {'bundle': bundle, 'record': record, 'envelope': envelope, 'exe': executable, 'sha256': digest['sha256'], 'epoch': epoch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-build', type=Path, required=True)
    parser.add_argument('--new-build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='p2upgrade-'+uuid.uuid4().hex[:8])
    parser.add_argument('--trial-root', type=Path, default=Path.home()/'LabVizD2')
    parser.add_argument('--old-package', required=True, help='package id of the old candidate')
    parser.add_argument('--old-setup-sha', required=True, help='SHA-256 of the old Setup')
    args = parser.parse_args()
    assert args.label.replace('-', '').isalnum()
    assert not registrations(), 'A real desktop installation is present; do not adopt it.'
    old_id = args.old_package
    old, new = artifact(args.old_build), artifact(args.new_build)
    old_epoch, new_epoch = old['epoch'], new['epoch']
    assert old['record']['packageId'] == old_id and old['sha256'] == args.old_setup_sha
    assert new['record']['packageId'] != old_id and new_epoch >= old_epoch
    output = unlinked(args.output)
    output.mkdir(parents=True, exist_ok=False)
    trial = unlinked(args.trial_root/args.label)
    trial.mkdir(parents=True, exist_ok=False)
    program, profile = trial/'installed', trial/'profile'
    write(trial/'acceptance.json', {'format': 'labviz-desktop-installer-acceptance-v1', 'appId': APP_ID, 'root': str(trial)})
    report = {'passed': False, 'inProgress': True, 'trial': str(trial), 'program': str(program),
              'oldPackageId': old_id, 'oldEpoch': old_epoch, 'newPackageId': new['record']['packageId'], 'newEpoch': new_epoch,
              'oldInstallerSha256': old['sha256'], 'newInstallerSha256': new['sha256'],
              'scope': 'Exact silent Setup and bundled services in a synthetic profile; no native UI, real shortcut or system component installation.',
              'nativeAcceptance': False, 'checks': []}
    def save():
        write(output/'verification.json', report)
    def check(name, **details):
        report['checks'].append({'name': name, **details})
        save()
        print(name, flush=True)
    save()
    environment = {**os.environ, 'PATH': os.pathsep.join((os.environ['SystemRoot']+'\\System32', os.environ['SystemRoot'], os.environ['SystemRoot']+'\\System32\\WindowsPowerShell\\v1.0'))}
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX', 'NODE_PATH'):
        environment.pop(name, None)
    def execute(command, label, timeout=600):
        with (output/(label+'.stdout.log')).open('xb') as stream:
            result = subprocess.run([str(item) for item in command], env=environment, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=stream, creationflags=subprocess.CREATE_NO_WINDOW,
                                    timeout=timeout, check=False)
        assert result.returncode == 0, (label, result.returncode)
    def current():
        value = read(program/'current-candidate.json')
        assert value['appId'] == APP_ID
        return value['packageId']
    def install(item, label):
        execute([item['exe'], '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', '/NOICONS', '/TASKS=!desktopicon',
                 '/DIR='+str(program), '/ACCEPTANCEROOT='+str(trial), '/LOG='+str(output/(label+'.inno.log'))], label)
        assert current() == item['record']['packageId']
        entries = registrations()
        assert entries and all(entry['hive'] == 'HKCU' and Path(entry['root']).resolve() == program for entry in entries)
        folder = program/'versions'/current()[:24]
        envelope = read(folder/'desktop-package.json')
        assert envelope == item['envelope']
        with ThreadPoolExecutor(max_workers=8) as pool:
            matches = pool.map(lambda item: sha(folder/item[0]) == item[1], envelope['manifest']['files'].items())
            assert all(matches)
        return folder
    def epoch():
        return read(profile/'appdata/LabVizV23/data-schema.json')['dataEpoch']
    def science(data=None):
        data = data or profile/'appdata/LabVizV23/data'
        with closing(sqlite3.connect((data/'labviz.db').as_uri()+'?mode=ro', uri=True)) as db:
            columns = [item[1] for item in db.execute('PRAGMA table_info(projects)')]
            fields = [name for name in ('id', 'title', 'current_revision_id', 'source_json', 'storage_mode', 'experiment_run_id') if name in columns]
            records = db.execute('SELECT '+','.join(fields)+' FROM projects ORDER BY id').fetchall()
        return {str(row[0]): dict(zip(fields, row)) for row in records}
    def draft_hashes(owner=profile):
        members = list((owner/'session').rglob('*'))
        if (owner/'preferences.json').exists():
            members.append(owner/'preferences.json')
        return {p.relative_to(owner).as_posix(): sha(p) for p in members if p.is_file()}
    def draft(label):
        identifier = read(profile/'appdata/LabVizV23/profile.json')['profileId']
        content = ('sample,value\n'+label+',4.5\n').encode()
        digest = hashlib.sha256(content).hexdigest()
        files = profile/'session/files'
        files.mkdir(parents=True, exist_ok=True)
        (files/digest).write_bytes(content)
        snapshot = {'format': 'labviz-desktop-checkpoint-v1', 'profileId': identifier, 'checkpointId': uuid.uuid4().hex, 'route': '/',
                    'entries': {'home': {'version': 1, 'projectId': None, 'base': {}, 'value': {'file': {'__labvizFile': digest, 'size': len(content), 'name': label+'.csv', 'type': 'text/csv', 'lastModified': 1}}}}}
        write(profile/'session/checkpoint.json', {'format': 'labviz-desktop-checkpoint-envelope-v1', 'payload': snapshot, 'sha256': hashlib.sha256(encoded(snapshot)).hexdigest()})
        write(profile/'preferences.json', {'format': 'labviz-desktop-preferences-v1', 'profileId': identifier, 'locale': 'en', 'storage': {}})
        return digest
    sys.path.insert(0, str(new['bundle']/'runtime'))
    from process_job import OwnedJob
    from controller import ApiControl, local_key
    @contextmanager
    def services(folder, label):
        config = profile/('driver-'+label+'.json')
        metadata = read(folder/'app/candidate-build.json')
        write(config, {'payloadRoot': str(folder/'app'), 'backendProfile': str(profile/'appdata/LabVizV23'),
                       'programRoot': str(profile/'program'), 'candidateId': metadata['candidateId']})
        env = {**environment, 'LOCALAPPDATA': str(profile/'appdata'), 'LABVIZ_DESKTOP_MODE': '1',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
        stream = (output/(label+'.service.log')).open('xb')
        child = subprocess.Popen([str(folder/'app/runtime/python/python.exe'), '-B', '-X', 'utf8', str(folder/'runtime/service_runner.py'), str(config)],
                                 env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, creationflags=subprocess.CREATE_NO_WINDOW)
        job = OwnedJob(child)
        ready = None
        try:
            deadline = time.monotonic()+180
            while time.monotonic() < deadline:
                assert child.poll() is None, 'Bundled services failed: '+label
                running = profile/'appdata/LabVizV23/logs/running.json'
                value = read(running) if running.exists() else {}
                if value.get('launcher', {}).get('pid') == child.pid and value.get('candidateId') == metadata['candidateId']:
                    ready = value
                    break
                time.sleep(.15)
            assert ready, 'No matching service readiness: '+label
            key = local_key(profile/'appdata/LabVizV23/access.dpapi')
            control = ApiControl(ready, key)
            control.verify_api()
            def request(suffix, body=None, content_type=None):
                control.verify_api()
                headers = {'Cookie': f"labviz_local_session_v23_installed_{ready['profileId']}={key}"}
                if content_type:
                    headers['Content-Type'] = content_type
                call = urllib.request.Request(f"http://127.0.0.1:{ready['apiPort']}/api/v1/"+suffix, data=body, headers=headers)
                with urllib.request.urlopen(call, timeout=90) as response:
                    return json.loads(response.read())
            yield request, control
        finally:
            if ready:
                write(profile/'appdata/LabVizV23/logs/stop.request', {'profileId': ready['profileId'], 'sessionId': ready['sessionId']})
            try:
                child.wait(timeout=90)
            finally:
                job.close()
                stream.close()
            assert child.returncode == 0, 'Bundled service exit failed: '+label
            stopped = read(profile/'appdata/LabVizV23/logs/desktop-service-exit.json')
            assert stopped['normal'] and not stopped['forced']
    def upload(folder, label):
        before = set(science())
        with services(folder, label) as (request, control):
            boundary = 'LabVizEpoch'+uuid.uuid4().hex
            content = b'x,y\n'+b''.join(f'{i},{i*2}\n'.encode() for i in range(1, 13))
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{label}.csv"\r\nContent-Type: text/csv\r\n\r\n').encode()+content+f'\r\n--{boundary}--\r\n'.encode()
            request('projects', body, 'multipart/form-data; boundary='+boundary)
            deadline = time.monotonic()+120
            while time.monotonic() < deadline and not control.call('activity')['idle']:
                time.sleep(.15)
            assert control.call('activity')['idle']
            new_ids = set(science())-before
            assert len(new_ids) == 1
            identifier = next(iter(new_ids))
            preview = request('projects/'+identifier+'/preview')
            assert preview['totalRows'] == 12 and len(preview['rows']) == 12
        return identifier, preview
    def previews(folder, label, identifiers):
        with services(folder, label) as (request, _):
            return {identifier: request('projects/'+identifier+'/preview') for identifier in identifiers}
    try:
        old_folder = install(old, 'old-setup-install')
        assert epoch() == old_epoch
        first_project, first_preview = upload(old_folder, 'before-upgrade')
        old_source = draft('draft-before-upgrade')
        frozen = profile/'appdata/LabVizV23/data/objects/epoch-acceptance-frozen.bin'
        # The epoch-one package keeps rendered artifacts in SQLite and may not
        # have created an external objects directory yet.
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(b'Frozen synthetic output, preserved across epoch migration.\n')
        frozen_hash = sha(frozen)
        baseline, baseline_drafts = science(), draft_hashes()
        check('old-setup-and-synthetic-project', projectId=first_project, draftSourceSha256=old_source, frozenSha256=frozen_hash)
        new_folder = install(new, 'new-setup-upgrade')
        assert epoch() == new_epoch and science() == baseline and draft_hashes() == baseline_drafts and sha(frozen) == frozen_hash
        assert previews(new_folder, 'upgraded-preview', [first_project])[first_project] == first_preview
        upgrade_receipt = (program/'last-upgrade.json').read_bytes()
        check('new-setup-upgrade-preserves-data-and-draft')
        second_project, second_preview = upload(new_folder, 'after-upgrade')
        new_source = draft('draft-after-upgrade')
        upgraded, upgraded_drafts = science(), draft_hashes()
        install(new, 'new-setup-same-version-repair')
        assert epoch() == new_epoch and science() == upgraded and draft_hashes() == upgraded_drafts
        assert (program/'last-upgrade.json').read_bytes() == upgrade_receipt
        check('same-version-setup-repair-preserves-the-original-rollback-baseline')
        rollback_log = output/'explicit-rollback.stdout.log'
        execute([new_folder/'app/runtime/python/python.exe', '-B', '-X', 'utf8', new_folder/'runtime/installation.py', 'rollback',
                 '--install-root', program, '--profile-base', profile, '--payload', new_folder/'app'], 'explicit-rollback', timeout=600)
        rollback = json.loads(rollback_log.read_text(encoding='utf-8-sig').splitlines()[-1])
        assert rollback['explicitRollback'] and rollback['packageId'] == old_id and current() == old_id
        assert epoch() == old_epoch and science() == baseline and draft_hashes() == baseline_drafts and sha(frozen) == frozen_hash
        retained_data = unlinked(Path(rollback['retainedDataDirectory']))
        retained_desktop = unlinked(Path(rollback['retainedDesktopDirectory']))
        assert retained_data.parent == profile/'appdata/LabVizV23' and retained_data.name.startswith('data.retained-')
        assert retained_desktop.parent == profile and retained_desktop.name.startswith('desktop.retained-')
        assert science(retained_data) == upgraded and draft_hashes(retained_desktop) == upgraded_drafts
        retained_backup = unlinked(Path(rollback['retainedCurrentBackup']['backupDirectory']))
        assert retained_backup.is_relative_to(profile/'backups')
        assert read(retained_backup/'desktop-backup.json')['manifest']['dataEpoch'] == new_epoch
        assert science(retained_backup/'data') == upgraded
        assert (retained_desktop/'session/files'/new_source).is_file()
        assert sha(retained_data/'objects/epoch-acceptance-frozen.bin') == frozen_hash
        assert previews(old_folder, 'rolled-back-preview', [first_project])[first_project] == first_preview
        assert second_project not in science() and second_project in science(retained_data)
        check('current-installed-helper-restores-the-old-package-and-retains-the-new-work',
              oldProjectId=first_project, retainedNewProjectId=second_project, retainedNewPreviewSha256=hashlib.sha256(encoded(second_preview)).hexdigest(),
              rollback=rollback)
        snapshots = draft_hashes()
        scientific = science()
        execute([program/'unins000.exe', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/ACCEPTANCEROOT='+str(trial),
                 '/LOG='+str(output/'final-uninstall.inno.log')], 'final-uninstall')
        assert not registrations() and not (program/'LabViz.exe').exists()
        assert science() == scientific and draft_hashes() == snapshots
        check('final-uninstall-retains-active-and-retained-profiles-and-clears-test-registration')
        report.update(passed=True, inProgress=False, registrationRemoved=True, retainedProfile=str(profile))
        save()
    except BaseException as error:
        report.update(inProgress=False, error=repr(error), registrationsAtFailure=registrations())
        save()
        raise


if __name__ == '__main__':
    main()
