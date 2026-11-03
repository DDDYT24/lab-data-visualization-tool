"""Bounded desktop data operations. The controller stops writers before execution."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from session import atomic, encoded, unlinked, io_path

DESKTOP_ID = 'org.labviz.desktop.preview'
# Keep in step with DATA_EPOCH in V2.0/api/labviz_api/installed_profile.py.
DATA_EPOCH = 2
HEX = re.compile(r'^[a-f0-9]{32}$')


def read(path):
    return json.loads(unlinked(path).read_text(encoding='utf-8'))


def identity(owner):
    owner = unlinked(Path(owner).absolute())
    if read(owner / 'desktop-profile.json') != {
        'format': 'labviz-desktop-profile-v1', 'appId': DESKTOP_ID, 'root': str(owner)
    }:
        raise ValueError('The desktop profile identity differs')
    return owner


def restored_copy(base, copy_id):
    if not isinstance(copy_id, str) or not HEX.fullmatch(copy_id):
        raise ValueError('Invalid restored copy identity')
    target = identity(base / 'restored-profiles' / copy_id)
    receipt = read(target / 'desktop-restored-copy.json')
    if receipt.get('format') != 'labviz-desktop-restored-copy-v1' or receipt.get('appId') != DESKTOP_ID or receipt.get('copyId') != copy_id or receipt.get('profileId') != read(target / 'appdata/LabVizV23/profile.json')['profileId']:
        raise ValueError('The restored copy has not completed verification')
    return target


@contextmanager
def backend(owner, payload):
    """Reuse installed lifecycle contracts inside this desktop's private AppData."""
    import sys
    source = str(payload / 'V2.0/api')
    if source not in sys.path:
        sys.path.insert(0, source)
    from labviz_api.installed_profile import InstalledProfile
    previous = os.environ.get('LOCALAPPDATA')
    os.environ['LOCALAPPDATA'] = str(owner / 'appdata')
    try:
        yield InstalledProfile.open(owner / 'appdata/LabVizV23')
    finally:
        if previous is None:
            os.environ.pop('LOCALAPPDATA', None)
        else:
            os.environ['LOCALAPPDATA'] = previous


def files(root):
    """Reject linked descendants before reading or copying any of their content."""
    from labviz_api.source_data import safe_files
    return safe_files(unlinked(root))


def digest(path):
    with io_path(unlinked(path)).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint_schema(owner, *, strict=False):
    path = owner / 'session/checkpoint.json'
    if not unlinked(path).exists():
        return 1
    from session import Session
    from labviz_api.installed_profile import InstalledProfile
    profile = read(owner / 'appdata/LabVizV23/profile.json')['profileId']
    check = Session(owner / 'session', profile, lambda _: {}, lambda _: [])
    if strict and check.warning:
        raise ValueError('The desktop checkpoint is incompatible or damaged; original bytes retained')
    return 1 if not check.warning else None


def backup(owner, payload):
    owner = identity(owner)
    with backend(owner, payload) as profile:
        from labviz_api.installed_lifecycle import readonly_source_guard, schema_fingerprint, verify_installed
        from labviz_api.source_data import backup_data
        identifier = uuid.uuid4().hex
        destination = unlinked(owner / 'backups' / identifier)
        destination.mkdir(parents=True, exist_ok=False)
        with profile.lock(), readonly_source_guard(profile.data):
            manifest = backup_data(profile.data, destination / 'data')
            manifest['installed'] = {'appId': '55B46BB7-101E-46DC-951A-3DE7E402EDD0',
                'sourceProfileId': profile.identifier, 'dataEpoch': profile.epoch(),
                'schemaSha256': schema_fingerprint(destination / 'data/labviz.db')}
            atomic(destination / 'data/manifest.json', encoded(manifest))
            verify_installed(destination / 'data')
            for relative in ('preferences.json', 'session'):
                source = unlinked(owner / relative)
                if source.is_dir():
                    for item in files(source):
                        target = unlinked(destination / 'desktop' / item.relative_to(owner))
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target)
                elif source.is_file():
                    target = destination / 'desktop' / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            content = {f.relative_to(destination).as_posix(): digest(f) for f in files(destination)}
            record = {'format': 'labviz-desktop-backup-v1', 'appId': DESKTOP_ID,
                'backupId': identifier, 'profileId': profile.identifier, 'dataEpoch': profile.epoch(),
                'checkpointSchema': checkpoint_schema(owner), 'createdUtcSeconds': time.time(), 'files': content}
            atomic(destination / 'desktop-backup.json', encoded({'manifest': record, 'sha256': hashlib.sha256(encoded(record)).hexdigest()}))
        verify(destination, payload)
        return {'backupId': identifier, 'backupDirectory': str(destination)}


def verify(directory, payload):
    directory = unlinked(directory)
    envelope = read(directory / 'desktop-backup.json')
    record = envelope['manifest']
    if envelope['sha256'] != hashlib.sha256(encoded(record)).hexdigest() or record['format'] != 'labviz-desktop-backup-v1' or record['appId'] != DESKTOP_ID:
        raise ValueError('Desktop backup manifest differs')
    if type(record['dataEpoch']) is not int or not 1 <= record['dataEpoch'] <= DATA_EPOCH:
        raise ValueError('A compatible application is required for this backup epoch')
    actual = {f.relative_to(directory).as_posix(): digest(f) for f in files(directory) if f.name != 'desktop-backup.json' or f.parent != directory}
    if record['files'] != actual:
        raise ValueError('Desktop backup files differ; all copies retained')
    from labviz_api.installed_lifecycle import verify_installed
    data = verify_installed(directory / 'data')
    if data['installed']['sourceProfileId'] != record['profileId']:
        raise ValueError('Backup profile identities differ')
    return record


def restore(directory, base, payload):
    base = identity(base)
    record = verify(directory, payload)
    if record['checkpointSchema'] != 1:
        raise ValueError('Resolve the saved checkpoint warning before restoring this desktop backup')
    identifier = uuid.uuid4().hex
    target = unlinked(base / 'restored-profiles' / identifier)
    target.mkdir(parents=True, exist_ok=False)
    atomic(target / 'desktop-profile.json', encoded({'format': 'labviz-desktop-profile-v1', 'appId': DESKTOP_ID, 'root': str(target)}))
    previous = os.environ.get('LOCALAPPDATA')
    os.environ['LOCALAPPDATA'] = str(target / 'appdata')
    try:
        from labviz_api.installed_profile import InstalledProfile
        from labviz_api.installed_lifecycle import _restore_new
        profile = InstalledProfile.open(target / 'appdata/LabVizV23', create=True)
        with profile.lock():
            _restore_new(directory / 'data', profile)
        source = directory / 'desktop'
        for item in files(source) if source.exists() else []:
            output = unlinked(target / item.relative_to(source))
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, output)
        preferences = target / 'preferences.json'
        if preferences.exists():
            from preferences import checked
            value = checked(read(preferences), record['profileId'])
            value['profileId'] = profile.identifier
            atomic(preferences, encoded(value))
        checkpoint = target / 'session/checkpoint.json'
        if checkpoint.exists():
            value = read(checkpoint)['payload']
            value['profileId'] = profile.identifier
            atomic(checkpoint, encoded({'format': 'labviz-desktop-checkpoint-envelope-v1', 'payload': value, 'sha256': hashlib.sha256(encoded(value)).hexdigest()}))
        checkpoint_schema(target, strict=True)
        atomic(target / 'desktop-restored-copy.json', encoded({'format': 'labviz-desktop-restored-copy-v1', 'appId': DESKTOP_ID, 'copyId': identifier, 'profileId': profile.identifier, 'sourceBackupId': record['backupId']}))
        return {'copyId': identifier, 'restoredProfileRoot': str(target), 'restoredProfileId': profile.identifier}
    finally:
        if previous is None:
            os.environ.pop('LOCALAPPDATA', None)
        else:
            os.environ['LOCALAPPDATA'] = previous


def clear_cache(owner, payload):
    """Remove only the regenerable calculation cache, with the writer stopped."""
    owner = identity(owner)
    with backend(owner, payload) as profile, profile.lock():
        if profile.epoch() > DATA_EPOCH:
            raise ValueError('A compatible runtime is required before clearing this cache')
        target = unlinked(profile.root / 'cache/matplotlib')
        if not target.exists():
            return {'removedFiles': 0, 'removedBytes': 0, 'cache': 'matplotlib'}
        if not target.is_dir():
            raise ValueError('The calculation cache is not a directory')
        # Check the entire tree before the first deletion, including empty junctions.
        members = files(target)
        directories = sorted((unlinked(p) for p in target.rglob('*') if p.is_dir()),
                             key=lambda p: len(p.parts), reverse=True)
        size = sum(p.stat().st_size for p in members)
        for path in members:
            io_path(unlinked(path)).unlink()
        for path in directories:
            io_path(unlinked(path)).rmdir()
        io_path(target).rmdir()
        return {'removedFiles': len(members), 'removedBytes': size, 'cache': 'matplotlib'}


class Maintenance:
    def __init__(self, owner, base, payload):
        self.owner, self.base, self.payload = identity(owner), identity(base), payload
        self.lock = threading.Lock()
        self.pending = None
        self.exit_pending = False
        self.suspension_pending = False
        self.status_path = owner / 'maintenance.json'
        self.result = read(self.status_path) if self.status_path.exists() else None

    def profile_status(self):
        copies = []
        unavailable = 0
        parent = unlinked(self.base / 'restored-profiles')
        if parent.exists():
            for path in sorted(parent.iterdir()):
                if not HEX.fullmatch(path.name):
                    continue
                try:
                    target = restored_copy(self.base, path.name)
                    receipt = read(target / 'desktop-restored-copy.json')
                    copies.append({'copyId': path.name, 'profileId': receipt['profileId']})
                except (ValueError, KeyError, TypeError, OSError):
                    unavailable += 1
        selection = unlinked(self.base / 'profile-selection.json')
        selected = None
        if selection.exists():
            value = read(selection)
            if (set(value) != {'format', 'appId', 'copyId'}
                    or value['format'] != 'labviz-desktop-selection-v1' or value['appId'] != DESKTOP_ID):
                raise ValueError('The selected desktop profile identity differs')
            restored_copy(self.base, value['copyId'])
            selected = value['copyId']
        active = next((row['copyId'] for row in copies
                       if self.owner == self.base / 'restored-profiles' / row['copyId']), None)
        return {'activeCopyId': active, 'nextCopyId': selected, 'copies': copies,
                'unavailableCopies': unavailable}

    def select_copy(self, copy_id):
        with self.lock:
            if self.pending or self.exit_pending or self.suspension_pending:
                raise ValueError('Wait for the current data operation or exit to finish before selecting a profile')
            selection = unlinked(self.base / 'profile-selection.json')
            if copy_id is None:
                if selection.exists():
                    selection.unlink()
            else:
                restored_copy(self.base, copy_id)
                atomic(selection, encoded({'format': 'labviz-desktop-selection-v1',
                    'appId': DESKTOP_ID, 'copyId': copy_id}))
            return {'selected': True, 'nextCopyId': copy_id}

    def submit(self, value):
        if not isinstance(value, dict) or not set(value) <= {'action', 'backupId', 'requestId'} or value.get('action') not in {'backup', 'verify', 'restore', 'prepare-update', 'clear-cache'}:
            raise ValueError('Unknown desktop maintenance operation')
        if value['action'] in {'verify', 'restore'} and not HEX.fullmatch(value.get('backupId', '')):
            raise ValueError('Choose an owned backup')
        with self.lock:
            if self.exit_pending:
                raise ValueError('Exit is pending. Cancel exit before starting data maintenance.')
            if self.suspension_pending:
                raise ValueError('Suspension is pending. Reopen the workspace before starting data maintenance.')
            if self.pending:
                raise ValueError('A desktop data operation is already in progress')
            request_id = value.get('requestId', uuid.uuid4().hex)
            if not HEX.fullmatch(request_id):
                raise ValueError('Invalid data operation identity')
            self.pending = {**value, 'requestId': request_id, 'startedUtcSeconds': time.time()}
            self.result = {**self.pending, 'state': 'waiting'}
            atomic(self.status_path, encoded(self.result))
            return self.result

    def check_exit(self, probe):
        """Fence submissions atomically with the controller's final idle check."""
        with self.lock:
            result = probe(self.pending is not None)
            self.exit_pending = result == 'exit-ready'
            return result

    def reopen_exit(self):
        with self.lock:
            self.exit_pending = False

    def commit_suspension(self, probe, maintenance_mode):
        with self.lock:
            if self.pending is not None and not maintenance_mode:
                return False
            if not probe():
                return False
            self.suspension_pending = True
            return True

    def reopen_suspension(self):
        with self.lock:
            self.suspension_pending = False

    def fail(self, message):
        with self.lock:
            if self.pending:
                self.result = {**self.pending, 'state': 'failed', 'message': str(message)[:1000]}
                atomic(self.status_path, encoded(self.result))
                self.pending = None

    def execute(self):
        job = self.pending
        try:
            if job['action'] in {'backup', 'prepare-update'}:
                result = backup(self.owner, self.payload)
            elif job['action'] == 'clear-cache':
                result = clear_cache(self.owner, self.payload)
            else:
                directory = unlinked(self.owner / 'backups' / job['backupId'])
                if job['action'] == 'verify':
                    with backend(self.owner, self.payload):
                        verify(directory, self.payload)
                    result = {'verified': True, 'backupId': job['backupId']}
                else:
                    with backend(self.owner, self.payload):
                        result = restore(directory, self.base, self.payload)
            with self.lock:
                self.result = {**job, **result, 'state': 'completed'}
                atomic(self.status_path, encoded(self.result))
                self.pending = None
            return self.result
        except Exception as error:
            self.fail(error)
            return self.result

    def handle(self, handler):
        try:
            if handler.command == 'GET' and handler.path == '/maintenance/status':
                entries = []
                parent = unlinked(self.owner / 'backups')
                if parent.exists():
                    for path in sorted(parent.iterdir(), reverse=True):
                        if HEX.fullmatch(path.name) and unlinked(path / 'desktop-backup.json').exists():
                            record = read(path / 'desktop-backup.json')['manifest']
                            entries.append({key: record[key] for key in ('backupId', 'createdUtcSeconds')})
                handler.respond(200, {'operation': self.result, 'backups': entries[:100],
                                     'profiles': self.profile_status()})
            elif handler.command == 'POST':
                count = int(handler.headers.get('Content-Length', '0'))
                if not 0 < count <= 4096:
                    raise ValueError('Invalid maintenance request size')
                value = json.loads(handler.rfile.read(count))
                if handler.path == '/maintenance/select-copy':
                    if set(value) != {'copyId'} or (value['copyId'] is not None and (not isinstance(value['copyId'], str) or not HEX.fullmatch(value['copyId']))):
                        raise ValueError('Choose a restored profile')
                    handler.respond(200, self.select_copy(value['copyId']))
                elif handler.path == '/maintenance/submit':
                    handler.respond(202, self.submit(value))
                else:
                    raise ValueError('Unknown data operation')
            else:
                raise ValueError('Unknown data operation')
        except (ValueError, KeyError, TypeError, OSError) as error:
            handler.respond(409, {'message': str(error)})
