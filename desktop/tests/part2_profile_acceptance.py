"""Launch selected synthetic profiles with bundled services, without a native window."""
from contextlib import contextmanager, closing
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

ROOT = Path(__file__).resolve().parents[2]
trial = Path(sys.argv[1]).resolve()
bundle = Path(sys.argv[2]).resolve()
assert trial.is_relative_to(Path.home() / 'LabVizD2')
assert bundle.is_relative_to(ROOT / 'outputs/desktop-d06')
assert json.loads((trial / 'session.json').read_text(encoding='utf-8-sig'))['shell'] == 'headless-test-adapter'
exit_record = json.loads((trial / 'exit.json').read_text(encoding='utf-8-sig'))
assert exit_record['hostExitCode'] == 0 and not exit_record['survivingOwnedProcesses']
sys.path.insert(0, str(bundle / 'runtime'))
sys.path.insert(0, str(bundle / 'app/V2.0/api'))
import installation
import maintenance
from controller import ApiControl, local_key
from session import atomic, encoded
from process_job import OwnedJob

base = trial / 'profile'
headless = json.loads((trial / 'part2-headless.json').read_text(encoding='utf-8-sig'))
assert headless['passed']
copies = headless['copies']
report = {'passed': False, 'runId': uuid.uuid4().hex, 'packageId': headless['packageId'], 'scope': 'Bundled services and synthetic profiles; no native UI', 'checks': []}


def check(name):
    report['checks'].append(name)
    (trial / 'part2-profiles.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(name, flush=True)


def scientific(owner):
    with closing(sqlite3.connect((owner / 'appdata/LabVizV23/data/labviz.db').as_uri() + '?mode=ro', uri=True)) as db:
        return {name: hashlib.sha256(json.dumps(db.execute('SELECT * FROM ' + name + ' ORDER BY rowid').fetchall(), default=lambda b: b.hex(), sort_keys=True).encode()).hexdigest()
                for name in ('projects', 'project_description_revisions', 'exports', 'figure_snapshots', 'research_heads', 'research_revisions', 'research_charts', 'research_figure_boards', 'research_outputs')}


def files(owner):
    return {p.relative_to(owner).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (owner / 'appdata/LabVizV23/data/objects').rglob('*') if p.is_file()}


@contextmanager
def services(owner, label):
    installation.initialise(owner)
    metadata = json.loads((bundle / 'app/candidate-build.json').read_text(encoding='utf-8-sig'))
    config = owner / ('part2-' + label + '.json')
    atomic(config, encoded({'payloadRoot': str(bundle / 'app'), 'backendProfile': str(owner / 'appdata/LabVizV23'),
        'programRoot': str(owner / 'program'), 'candidateId': metadata['candidateId']}))
    environment = {**os.environ, 'LOCALAPPDATA': str(owner / 'appdata'), 'LABVIZ_DESKTOP_MODE': '1',
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX', 'NODE_PATH'):
        environment.pop(name, None)
    with (trial / ('part2-' + label + '-' + report['runId'] + '.log')).open('xb') as log:
        child = subprocess.Popen([str(bundle / 'app/runtime/python/python.exe'), '-B', '-X', 'utf8', str(bundle / 'runtime/service_runner.py'), str(config)],
            env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        job = OwnedJob(child)
        ready = None
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                assert child.poll() is None, 'Bundled service exited before readiness; inspect the run log'
                record = owner / 'appdata/LabVizV23/logs/running.json'
                value = json.loads(record.read_text(encoding='utf-8-sig')) if record.exists() else {}
                if value.get('launcher', {}).get('pid') == child.pid:
                    ready = value
                    break
                time.sleep(.1)
            assert ready
            key = local_key(owner / 'appdata/LabVizV23/access.dpapi')
            api = ApiControl(ready, key)
            api.verify_api()
            yield ready, key, api
        finally:
            if ready:
                atomic(owner / 'appdata/LabVizV23/logs/stop.request', encoded({'profileId': ready['profileId'], 'sessionId': ready['sessionId']}))
            try:
                child.wait(timeout=65)
            finally:
                job.close()
            if ready:
                assert child.returncode == 0
                assert json.loads((owner / 'appdata/LabVizV23/logs/desktop-service-exit.json').read_text(encoding='utf-8-sig'))['normal']


try:
    original = scientific(base), files(base)
    second = Path(copies[1]['root'])
    second_before = scientific(second), files(second)
    selection = maintenance.Maintenance(base, base, bundle / 'app')
    selection.select_copy(copies[0]['copyId'])
    first = installation.selected(base)
    assert first == Path(copies[0]['root'])
    first_before = scientific(first)
    with services(first, 'copy-import') as (ready, key, api):
        assert ready['profileId'] == copies[0]['profileId']
        boundary = 'LabVizPart2' + uuid.uuid4().hex
        raw = (ROOT / 'outputs/desktop-part2-20261028/before/fixtures/desktop-small.csv').read_bytes()
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="independent.csv"\r\nContent-Type: text/csv\r\n\r\n').encode() + raw + f'\r\n--{boundary}--\r\n'.encode()
        request = urllib.request.Request(f"http://127.0.0.1:{ready['apiPort']}/api/v1/projects", data=body,
            headers={'Content-Type': 'multipart/form-data; boundary=' + boundary,
                     'Cookie': f"labviz_local_session_v23_installed_{ready['profileId']}={key}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            assert response.status == 202
        deadline = time.monotonic() + 120
        while not api.call('activity')['idle'] and time.monotonic() < deadline:
            time.sleep(.1)
        assert api.call('activity')['idle']
    assert scientific(first)['projects'] != first_before['projects']
    assert (scientific(base), files(base)) == original
    assert (scientific(second), files(second)) == second_before
    check('selected-copy-authenticated-import-keeps-original-and-other-copy-unchanged')
    changed = scientific(first), files(first)
    selection.select_copy(copies[1]['copyId'])
    with services(installation.selected(base), 'second-copy') as (ready, _, _):
        assert ready['profileId'] == copies[1]['profileId']
    assert (scientific(second), files(second)) == second_before
    check('second-copy-starts-with-its-own-profile-identity')
    selection.select_copy(None)
    assert installation.selected(base) == base
    with services(base, 'original') as (ready, _, _):
        assert ready['profileId'] not in [item['profileId'] for item in copies]
    assert (scientific(base), files(base)) == original
    assert (scientific(first), files(first)) == changed
    check('switch-back-original-and-copy-retain-independent-scientific-data-and-frozen-files')
    report['passed'] = True
except Exception as error:
    report['error'] = repr(error)
    raise
finally:
    (trial / 'part2-profiles.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
