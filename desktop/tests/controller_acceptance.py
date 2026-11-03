"""Exercise real bundled controller startup, failure cleanup and profile ownership."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / 'outputs/desktop-d34/bundle/app'
TRIAL = Path.home() / 'LabVizD2' / ('controller-' + uuid.uuid4().hex[:8])
TRIAL.mkdir(parents=True)
candidate = json.loads((PAYLOAD / 'candidate-build.json').read_text())['candidateId']
results = []


def run(name, timeout=180, unrelated=False, crash=False):
    profile = TRIAL / name
    if unrelated:
        profile.mkdir()
        (profile / 'keep.txt').write_text('unrelated')
    config = TRIAL / (name + '.json')
    config.write_text(json.dumps({'format': 'labviz-desktop-session-v1', 'profileRoot': str(profile), 'payloadRoot': str(PAYLOAD), 'candidateId': candidate, 'startupTimeoutSeconds': timeout}))
    child = subprocess.Popen([str(PAYLOAD/'runtime/python/python.exe'), '-B', '-X', 'utf8', str(ROOT/'desktop/runtime/controller.py'), str(config)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    messages = queue.Queue()
    def consume():
        for line in child.stdout:
            messages.put(json.loads(line))
    threading.Thread(target=consume, daemon=True).start()
    known = {}
    ready = False
    failed = False
    deadline = time.monotonic()+220
    while child.poll() is None and time.monotonic()<deadline:
        try:
            for member in psutil.Process(child.pid).children(recursive=True):
                known[member.pid] = member.create_time()
        except psutil.NoSuchProcess:
            pass
        try:
            value = messages.get(timeout=.1)
            if value['type']=='ready':
                ready = True
                # Secret-bearing ready messages remain in memory only.
                assert value['candidateId']==candidate
                if crash:
                    child.kill()
                else:
                    child.stdin.write('stop\n')
                    child.stdin.flush()
            if value['type']=='failed':
                failed = True
        except queue.Empty:
            pass
    assert child.poll() is not None, 'Controller deadline exceeded'
    time.sleep(2)
    survivors = []
    for pid, creation in known.items():
        try:
            if psutil.Process(pid).create_time()==creation:
                survivors.append(pid)
        except psutil.NoSuchProcess:
            pass
    assert not survivors, survivors
    if unrelated:
        assert (profile/'keep.txt').read_text()=='unrelated'
        assert not (profile/'desktop-profile.json').exists()
    elif not crash:
        receipt = json.loads((profile/'logs/desktop-controller-exit.json').read_text())
        if ready:
            assert receipt['runningRecordRemoved']
    assert ready if timeout>1 and not unrelated else failed
    result = {'case': name, 'ready': ready, 'failureObserved': failed, 'ownedSurvivors': survivors, 'exitCode': child.returncode}
    results.append(result)
    print(json.dumps(result), flush=True)


if '--crash-only' in sys.argv:
    run('controller-loss', crash=True)
else:
    run('timeout', timeout=1)
    run('normal')
    run('unrelated', unrelated=True)
(TRIAL/'results.json').write_text(json.dumps(results, indent=2))
print(json.dumps({'passed': True, 'trial': str(TRIAL)}))
