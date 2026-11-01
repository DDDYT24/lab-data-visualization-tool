"""Run the formal GUI with an isolated synthetic acceptance profile."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--idle', type=int, default=45)
    parser.add_argument('--bundle', type=Path, default=ROOT/'outputs/desktop-d02/bundle')
    args = parser.parse_args()
    assert args.label.replace('-', '').isalnum()
    trial = Path.home() / 'LabVizD2' / args.label
    trial.mkdir(parents=True, exist_ok=args.restart)
    if args.restart:
        old_exit = json.loads((trial / 'exit.json').read_text())
        assert old_exit['hostExitCode'] == 0 and not old_exit['survivingOwnedProcesses']
        stamp = str(int(time.time()))
        for name in ('exit.json', 'host.json', 'services.json', 'process-identities.json'):
            source = trial / name
            if source.exists():
                source.rename(trial / (stamp + '-' + name))
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        port = server.getsockname()[1]
    config = {'format': 'labviz-desktop-acceptance-v1', 'id': uuid.uuid4().hex, 'profileRoot': str(trial / 'profile'), 'debugPort': port, 'startupTimeoutSeconds': args.timeout, 'idleSeconds':args.idle, 'shell': 'tauri-desktop'}
    write(trial / 'session.json', config)
    write(trial / 'launcher.json', config)
    host = args.bundle.resolve() / 'LabViz.exe'
    started = time.time()
    process = subprocess.Popen([str(host), '--acceptance', str(trial / 'session.json')], creationflags=subprocess.CREATE_NO_WINDOW)
    write(trial / 'host.json', {'pid': process.pid, 'startedUtcSeconds': started})
    print(json.dumps({'trial': str(trial), 'pid': process.pid, 'debugPort': port}), flush=True)
    known = {}
    observed_cycle = 0
    while process.poll() is None:
        try:
            for member in [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]:
                try:
                    known[member.pid] = member.create_time()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            break
        write(trial / 'process-identities.json', known)
        status = trial / 'profile/logs/desktop-host.json'
        if status.exists():
            record = json.loads(status.read_text())
            if record['pid'] == process.pid:
                services = json.loads((trial / 'profile/logs/desktop-services.json').read_text())
                cycle = services.get('cycle', 1)
                if cycle > observed_cycle and record['origin'] == services['webUrl']:
                    write(trial / 'services.json', services)
                    print(json.dumps({'ready': True, 'origin': record['origin'], 'profileId': record['profileId'], 'cycle':cycle}), flush=True)
                    observed_cycle = cycle
        entries=[]
        for pid, creation in known.items():
            try:
                member=psutil.Process(pid)
                if member.create_time()==creation:
                    memory=member.memory_info()
                    entries.append({'pid':pid,'name':member.name(),'privateCommitBytes':memory.private,'workingSetBytes':memory.rss})
            except psutil.NoSuchProcess:
                pass
        with (trial/'resources.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'utcSeconds':time.time(),'cycle':observed_cycle,'privateCommitBytes':sum(p['privateCommitBytes'] for p in entries),'processes':entries})+'\n')
        time.sleep(.25)
    time.sleep(3)
    survivors = []
    for pid, creation in known.items():
        try:
            member = psutil.Process(pid)
            if member.create_time() == creation:
                survivors.append({'pid': pid, 'name': member.name()})
        except psutil.NoSuchProcess:
            pass
    result = {'hostExitCode': process.wait(), 'survivingOwnedProcesses': survivors, 'elapsedSeconds': time.time()-started}
    write(trial / 'exit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
