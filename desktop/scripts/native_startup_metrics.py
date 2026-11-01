"""Measure a real native process using its preinstalled readiness observer."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
import psutil


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


class NativeRun:
    def __init__(self, trial, bundle, kind, iteration):
        self.trial, self.iteration, self.known = trial, iteration, {}
        self.cycle = 0
        for name in ('host.json', 'session.json', 'services.json', 'exit.json', 'process-identities.json'):
            previous = trial/name
            if previous.exists():
                previous.rename(trial/f'run-{iteration-1}-{name}')
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        self.config = {'format': 'labviz-desktop-acceptance-v1', 'id': uuid.uuid4().hex,
                       'profileRoot': str(trial/'profile'), 'shell': 'tauri-desktop',
                       'debugPort': port, 'startupTimeoutSeconds': 180, 'idleSeconds': 45,
                       'readyProbe': {'kind': kind, 'expectedTitle': 'D3 final unsaved title' if kind == 'figure' else None}}
        write(trial/'session.json', self.config)
        self.started = time.perf_counter()
        self.process = subprocess.Popen([str(bundle/'LabViz.exe'), '--acceptance', str(trial/'session.json')], creationflags=subprocess.CREATE_NO_WINDOW)
        self.owner = psutil.Process(self.process.pid)
        self.created = self.owner.create_time()
        assert Path(self.owner.exe()).resolve() == (bundle/'LabViz.exe').resolve()
        self.log = Path(os.environ['LOCALAPPDATA'])/'LabVizDesktopPreviewBoot'/f'host-{self.process.pid}.jsonl'
        write(trial/'host.json', {'pid': self.process.pid, 'createdUtcSeconds': self.created,
                                 'executable': str(bundle/'LabViz.exe')})

    def events(self):
        try:
            return [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines() if line]
        except (OSError, ValueError):
            return []

    def tick(self):
        try:
            for member in [self.owner, *self.owner.children(recursive=True)]:
                try:
                    self.known[member.pid] = member.create_time()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
        try:
            write(self.trial/'process-identities.json', self.known)
        except PermissionError:
            # Another test reader can briefly hold this file open on Windows.
            # Keep tracking the identities in memory and retry on the next tick.
            pass
        try:
            record = read(self.trial/'profile/logs/desktop-host.json')
            services = read(self.trial/'profile/logs/desktop-services.json')
            if record['pid'] == self.process.pid and record['origin'] == services['webUrl'] and services.get('cycle', 1) > self.cycle:
                write(self.trial/'services.json', services)
                self.cycle = services.get('cycle', 1)
        except (OSError, ValueError):
            pass

    def command(self, action):
        write(self.trial/'commands.json', {'acceptanceId': self.config['id'], 'id': uuid.uuid4().hex, 'action': action})

    def finish(self):
        time.sleep(3)
        survivors = []
        for pid, created in self.known.items():
            try:
                member = psutil.Process(pid)
                if member.create_time() == created:
                    survivors.append({'pid': pid, 'createdUtcSeconds': created, 'name': member.name()})
            except psutil.NoSuchProcess:
                pass
        result = {'hostExitCode': self.process.wait(), 'survivingOwnedProcesses': survivors}
        write(self.trial/'exit.json', result)
        assert result['hostExitCode'] == 0 and not survivors, result
        return result

    def abort(self, output, error, grace_seconds=30):
        """Bound cleanup to the original creation-identified synthetic process tree."""
        receipt = {'passed': False, 'pid': self.process.pid, 'error': repr(error),
                   'forcedOwnedCleanup': False, 'cleanupErrors': []}
        if self.process.poll() is None:
            try:
                self.command('exit')
            except OSError as cause:
                receipt['cleanupErrors'].append(repr(cause))
            deadline = time.monotonic() + grace_seconds
            while self.process.poll() is None and time.monotonic() < deadline:
                try:
                    self.tick()
                except (OSError, psutil.Error) as cause:
                    receipt['cleanupErrors'].append(repr(cause))
                time.sleep(.1)
        if self.process.poll() is None:
            receipt['forcedOwnedCleanup'] = True
            try:
                assert self.owner.create_time() == self.created
                children = self.owner.children(recursive=True)
                for child in reversed(children):
                    try:
                        if self.known.get(child.pid) == child.create_time():
                            child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                self.owner.terminate()
                self.process.wait(timeout=15)
            except (OSError, psutil.Error, subprocess.TimeoutExpired, AssertionError) as cause:
                receipt['cleanupErrors'].append(repr(cause))
        time.sleep(.5)
        survivors = []
        for pid, created in self.known.items():
            try:
                member = psutil.Process(pid)
                if member.create_time() == created:
                    survivors.append({'pid': pid, 'createdUtcSeconds': created, 'name': member.name()})
            except psutil.NoSuchProcess:
                pass
        receipt.update(hostExitCode=self.process.poll(), survivingOwnedProcesses=survivors)
        write(output/f'run-{self.iteration}-failure-cleanup.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--kind', required=True, choices=['home', 'figure'])
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--leave-running', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert args.label.replace('-', '').isalnum() and 1 <= args.runs <= 10
    bundle = args.bundle.resolve()
    trial = Path.home()/'LabVizD2'/args.label
    trial.mkdir(exist_ok=False)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.seed:
        seed = args.seed.resolve()
        assert seed.is_relative_to((Path.home()/'LabVizD2').resolve())
        previous = read(seed/'exit.json')
        assert previous['hostExitCode'] in (0, 15) and not previous['survivingOwnedProcesses']
        for name in ('appdata', 'session'):
            shutil.copytree(seed/'profile'/name, trial/'profile'/name)
        for name in ('desktop-profile.json', 'preferences.json'):
            shutil.copyfile(seed/'profile'/name, trial/'profile'/name)
        for marker, destination in [(trial/'profile/desktop-profile.json', trial/'profile'), (trial/'profile/appdata/LabVizV23/profile.json', trial/'profile/appdata/LabVizV23')]:
            value = read(marker)
            value['root'] = str(destination)
            write(marker, value)
        shutil.copyfile(seed/'export-links.json', trial/'export-links.json')
    package = read(bundle/'desktop-package.json')
    report = {'format': 'labviz-native-cold-start-v2', 'packageId': package['packageId'], 'kind': args.kind,
              'nativeAcceptance': True, 'passed': False, 'runs': [], 'cdpAttachedBeforeReady': False,
              'clockMethod': 'OS process creation time plus native Rust Instant elapsed from main entry to the preinstalled renderer signal receipt; renderer emission timestamp also retained',
              'cacheScope': 'Process cold launches on this Windows host; no OS file-cache purge. First launch has a new WebView profile; later launches retain browser and filesystem caches.',
              'driverSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for iteration in range(1, args.runs+1):
        run = NativeRun(trial, bundle, args.kind, iteration)
        try:
            deadline = time.monotonic()+180
            signal = None
            while run.process.poll() is None and time.monotonic() < deadline:
                run.tick()
                signal = next((event for event in run.events() if event['event'] == 'acceptance-editor-ready' and event['sequence'] == 1 and event['kind'] == args.kind), None)
                if signal:
                    break
                time.sleep(.025)
            if signal is None:
                write(output/f'run-{iteration}-failed.json', {'passed': False, 'readinessSignalAbsent': True, 'pid': run.process.pid, 'events': run.events()})
                raise RuntimeError('No native readiness signal; original logs retained')
            observed = (time.perf_counter()-run.started)*1000
            startup = (signal['nativeEntryUtcSeconds']-run.created)*1000+signal['nativeEntryToReceiptMs']
            renderer = (signal['rendererUtcSeconds']-run.created)*1000
            drift = (signal['utcSeconds']-signal['nativeEntryUtcSeconds'])*1000-signal['nativeEntryToReceiptMs']
            assert abs(drift) < 50 and 0 <= renderer <= startup+50 and startup >= 0
            # Readiness timing is already finished; service metadata is collected afterwards.
            until = time.monotonic()+15
            while not (trial/'services.json').exists():
                assert time.monotonic() < until
                run.tick()
                time.sleep(.025)
            entry = {'iteration': iteration, 'pid': run.process.pid, 'processCreatedUtcSeconds': run.created,
                     'nativeLaunchToEditableMs': startup, 'rendererLaunchToEditableMs': renderer,
                     'driverObservedUpperBoundMs': observed, 'nativeClockDriftMs': drift,
                     'nativeReceiptAfterRendererMs': (signal['utcSeconds']-signal['rendererUtcSeconds'])*1000,
                     'serviceReadyMs': read(trial/'services.json')['serviceReadyMs'], 'signal': signal}
            timing_file = trial/'profile/appdata/LabVizV23/logs/desktop-startup-timing.json'
            if timing_file.exists():
                entry['serviceTimingDetail'] = read(timing_file)
            report['runs'].append(entry)
            write(output/f'run-{iteration}.json', entry)
            print(json.dumps({'kind': args.kind, 'iteration': iteration, 'nativeLaunchToEditableMs': startup}), flush=True)
            if args.leave_running and iteration == args.runs:
                report['passed'] = True
                report['finalNativeLeftForFollowup'] = True
                write(output/'cold-start.json', report)
                while run.process.poll() is None:
                    run.tick()
                    time.sleep(.1)
                write(output/'final-shutdown.json', run.finish())
            else:
                run.command('exit')
                while run.process.poll() is None:
                    run.tick()
                    time.sleep(.1)
                write(output/f'run-{iteration}-shutdown.json', run.finish())
        except BaseException as error:
            run.abort(output, error)
            raise
    report['passed'] = True
    report['finalNativeLeftForFollowup'] = False
    write(output/'cold-start.json', report)


if __name__ == '__main__':
    main()
