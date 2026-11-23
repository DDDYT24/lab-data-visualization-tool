"""Release acceptance of one frozen desktop candidate, without a native window.

Usage, from the repository root:
    python -B -X utf8 desktop/scripts/release_acceptance.py <stage> --candidate <folder> --label <label>

Stages:
    session    headless host; the full scientific workflow (15 checks, 17 exports); the part 2
               headless and profile checks; a normal exit with no surviving process. Then writes
               <candidate>/background-evidence.json, which compiling the Setup requires.
    installer  needs the compiled Setup and the label of a passed session: silent install, repair,
               update and rollback; retention after upgrade; recovery from a failed update
    large      the 100,000-row workflow and a normal exit

The candidate folder holds bundle/ and, once frozen, build/; the acceptance scripts require it
under outputs/desktop-d06 of this worktree. Logs and results go to <candidate>/<label>/, the
synthetic profile to ~/LabVizD2/<label>. After a failure the host is asked to exit; fix the cause
and run the stage again with a fresh label. Nothing here launches a native window or signs off
native behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path.home()/'LabVizD2'
EXPECTED = {'full': (15, 17), 'large': (7, 7)}


def fail(message):
    raise SystemExit(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('session', 'installer', 'large'))
    parser.add_argument('--candidate', type=Path, required=True, help='folder holding bundle/ and build/')
    parser.add_argument('--label', required=True, help='fresh name of the trial; installer reuses the session label')
    parser.add_argument('--node', type=Path, help="Node.js for the drivers (default: the candidate's own)")
    parser.add_argument('--sampler-python', type=Path, default=Path(sys.executable),
                        help='Python with psutil for the resource samplers (default: this interpreter)')
    args = parser.parse_args(argv)
    if not args.label.replace('-', '').isalnum() or args.label in ('bundle', 'build', 'provenance'):
        parser.error('the label may hold only letters, digits and hyphens, and may not name a candidate folder')
    args.candidate = args.candidate.resolve()
    args.bundle, args.build = args.candidate/'bundle', args.candidate/'build'
    args.node = (args.node or args.bundle/'app/runtime/node/node.exe').resolve()
    args.trial, args.logs = TRIALS/args.label, args.candidate/args.label
    return args


class Run:
    def __init__(self, args, record_name):
        self.args, self.path = args, args.logs/record_name
        self.env = dict(os.environ, LABVIZ_TEST_PYTHON=str(args.sampler_python))
        # The acceptance scripts check with assert, which PYTHONOPTIMIZE removes.
        self.env.pop('PYTHONOPTIMIZE', None)
        self.results = read(self.path)['results'] if self.path.exists() else {}

    def record(self, name, **value):
        self.results.setdefault(name, {}).update(value)
        self.path.write_text(json.dumps({'label': self.args.label, 'trial': str(self.args.trial),
                                         'results': self.results}, indent=2), encoding='utf-8')

    def step(self, name, command, timeout=1800):
        log, started = self.args.logs/(name+'.log'), time.time()
        with log.open('xb') as stream:
            try:
                code = subprocess.run([str(item) for item in command], cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                                      stdout=stream, stderr=stream, timeout=timeout,
                                      creationflags=subprocess.CREATE_NO_WINDOW).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        self.record(name, exitCode=code, seconds=round(time.time()-started, 1))
        print(name, code, flush=True)
        if code != 0:
            fail(f'{name} failed ({code}); see {log}')


def request_exit(trial):
    session_id = read(trial/'session.json')['id']
    temporary = trial/('commands.json.'+uuid.uuid4().hex)
    temporary.write_text(json.dumps({'acceptanceId': session_id, 'id': uuid.uuid4().hex, 'action': 'exit'}), encoding='utf-8')
    os.replace(temporary, trial/'commands.json')


def run_host(args, run, idle_seconds, work):
    """Start the headless host, do the work, and end the host normally even when the work failed."""
    log = args.logs/'host.log'
    with log.open('xb') as stream:
        host = subprocess.Popen([str(args.node), str(ROOT/'desktop/scripts/headless_session.mjs'), args.label,
                                 str(args.bundle), str(idle_seconds)],
                                cwd=ROOT, env=run.env, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.time()+300
            while '"ready":true' not in log.read_text(encoding='utf-8', errors='replace'):
                if time.time() > deadline or host.poll() is not None:
                    fail(f'the headless host did not become ready; see {log}')
                time.sleep(2)
            work()
        finally:
            if host.poll() is None:
                try:
                    request_exit(args.trial)
                    host.wait(timeout=300)
                except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                    print(f'the headless host (process {host.pid}) is still running: {error!r}', file=sys.stderr, flush=True)
    result = read(args.trial/'exit.json')
    run.record('normal-exit', hostExitCode=result['hostExitCode'], surviving=result['survivingOwnedProcesses'], runId=result['runId'])
    if result['hostExitCode'] != 0 or result['survivingOwnedProcesses']:
        fail(f'the host did not exit cleanly; see {args.trial/"exit.json"}')


def workflow(args, run, scenario, timeout):
    run.step('workflow', [args.node, ROOT/'desktop/scripts/cdp_workflow.mjs', args.trial, scenario], timeout)
    path = args.trial/f'workflow-{scenario}.json'
    report = read(path)
    counts = (len(report['checks']), len(report['exports']))
    run.record('workflow', passed=report['passed'], checks=counts[0], exports=counts[1])
    if report['passed'] is not True or counts != EXPECTED[scenario]:
        fail(f'workflow {scenario}: passed={report["passed"]}, checks and exports {counts}, expected {EXPECTED[scenario]}; see {path}')


def candidate_identity(args):
    """The package of the bundle, and the commit its freeze recorded for exactly that package."""
    package = read(args.bundle/'desktop-package.json')['packageId']
    frozen = read(args.build/'candidate.json')
    if frozen['packageId'] != package:
        fail(f'build/candidate.json is for package {frozen["packageId"]}, the bundle is {package}')
    return package, frozen['sourceCommit']


def evidence_to_write(args):
    """Where the evidence goes, or None when this package already has some, which is kept."""
    package = candidate_identity(args)[0]
    path = args.candidate/'background-evidence.json'
    if not path.exists():
        return path
    existing = read(path)['packageId']
    if existing != package:
        fail(f'{path} is for package {existing}, the bundle is {package}')
    return None


def write_evidence(args, run, path):
    package, commit = candidate_identity(args)
    measured = run.results['workflow']
    value = {'passed': True, 'inProgress': False, 'packageId': package,
             'scope': f'Bundled scientific workflow, part 2 headless acceptance and profile-copy acceptance (trial {args.label}); no native window',
             'scientificChecks': measured['checks'], 'frozenExports': measured['exports'], 'nativeAcceptance': False,
             'sourceCommit': commit}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
    print('evidence', path, flush=True)


def session(args):
    evidence = evidence_to_write(args)
    args.logs.mkdir(exist_ok=False)
    run = Run(args, 'pipeline.json')

    def work():
        workflow(args, run, 'full', 1800)
        run.step('part2-headless', [args.node, ROOT/'desktop/tests/part2_headless_acceptance.mjs', args.trial, args.bundle])
    run_host(args, run, 45, work)
    run.step('part2-profile', [sys.executable, '-B', '-X', 'utf8', ROOT/'desktop/tests/part2_profile_acceptance.py', args.trial, args.bundle])
    if evidence:
        write_evidence(args, run, evidence)
    else:
        print('the candidate keeps its existing background evidence', flush=True)
    return run


def installer(args):
    run = Run(args, 'pipeline.json')
    if run.results.get('part2-profile', {}).get('exitCode') != 0:
        fail(f'installer needs the label of a passed session; see {run.path}')
    if not (args.build/'build-verification.json').exists():
        fail(f'compile the Setup first: {args.build} has no build-verification.json')
    python = [sys.executable, '-B', '-X', 'utf8']
    run.step('installer', [*python, ROOT/'desktop/tests/d6_installer_acceptance.py', args.trial, args.build])
    run.step('retention', [*python, ROOT/'desktop/tests/d6_retention_after_upgrade.py', args.trial, args.bundle])
    run.step('installer-failure', [*python, ROOT/'desktop/tests/d6_installer_failure_acceptance.py', args.trial, args.build])
    return run


def large(args):
    args.logs.mkdir(exist_ok=False)
    run = Run(args, 'large.json')
    run_host(args, run, 90, lambda: workflow(args, run, 'large', 3600))
    return run


def main(argv=None):
    if sys.flags.optimize:
        fail('run without -O or PYTHONOPTIMIZE: the acceptance scripts check with assert')
    args = parse(argv)
    if not args.node.is_file():
        fail('Node.js not found: '+str(args.node))
    run = {'session': session, 'installer': installer, 'large': large}[args.stage](args)
    print(json.dumps(run.results))


if __name__ == '__main__':
    main()
