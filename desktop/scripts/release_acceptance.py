"""Release acceptance of one frozen desktop candidate, without a native window.

Usage, from the repository root:
    python -B -X utf8 desktop/scripts/release_acceptance.py <stage> --candidate <folder> --label <label>

Stages:
    session    headless host; the full scientific workflow (15 checks, 17 exports); the part 2
               headless and profile checks; a normal exit with no surviving process
               (it then writes <candidate>/background-evidence.json, which compiling requires)
    installer  silent install, repair, update and rollback; retention after upgrade; recovery
               from a failed update
    all        session, then installer
    large      the 100,000-row workflow and a normal exit
    host, workflow, stop
               the session stage in separate steps, to retry after one step failed; stop asks the
               host to exit normally (pass --failed when the workflow did not pass)

The candidate folder holds bundle/ and, once compiled, build/. Logs and results go to
<candidate>/<label>/; the synthetic profile goes to <trial root>/<label>. Nothing here launches a
native window or signs off native behaviour.
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
WORKFLOW_CHECKS, WORKFLOW_EXPORTS = 15, 17


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('session', 'installer', 'all', 'large', 'host', 'workflow', 'stop'))
    parser.add_argument('--candidate', type=Path, required=True, help='folder holding bundle/ and build/')
    parser.add_argument('--label', required=True, help='fresh name of the trial and of its log folder')
    parser.add_argument('--trial-root', type=Path, default=Path.home()/'LabVizD2')
    parser.add_argument('--node', type=Path, help="Node.js for the drivers (default: the candidate's own)")
    parser.add_argument('--sampler-python', type=Path, default=Path(sys.executable),
                        help='Python with psutil for the resource samplers (default: this interpreter)')
    parser.add_argument('--failed', action='store_true', help='stop: the workflow is known to have failed')
    args = parser.parse_args(argv)
    if not args.label.replace('-', '').isalnum():
        parser.error('the label may hold only letters, digits and hyphens')
    args.candidate = args.candidate.resolve()
    args.bundle, args.build = args.candidate/'bundle', args.candidate/'build'
    args.node = (args.node or args.bundle/'app/runtime/node/node.exe').resolve()
    args.trial = args.trial_root.resolve()/args.label
    args.logs = args.candidate/args.label
    return args


class Run:
    def __init__(self, args, record_name):
        self.args, self.record_name = args, record_name
        self.env = dict(os.environ, LABVIZ_TEST_PYTHON=str(args.sampler_python))
        path = args.logs/record_name
        self.results = json.loads(path.read_text(encoding='utf-8'))['results'] if path.exists() else {}

    def record(self):
        value = {'label': self.args.label, 'trial': str(self.args.trial), 'results': self.results}
        (self.args.logs/self.record_name).write_text(json.dumps(value, indent=2), encoding='utf-8')

    def step(self, name, command, timeout=1800):
        started = time.time()
        with (self.args.logs/(name+'.log')).open('wb') as stream:
            code = subprocess.run([str(item) for item in command], cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=stream, timeout=timeout,
                                  creationflags=subprocess.CREATE_NO_WINDOW).returncode
        self.results[name] = {'exitCode': code, 'seconds': round(time.time()-started, 1)}
        self.record()
        print(name, code, flush=True)
        if code:
            raise SystemExit(f'{name} failed; see {self.args.logs/(name+".log")}')

    def start_host(self, idle_seconds):
        stream = (self.args.logs/'host.log').open('wb')
        host = subprocess.Popen([str(self.args.node), str(ROOT/'desktop/scripts/headless_session.mjs'), self.args.label,
                                 str(self.args.bundle), str(idle_seconds)],
                                cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.time()+300
        while '"ready":true' not in (self.args.logs/'host.log').read_text(encoding='utf-8', errors='replace'):
            if time.time() > deadline or host.poll() is not None:
                raise SystemExit('the headless host did not become ready; see '+str(self.args.logs/'host.log'))
            time.sleep(2)
        return host, stream


def request_exit(trial):
    session = json.loads((trial/'session.json').read_text(encoding='utf-8'))
    temporary = trial/('commands.json.'+uuid.uuid4().hex)
    temporary.write_text(json.dumps({'acceptanceId': session['id'], 'id': uuid.uuid4().hex, 'action': 'exit'}),
                         encoding='utf-8')
    os.replace(temporary, trial/'commands.json')
    return session


def read_exit(trial):
    return json.loads((trial/'exit.json').read_text(encoding='utf-8-sig'))


def workflow_passed(report):
    return report['passed'] and len(report['checks']) == WORKFLOW_CHECKS and len(report['exports']) == WORKFLOW_EXPORTS


def session(args, run):
    host, stream = run.start_host(45)
    run.step('workflow', [args.node, ROOT/'desktop/scripts/cdp_workflow.mjs', args.trial, 'full'])
    assert workflow_passed(json.loads((args.trial/'workflow-full.json').read_text(encoding='utf-8')))
    run.step('part2-headless', [args.node, ROOT/'desktop/tests/part2_headless_acceptance.mjs', args.trial, args.bundle])
    request_exit(args.trial)
    host.wait(timeout=300)
    stream.close()
    exit_record = read_exit(args.trial)
    assert exit_record['hostExitCode'] == 0 and not exit_record['survivingOwnedProcesses']
    run.results['normal-exit'] = {'hostExitCode': exit_record['hostExitCode'], 'runId': exit_record['runId']}
    run.record()
    run.step('part2-profile', [sys.executable, '-B', '-X', 'utf8', ROOT/'desktop/tests/part2_profile_acceptance.py',
                               args.trial, args.bundle])
    write_evidence(args)


def write_evidence(args):
    """The background evidence that desktop/packaging/build_candidate.py requires before compiling."""
    path = args.candidate/'background-evidence.json'
    if path.exists():
        raise SystemExit('keep the existing evidence '+str(path))
    package = json.loads((args.bundle/'desktop-package.json').read_text(encoding='utf-8'))['packageId']
    frozen = args.build/'candidate.json'
    commit = json.loads(frozen.read_text(encoding='utf-8')).get('sourceCommit') if frozen.exists() else None
    value = {'passed': True, 'inProgress': False, 'packageId': package,
             'scope': f'Bundled scientific workflow, part 2 headless acceptance and profile-copy acceptance (trial {args.label}); no native window',
             'scientificChecks': WORKFLOW_CHECKS, 'frozenExports': WORKFLOW_EXPORTS, 'nativeAcceptance': False,
             'sourceCommit': commit}
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print('evidence', path, flush=True)


def installer(args, run):
    python = [sys.executable, '-B', '-X', 'utf8']
    run.step('installer', [*python, ROOT/'desktop/tests/d6_installer_acceptance.py', args.trial, args.build])
    run.step('retention', [*python, ROOT/'desktop/tests/d6_retention_after_upgrade.py', args.trial, args.bundle])
    run.step('installer-failure', [*python, ROOT/'desktop/tests/d6_installer_failure_acceptance.py', args.trial, args.build])


def large(args, run):
    host, stream = run.start_host(90)
    started = time.time()
    with (args.logs/'workflow.log').open('wb') as log:
        code = subprocess.run([str(args.node), str(ROOT/'desktop/scripts/cdp_workflow.mjs'), str(args.trial), 'large'],
                              cwd=ROOT, env=run.env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, timeout=3600,
                              creationflags=subprocess.CREATE_NO_WINDOW).returncode
    run.results['workflow-large'] = {'exitCode': code, 'seconds': round(time.time()-started, 1)}
    run.record()
    report = json.loads((args.trial/'workflow-large.json').read_text(encoding='utf-8'))
    run.results['workflow-large'].update(passed=report['passed'], checks=len(report['checks']), exports=len(report['exports']))
    run.record()
    request_exit(args.trial)
    host.wait(timeout=300)
    stream.close()
    exit_record = read_exit(args.trial)
    run.results['normal-exit'] = {'hostExitCode': exit_record['hostExitCode'], 'surviving': exit_record['survivingOwnedProcesses']}
    run.record()
    return code == 0 and report['passed'] and exit_record['hostExitCode'] == 0 and not exit_record['survivingOwnedProcesses']


def receipt(args, mode, command, logfile, env):
    """One step of a retried session, with a receipt that is never overwritten."""
    record = {'command': [str(item) for item in command], 'cwd': str(ROOT), 'samplerPython': env['LABVIZ_TEST_PYTHON'],
              'startedUtcSeconds': time.time(), 'nativeWindowLaunched': False, 'log': str(logfile)}
    path = args.logs/('step-'+mode+'.json')
    if path.exists():
        raise SystemExit('keep the existing receipt '+str(path)+'; use a fresh label to retry')
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    with logfile.open('xb') as stream:
        code = subprocess.run(record['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                              creationflags=subprocess.CREATE_NO_WINDOW).returncode
    record.update(finishedUtcSeconds=time.time(), exitCode=code)
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record))
    return code


def stop(args):
    session_record = json.loads((args.trial/'session.json').read_text(encoding='utf-8'))
    report = json.loads((args.trial/'workflow-full.json').read_text(encoding='utf-8'))
    assert session_record['shell'] == 'headless-test-adapter'
    assert not report['passed'] if args.failed else workflow_passed(report)
    request_exit(args.trial)
    value = {'requestedUtcSeconds': time.time(), 'runId': session_record['id'], 'action': 'exit',
             'forcedTermination': False, 'workflowPassed': report['passed']}
    (args.logs/'step-stop.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({'normalExitRequested': True, 'runId': session_record['id'], 'workflowPassed': report['passed']}))


def main(argv=None):
    args = parse(argv)
    if not args.node.is_file():
        raise SystemExit('Node.js not found: '+str(args.node))
    if args.stage == 'stop':
        return stop(args)
    if args.stage in ('host', 'workflow'):
        args.logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, LABVIZ_TEST_PYTHON=str(args.sampler_python))
        if args.stage == 'host':
            command = [args.node, ROOT/'desktop/scripts/headless_session.mjs', args.label, args.bundle, '45']
        else:
            command = [args.node, ROOT/'desktop/scripts/cdp_workflow.mjs', args.trial, 'full']
        raise SystemExit(receipt(args, args.stage, command, args.logs/(args.stage+'.log'), env))
    if args.stage == 'large':
        args.logs.mkdir(parents=True, exist_ok=False)
        run = Run(args, 'large.json')
        passed = large(args, run)
        print(json.dumps(run.results))
        raise SystemExit(0 if passed else 1)
    # A later stage may reuse the label of an earlier one: "session", then "installer".
    args.logs.mkdir(parents=True, exist_ok=args.stage != 'all')
    run = Run(args, 'pipeline.json')
    if args.stage in ('all', 'session'):
        session(args, run)
    if args.stage in ('all', 'installer'):
        installer(args, run)
    print(json.dumps(run.results))


if __name__ == '__main__':
    main()
