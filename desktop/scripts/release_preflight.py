"""Verify a private artifact and report release blockers without publishing or signing."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'desktop/runtime'))
from session import unlinked,io_path

GATES={
    'window-tray-exit':'native-windows','native-files':'native-windows',
    'draft-reconstruction':'native-windows','native-suspension':'native-windows',
    'native-resource-budgets':'native-windows','mandatory-measurements':'native-windows',
    'busy-tray-interruption':'native-windows','data-tools-pdf-clipboard-links':'native-windows',
    'windows-session-power':'native-windows','clean-windows-components':'clean-windows',
    'physical-offline':'physical-offline','current-version-native-coexistence':'native-windows',
    'accessibility-researcher-trial':'native-windows','signed-distribution':'release-review',
}

def read(path):return json.loads(io_path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with io_path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def headless_evidence(path):
    if path.suffix.lower() not in {'.json','.jsonl'}:return False
    if path.suffix.lower()=='.jsonl':
        with io_path(path).open(encoding='utf-8-sig') as stream:value=json.loads(stream.readline(65536))
    else:value=read(path)
    return isinstance(value,dict) and ('headless' in str(value.get('scope','')).lower() or value.get('nativeAppLaunched') is False or value.get('nativeAcceptance') is False)
def member(root,name):
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:raise ValueError('Invalid evidence/member path')
    result=unlinked(root.joinpath(*relative.parts))
    if not result.is_relative_to(root):raise ValueError('Evidence/member escaped its root')
    return result

def verify_artifact(bundle,build):
    package=read(bundle/'desktop-package.json');manifest=package['manifest']
    if manifest.get('format')!='labviz-desktop-package-v1' or manifest.get('appId')!='org.labviz.desktop.preview' or manifest.get('dataEpoch')!=1 or manifest.get('checkpointSchema')!=1:raise ValueError('Unsupported desktop package identity or schema')
    if hashlib.sha256(canonical(manifest)).hexdigest()!=package['packageId']:raise ValueError('Package identity differs')
    record=read(build/'build-verification.json')
    if record['packageId']!=package['packageId'] or record['compilerExitCode']!=0 or len(record['installers'])!=1:raise ValueError('Installer build differs')
    name,installer=next(iter(record['installers'].items()));setup=member(build,name)
    if sha(setup)!=installer['sha256'] or setup.stat().st_size!=installer['bytes']:raise ValueError('Setup bytes differ')
    def check(item):
        name,want=item
        if sha(member(bundle,name))!=want:raise ValueError('Bundle member differs: '+name)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,manifest['files'].items()))
    actual=set()
    for directory,folders,files in os.walk(bundle,followlinks=False):
        for name in folders+files:unlinked(Path(directory)/name)
        actual.update((Path(directory)/name).relative_to(bundle).as_posix() for name in files)
    if actual!=set(manifest['files'])|{'desktop-package.json'}:raise ValueError('Unexpected bundle members')
    return package,setup,installer

def signature(setup):
    command="[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false); $ErrorActionPreference='Stop'; $m=Import-Module (Join-Path $PSHOME 'Modules\\Microsoft.PowerShell.Security\\Microsoft.PowerShell.Security.psd1') -Force -PassThru; $s=& $m.ExportedCmdlets['Get-AuthenticodeSignature'] -LiteralPath $env:LABVIZ_PREFLIGHT_SIGN_TARGET; [ordered]@{status=$s.Status.ToString();thumbprint=$s.SignerCertificate.Thumbprint;timestamped=($null -ne $s.TimeStamperCertificate)} | ConvertTo-Json -Compress"
    shell=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    result=subprocess.run([str(shell),'-NoProfile','-NonInteractive','-Command',command],env={**os.environ,'PSModulePath':str(shell.parent/'Modules'),'LABVIZ_PREFLIGHT_SIGN_TARGET':str(setup)},capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW,timeout=60)
    if result.returncode:raise ValueError('Signature inspection failed')
    return json.loads(result.stdout.decode('utf-8-sig'))

def signature_blockers(value,expected):
    blockers=[]
    if value.get('status')!='Valid':blockers.append('SETUP_UNSIGNED_OR_UNTRUSTED')
    if not expected:blockers.append('SIGNER_IDENTITY_UNSPECIFIED')
    elif str(value.get('thumbprint') or '').upper()!=expected.upper():blockers.append('SIGNER_IDENTITY_DIFFERS')
    if value.get('status')=='Valid' and not value.get('timestamped'):blockers.append('SIGNATURE_TIMESTAMP_MISSING')
    return blockers

def evaluate_record(record,package_id,setup_sha,evidence_root,policy):
    blockers=[]
    def block(code,detail):blockers.append({'code':code,'detail':detail})
    if not isinstance(record,dict):return [{'code':'ACCEPTANCE_RECORD_MALFORMED','detail':'Expected an object'}]
    if record.get('format')!='labviz-desktop-release-record-v1':block('ACCEPTANCE_FORMAT_DIFFERS','Use the release-record-v1 template')
    if record.get('packageId')!=package_id or record.get('setupSha256')!=setup_sha:block('ACCEPTANCE_IDENTITY_DIFFERS','Bind the exact package and Setup hashes')
    gates=record.get('gates',[])
    if not isinstance(gates,list) or not all(isinstance(gate,dict) and isinstance(gate.get('id'),str) for gate in gates):
        return [{'code':'GATE_SET_DIFFERS','detail':'Expected a list of named gate objects'}]
    ids=[gate.get('id') for gate in gates]
    if len(ids)!=len(set(ids)) or set(ids)!=set(GATES):block('GATE_SET_DIFFERS','Every required gate must appear exactly once')
    machine=record.get('machine',{})
    if not isinstance(machine,dict):machine={}
    def text(value):return isinstance(value,str) and bool(value.strip())
    operator=machine.get('operator')
    for gate in gates:
        identifier=gate.get('id')
        if identifier not in GATES:continue
        if gate.get('outcome')!='pass':block('GATE_NOT_PASSED',identifier);continue
        if gate.get('scope')!=GATES[identifier]:block('OBSERVATION_SCOPE_DIFFERS',identifier)
        if not text(operator) and not text(gate.get('operator')):block('OPERATOR_MISSING',identifier)
        if not text(gate.get('machineId')) and not text(machine.get('id')):block('MACHINE_ID_MISSING',identifier)
        try:
            if datetime.fromisoformat(gate.get('utcTime','').replace('Z','+00:00')).tzinfo is None:raise ValueError()
        except (AttributeError,TypeError,ValueError):block('OBSERVATION_TIME_MISSING',identifier)
        if not isinstance(gate.get('actions'),list) or not gate['actions'] or not all(text(value) for value in gate['actions']) or gate.get('expected') in (None,'') or gate.get('actual') in (None,''):block('OBSERVATION_DETAIL_MISSING',identifier)
        evidence=gate.get('evidenceSha256',{})
        if not isinstance(evidence,dict) or not evidence:block('EVIDENCE_MISSING',identifier);continue
        non_headless=False
        for name,want in evidence.items():
            try:
                path=member(evidence_root,name)
                if not re.fullmatch('[a-fA-F0-9]{64}',str(want)) or sha(path).lower()!=want.lower():raise ValueError()
                non_headless|=not headless_evidence(path)
            except (OSError,ValueError,TypeError):block('EVIDENCE_INVALID',identifier+':'+str(name))
        if GATES[identifier]!='release-review' and not non_headless:block('HEADLESS_EVIDENCE_CANNOT_ACCEPT_NATIVE',identifier)
    metrics=record.get('metrics',{})
    if not isinstance(metrics,dict):metrics={}
    def numeric(value):return type(value) in (int,float) and math.isfinite(value) and value>0
    limits=policy['numericGates']
    for key,limit in [('nativeHostPrivateCommitMiB',limits['nativeHostPrivateCommitMiB']),('completeSuspendedPrivateCommitMiB',limits['completeSuspendedPrivateCommitMiB'])]:
        value=metrics.get(key)
        if not numeric(value):block('NATIVE_METRIC_MISSING',key)
        elif value>limit:block('NATIVE_BUDGET_EXCEEDED',key)
    samples=metrics.get('smallEditableRestoreSamplesMs')
    if not isinstance(samples,list) or len(samples)<10 or not all(numeric(value) for value in samples):block('RESTORE_SAMPLES_MISSING','At least ten actual native small-workspace restores')
    elif max(samples)>limits['smallWorkspaceEditableRestoreMs']:block('NATIVE_BUDGET_EXCEEDED','smallWorkspaceEditableRestoreMs')
    reports=record.get('reports',{})
    if not isinstance(reports,dict):reports={}
    for required in policy['requiredReports']:
        metric=required['metric'];item=reports.get(metric,{})
        try:
            path=member(evidence_root,item['path'])
            if sha(path)!=item['sha256']:raise ValueError()
            if metric!='installation' and headless_evidence(path):block('MANDATORY_REPORT_IS_HEADLESS',metric)
        except (KeyError,TypeError,OSError,ValueError):block('MANDATORY_REPORT_MISSING',metric)
    return blockers

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','build','acceptance-record','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-signer-thumbprint')
    args=parser.parse_args();bundle=unlinked(args.bundle.absolute());build=unlinked(args.build.absolute());output=unlinked(args.output.absolute())
    if output.exists() or output.is_relative_to(bundle) or output.is_relative_to(build):raise ValueError('Use a fresh report outside the bundle and build')
    result={'artifactIntegrityPassed':False,'releaseAccepted':False,'nativeAppLaunched':False,'scope':'Read-only release preflight; does not sign, publish, operate windows or replace shortcuts','blockers':[]}
    try:
        package,setup,installer=verify_artifact(bundle,build);result.update(artifactIntegrityPassed=True,packageId=package['packageId'],setupSha256=installer['sha256'])
        sig=signature(setup);result['signature']=sig
        result['blockers'] += [{'code':code,'detail':'Intended publisher signature'} for code in signature_blockers(sig,args.expected_signer_thumbprint)]
        dependencies=read(bundle/'licenses/DEPENDENCIES.json');issues=dependencies['webReviewIssues']
        if issues:result['blockers'].append({'code':'UPSTREAM_NOTICE_REVIEW_OPEN','detail':[item['name']+'@'+item['version'] for item in issues]})
        record_path=unlinked(args.acceptance_record.absolute());record=read(record_path)
        result['blockers']+=evaluate_record(record,package['packageId'],installer['sha256'],record_path.parent,read(ROOT/'desktop/contracts/resource-budget-v1.json'))
        result['releaseAccepted']=not result['blockers']
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError,subprocess.SubprocessError) as error:result['blockers'].append({'code':'PREFLIGHT_INPUT_OR_INTEGRITY_FAILURE','detail':type(error).__name__})
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result));return 0 if result['releaseAccepted'] else 2 if result['artifactIntegrityPassed'] else 1

if __name__=='__main__':raise SystemExit(main())
