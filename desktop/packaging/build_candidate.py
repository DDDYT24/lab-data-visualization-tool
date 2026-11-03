"""Freeze a private desktop package; compile only against matching background evidence."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import email
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'desktop/runtime'))
sys.path.insert(0,str(ROOT/'desktop/packaging'))
from session import encoded, unlinked
from maintenance import DATA_EPOCH
import provenance
from licenses import python_inventory, rust_inventory, web_inventory


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def freeze(bundle, output, cargo):
    if (bundle/'desktop-package.json').exists():raise ValueError('Use a fresh staged bundle; preserve the frozen package and its evidence')
    validate_provenance(bundle)
    licenses=bundle/'licenses';licenses.mkdir(exist_ok=True)
    web=web_inventory(ROOT,bundle)
    python=python_inventory(bundle)
    rust=rust_inventory(ROOT,bundle,cargo)
    inventory={'python':python,'rust':rust,'webReviewIssues':web['reviewIssues'],'webLockSha256':sha(ROOT/'V2.0/web/package-lock.json'),'pythonLockSha256':sha(ROOT/'V2.0/api/requirements.lock.txt'),'cargoLockSha256':sha(ROOT/'desktop/host/Cargo.lock'),'runtimeVersions':json.loads((ROOT/'V2.0/packaging/windows-v23/runtimes.json').read_text(encoding='utf-8'))}
    (licenses/'DEPENDENCIES.json').write_bytes(encoded(inventory))
    shutil.copyfile(ROOT/'LICENSE',licenses/'LABVIZ-LICENSE.txt')
    for name in ('webview2.json','identity.json'):shutil.copyfile(ROOT/'desktop/packaging'/name,bundle/name)
    files=sorted(p for p in bundle.rglob('*') if p.is_file())
    with ThreadPoolExecutor(max_workers=8) as pool:hashes=list(pool.map(sha,files))
    manifest={'format':'labviz-desktop-package-v1','appId':'org.labviz.desktop.preview','dataEpoch':DATA_EPOCH,'checkpointSchema':1,
        'version':'2.3.0-desktop.1','sourceCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'sourceDirty':provenance.git_identity(ROOT)['dirty'],'tauriVersion':'2.12.4','webview2':json.loads((bundle/'webview2.json').read_text(encoding='utf-8')),
        'buildProvenance': {name:sha(bundle/'provenance'/name) for name in ['host.json','web.json','stage.json']},
        'webBuildId':json.loads((bundle/'app/candidate-build.json').read_text(encoding='utf-8'))['webBuildId'],
        'payloadCandidateId':json.loads((bundle/'desktop-runtime.json').read_text(encoding='utf-8'))['candidateId'],
        'installerSources':{p.relative_to(ROOT/'desktop/packaging').as_posix():sha(p) for p in (ROOT/'desktop/packaging').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.ps1','.iss','.json','.txt'}},
        'files':{p.relative_to(bundle).as_posix():digest for p,digest in zip(files,hashes)}}
    identifier=hashlib.sha256(encoded(manifest)).hexdigest();(bundle/'desktop-package.json').write_bytes(encoded({'manifest':manifest,'packageId':identifier}))
    support=output/'installer-support.zip'
    members=[]
    members += [p for p in (bundle/'app/runtime/python').iterdir() if p.is_file()]
    members += [bundle/'app/V2.0/api/labviz_api'/name for name in ['__init__.py','installed_profile.py','installed_lifecycle.py','source_data.py']]
    members += list((bundle/'runtime').glob('*.py'))
    with zipfile.ZipFile(support,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in sorted(members):archive.write(p,p.relative_to(bundle).as_posix())
    return {'packageId':identifier,'payloadCandidateId':manifest['payloadCandidateId'],'bundleBytes':sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()),'hostSha256':sha(bundle/'LabViz.exe'),'packageManifestSha256':sha(bundle/'desktop-package.json'),'supportZipSha256':sha(support),'nativeAcceptancePending':True,'publicReleaseAccepted':False,'licenseInventory':{'python':len(python),'rust':len(rust),'web':len(web['packages']),'webReviewIssues':len(web['reviewIssues'])},'sourceCommit':manifest['sourceCommit']}


def validate_host_evidence(bundle, receipt):
    proof=bundle/'provenance'
    out=Path(receipt['hostBuildScriptOutDir'])
    for name in ['acl-manifests.json','capabilities.json']:
        key=(out/name).relative_to(ROOT).as_posix()
        if sha(proof/('host-'+name))!=receipt['outputs']['files'].get(key):
            raise ValueError('Staged compiled host ACL differs: '+name)
    permission='desktop/host/permissions/acceptance-ready.toml'
    if sha(proof/'host-permissions.toml')!=receipt['sourceAfter']['files'].get(permission):
        raise ValueError('Staged host permission input differs')
    acl=json.loads((proof/'host-acl-manifests.json').read_text(encoding='utf-8'))['__app-acl__']
    expected={'allow-acceptance-editor-ready','deny-acceptance-editor-ready'}
    if set(acl['permissions'])!=expected or acl.get('default_permission') or acl.get('permission_sets'):
        raise ValueError('Compiled application permissions differ')
    allow=acl['permissions']['allow-acceptance-editor-ready']['commands']
    deny=acl['permissions']['deny-acceptance-editor-ready']['commands']
    capabilities=json.loads((proof/'host-capabilities.json').read_text(encoding='utf-8'))
    capability=capabilities.get('acceptance-ready',{})
    if (allow!={'allow':['acceptance_editor_ready'],'deny':[]}
            or deny!={'allow':[],'deny':['acceptance_editor_ready']}
            or set(capabilities)!={'acceptance-ready'}
            or capability.get('windows')!=['main'] or capability.get('local') is not False
            or capability.get('webviews',[]) or capability.get('identifier')!='acceptance-ready'
            or capability.get('remote')!={'urls':['http://127.0.0.1:*']}
            or capability.get('permissions')!=['allow-acceptance-editor-ready']):
        raise ValueError('Compiled host capability scope differs')


def validate_provenance(bundle):
    proof=bundle/'provenance'
    receipts=provenance.validate(ROOT,proof)
    validate_host_evidence(bundle,receipts['host'])
    stage=json.loads((proof/'stage.json').read_text(encoding='utf-8'))
    build=json.loads((bundle/'desktop-build.json').read_text(encoding='utf-8'))
    if (stage.get('format')!='labviz-stage-provenance-v1' or stage['apiBefore']!=stage['apiAfter']
            or stage['apiAfter']!=provenance.source(ROOT,'api') or build.get('provenanceSha256')!=sha(proof/'stage.json')
            or stage['hostReceiptSha256']!=sha(proof/'host.json') or stage['webReceiptSha256']!=sha(proof/'web.json')
            or stage['hostSha256']!=sha(bundle/'LabViz.exe') or stage['payloadCandidateId']!=build['candidateId']):
        raise ValueError('Staging provenance differs from the exact source/build')
    for name,want in stage['tools'].items():
        if sha(ROOT/name)!=want:raise ValueError('Staging tool changed: '+name)
    for name,want in stage['runtimeSha256'].items():
        if sha(bundle/'runtime'/name)!=want or sha(ROOT/'desktop/runtime'/name)!=want:
            raise ValueError('Runtime differs from staging provenance: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cargo-home',type=Path);parser.add_argument('--compiler',type=Path);parser.add_argument('--evidence',type=Path)
    args=parser.parse_args();bundle=unlinked(args.bundle.resolve());output=unlinked(args.output.resolve())
    if not bundle.is_relative_to(ROOT/'outputs/desktop-d06') or not output.is_relative_to(ROOT/'outputs/desktop-d06'):raise ValueError('Use this desktop worktree private D6 output')
    if args.compiler:
        validate_provenance(bundle)
        record=json.loads((output/'candidate.json').read_text(encoding='utf-8'));package=json.loads((bundle/'desktop-package.json').read_text(encoding='utf-8'))
        if record['packageId']!=package['packageId']:raise ValueError('Build identity differs')
        for name,want in package['manifest']['installerSources'].items():
            if sha(ROOT/'desktop/packaging'/name)!=want:raise ValueError('Installer source changed after freezing: '+name)
        if not args.evidence:raise ValueError('Matching background verification is required')
        evidence=json.loads(args.evidence.read_text(encoding='utf-8'))
        if not evidence.get('passed') or evidence.get('inProgress') or evidence['packageId']!=record['packageId']:raise ValueError('Background evidence is incomplete or for another package')
        members=list(package['manifest']['files'])
        with ThreadPoolExecutor(max_workers=8) as pool:hashes=list(pool.map(lambda name:sha(bundle/name),members))
        if dict(zip(members,hashes))!=package['manifest']['files']:raise ValueError('Frozen bundle changed')
        if {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()} != set(package['manifest']['files']) | {'desktop-package.json'}:
            raise ValueError('Unexpected files appeared in the frozen desktop bundle')
        if sha(output/'installer-support.zip')!=record['supportZipSha256']:raise ValueError('Installer support bytes changed')
        log=output/'compiler.log'
        with log.open('xb') as stream:
            result=subprocess.run([str(args.compiler.resolve()),'/DBundleRoot='+str(bundle),'/DPackageId='+record['packageId'],'/DPackageDir='+record['packageId'][:24],'/DSupportZip='+str(output/'installer-support.zip'),'/DOutputDir='+str(output),str(ROOT/'desktop/packaging/LabVizDesktop.iss')],stdout=stream,stderr=stream,check=False)
        executables=list(output.glob('*-Setup.exe'));record.update(compilerExitCode=result.returncode,backgroundEvidenceSha256=sha(args.evidence),installers={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in executables})
        (output/'build-verification.json').write_text(json.dumps(record,indent=2))
        if result.returncode or len(executables)!=1:raise ValueError('Compilation failed; retain and inspect compiler.log')
    else:
        if not args.cargo_home:raise ValueError('Provide the exact Cargo cache used to build the host')
        output.mkdir(parents=True,exist_ok=False);record=freeze(bundle,output,args.cargo_home.resolve());(output/'candidate.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
