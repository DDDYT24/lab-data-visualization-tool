"""Stage changed desktop sources with the verified D1 scientific runtime."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import ctypes
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/desktop-d34/bundle'
EXTRA_RUNTIME = ()
BUILD_SCOPE = 'Private D3/D4 desktop changes on the frozen RC baseline'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    global OUT
    if not EXTRA_RUNTIME and (ROOT/'desktop/runtime/installation.py').exists():
        raise ValueError('The D3-D5 bundle is frozen. Use stage_d6.py for current sources.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--api-only', action='store_true')
    parser.add_argument('--output', type=Path, help='Fresh private directory below this checkout outputs/')
    parser.add_argument('--provenance', type=Path, required=True, help='Successful host/web build receipts and logs')
    args = parser.parse_args()
    if args.output:
        OUT = args.output.resolve()
    spec = importlib.util.spec_from_file_location('stage', ROOT/'V2.0/packaging/windows-v23/stage_candidate.py')
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    stage.checked(OUT)
    assert OUT.resolve().is_relative_to((ROOT/'outputs').resolve())
    if OUT.resolve() == (ROOT/'outputs').resolve():
        raise ValueError('Choose a dedicated preview directory, not outputs itself.')
    if (OUT/'desktop-package.json').exists():
        raise ValueError('This candidate is frozen. Choose a new --output directory; never restage it.')
    import sys
    sys.path.insert(0, str(ROOT/'desktop/packaging'))
    import provenance
    provenance.validate(ROOT, args.provenance.resolve())
    api_before = provenance.source(ROOT, 'api')
    if os.name=='nt':
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype=wintypes.HANDLE;kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        for name in ('LabViz.exe','app/runtime/python/python.exe','app/runtime/node/node.exe'):
            executable=OUT/name
            if executable.exists():
                handle=kernel.CreateFileW(str(executable),0x40000000,7,None,3,0,None)
                if handle==ctypes.c_void_p(-1).value:raise ValueError('The preview runtime is in use or not writable. Stop every owned preview process before staging: '+name)
                kernel.CloseHandle(handle)
    payload = OUT / 'app'
    if args.prepare:
        OUT.mkdir(parents=True, exist_ok=False)
        old = ROOT/'outputs/desktop-d01/payload'
        manifest = json.loads((old/'payload-manifest.json').read_text())
        runtime = {name: digest for name, digest in manifest['files'].items() if name.startswith('runtime/')}
        def verify(item):
            name, digest = item
            assert sha(old/name) == digest, name
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(verify, runtime.items()))
        stage.copy_tree(old/'runtime', payload/'runtime')
    assert (payload/'runtime/python/python.exe').is_file()
    # All mutable staging is inside this explicitly owned preview folder.
    # Clear only the staged website, which must not be used by a running preview.
    web = payload/'V2.0'
    dist = ROOT/'V2.0/web/.next-v23-package'
    if not args.api_only:
        for sub in ('web', 'node_modules'):
            target = web/sub
            if target.exists():
                assert target.resolve().is_relative_to(OUT.resolve()) and not target.is_symlink() and not target.is_junction()
                shutil.rmtree(target)
        stage.copy_tree(dist/'standalone', web)
        stage.copy_tree(dist/'static', web/'web/.next-v23-package/static')
        stage.copy_tree(ROOT/'V2.0/web/public', web/'web/public')
        stage.portable_web_config(web/'web')
    for entry in ('labviz_api', 'migrations', 'samples/v22'):
        stage.copy_tree(ROOT/'V2.0/api'/entry, web/'api'/entry)
    for name in ('requirements.txt', 'requirements.lock.txt', 'pyproject.toml', 'alembic.ini'):
        shutil.copyfile(ROOT/'V2.0/api'/name, web/'api'/name)
    for entry in ('contracts', 'assets'):
        stage.copy_tree(ROOT/'V2.0'/entry, web/entry)
    stage.copy_tree(ROOT/'V2.0/packaging/windows-v23/bin', payload/'bin')
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
        shutil.copyfile(ROOT/name, payload/name)
    cache_script = '''
import compileall, py_compile, pathlib
root=pathlib.Path.cwd()
sites=root/'runtime/python/Lib/site-packages'
packages=['fastapi','starlette','pydantic','pydantic_core','anyio','boto3','botocore','uvicorn','h11','sniffio','annotated_types','typing_inspection','multipart','python_multipart','annotated_doc','dateutil']
paths=[root/'V2.0/api/labviz_api']+[sites/name for name in packages]+[sites/'typing_extensions.py']
for path in paths:
    if not path.exists(): continue
    if path.is_dir():
        if not compileall.compile_dir(str(path),quiet=2,force=True,stripdir=str(root),prependdir='LabViz',invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH): raise RuntimeError('Bytecode compilation failed')
    else:
        py_compile.compile(str(path),dfile='LabViz/'+path.relative_to(root).as_posix(),doraise=True,invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
'''
    subprocess.run([str(payload/'runtime/python/python.exe'),'-B','-X','utf8','-c',cache_script],cwd=payload,env={**os.environ,'PYTHONHASHSEED':'0'},check=True)
    names = sorted(p for p in payload.rglob('*') if p.is_file() and p.name not in ('candidate-build.json', 'payload-manifest.json'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(sha, names))
    files = {p.relative_to(payload).as_posix(): digest for p, digest in zip(names, hashes)}
    candidate = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    # Declare the epoch of the API this payload carries, which its runtime checks against.
    source = (payload/'V2.0/api/labviz_api/installed_profile.py').read_text(encoding='utf-8')
    epoch = int(re.search(r'^DATA_EPOCH = (\d+)$', source, re.M).group(1))
    metadata = {'format': 'labviz-desktop-payload-v1', 'appId': stage.APP_ID, 'candidateId': candidate, 'dataEpoch': epoch,
                'sourceCommit': provenance.git_identity(ROOT)['head'], 'scope': BUILD_SCOPE, 'webBuildId': (dist/'BUILD_ID').read_text().strip()}
    (payload/'candidate-build.json').write_text(json.dumps(metadata, indent=2))
    (payload/'payload-manifest.json').write_text(json.dumps({'files': files, 'candidateId': candidate}, indent=2))
    host = ROOT/'desktop/prototypes/tauri/target/release/LabViz.exe'
    shutil.copyfile(host, OUT/'LabViz.exe')
    (OUT/'runtime').mkdir(exist_ok=True)
    runtime = {}
    for name in ('controller.py', 'preferences.py', 'process_job.py', 'session.py', 'service_runner.py', *EXTRA_RUNTIME):
        shutil.copyfile(ROOT/'desktop/runtime'/name, OUT/'runtime'/name)
        runtime[name] = sha(OUT/'runtime'/name)
    binding = {'format': 'labviz-desktop-runtime-v1', 'payload': 'app', 'controller': 'runtime/controller.py', 'candidateId': candidate,
               'controllerSha256': runtime['controller.py'], 'preferencesSha256': runtime['preferences.py'], 'processJobSha256': runtime['process_job.py'], 'sessionSha256': runtime['session.py'], 'serviceRunnerSha256':runtime['service_runner.py']}
    if EXTRA_RUNTIME:
        binding.update(maintenanceSha256=runtime['maintenance.py'], installationSha256=runtime['installation.py'])
    (OUT/'desktop-runtime.json').write_text(json.dumps(binding, indent=2))
    receipt = {'format': 'labviz-desktop-build-v1', 'candidateId': candidate, 'hostSha256': sha(OUT/'LabViz.exe'), 'runtimeSha256': runtime, 'payloadFiles': len(files), 'automaticSuspension': True}
    api_after = provenance.source(ROOT, 'api')
    if api_before != api_after:
        raise ValueError('API or copied contract/assets changed during staging; candidate not frozen')
    provenance.validate(ROOT, args.provenance.resolve())
    proof = OUT/'provenance'; proof.mkdir(exist_ok=True)
    for name in ('host.json', 'host.log', 'web.json', 'web.log'):
        shutil.copyfile(args.provenance.resolve()/name, proof/name)
    record = {'format': 'labviz-stage-provenance-v1', 'apiBefore': api_before, 'apiAfter': api_after,
              'hostReceiptSha256': sha(proof/'host.json'), 'webReceiptSha256': sha(proof/'web.json'),
              'runtimeSha256': runtime, 'hostSha256': receipt['hostSha256'], 'payloadCandidateId': candidate,
              'tools': {name: sha(ROOT/name) for name in ['desktop/scripts/stage_d34.py', 'desktop/scripts/stage_d6.py', 'V2.0/packaging/windows-v23/stage_candidate.py']}}
    (proof/'stage.json').write_bytes(provenance.canonical(record))
    receipt['provenanceSha256'] = sha(proof/'stage.json')
    (OUT/'desktop-build.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
