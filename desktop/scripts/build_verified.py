"""Build fresh desktop components and seal their source/output receipts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'desktop/packaging'))
import provenance


def build_directory(root, requested):
    requested = requested.absolute()
    for member in (requested, *requested.parents):
        if member.is_symlink() or member.is_junction():
            raise ValueError('Linked build output paths are not supported')
    directory = requested.resolve()
    boundary = (root/'outputs').resolve()
    if directory == boundary or not directory.is_relative_to(boundary):
        raise ValueError('Use a dedicated private output below this checkout outputs/')
    for member in (directory, *directory.parents):
        if member == boundary:
            break
        if any((member/name).exists() for name in ('desktop-package.json', 'desktop-runtime.json', 'candidate.json')):
            raise ValueError('Preserve existing candidates; choose a separate build output')
    allowed = {'host.json', 'host.log', 'web.json', 'web.log'}
    if directory.exists() and (not directory.is_dir() or any(p.name not in allowed or not p.is_file() for p in directory.iterdir())):
        raise ValueError('Build output contains unrelated files; choose a fresh output')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component', choices=['host', 'web'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cargo-home', type=Path)
    parser.add_argument('--node', type=Path)
    args = parser.parse_args()
    tools_before = {'builderSha256': provenance.sha(Path(__file__)),
                    'provenanceToolSha256': provenance.sha(ROOT/'desktop/packaging/provenance.py')}
    directory = build_directory(ROOT, args.output)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/(args.component+'.json')
    if receipt.exists():
        raise ValueError('Preserve the existing build receipt; choose a new output')
    before = provenance.source(ROOT, args.component)
    environment = dict(os.environ)
    if args.component == 'host':
        if args.cargo_home is None:
            raise ValueError('Provide the private Cargo cache')
        cargo = args.cargo_home.resolve()
        environment.update(CARGO_HOME=str(cargo), RUSTUP_HOME=str(cargo.parent/'rustup'),
                           CARGO_TARGET_DIR=str(ROOT/'desktop/prototypes/tauri/target'))
        # An extra final-crate rustc cfg forces a fresh compile, while preserving
        # the exact locked dependency cache. No cleanup or source touch is needed.
        command = [str(cargo/'bin/cargo.exe'), 'rustc', '--release', '--locked', '--offline',
                   '--message-format=json-render-diagnostics',
                   '--manifest-path', str(ROOT/'desktop/host/Cargo.toml'), '--', '--cfg', 'labviz_build_'+uuid.uuid4().hex]
        cwd = ROOT
    else:
        if args.node is None:
            raise ValueError('Provide the exact bundled Node executable')
        environment.update(LABVIZ_PACKAGE_BUILD='1', NEXT_TELEMETRY_DISABLED='1')
        command = [str(args.node.resolve()), str(ROOT/'V2.0/web/node_modules/next/dist/bin/next'), 'build']
        cwd = ROOT/'V2.0/web'
    next_env = ROOT/'V2.0/web/next-env.d.ts'
    original = next_env.read_bytes() if args.component == 'web' else None
    log = directory/(args.component+'.log')
    try:
        with log.open('xb') as stream:
            result = subprocess.run(command, cwd=cwd, env=environment, stdout=stream, stderr=stream, check=False)
        after = provenance.source(ROOT, args.component)
        value = {'format': 'labviz-component-build-v1', 'component': args.component,
                 'sourceBefore': before, 'sourceAfter': after, 'exitCode': result.returncode,
                 'command': command, 'git': provenance.git_identity(ROOT), 'logSha256': provenance.sha(log),
                 **tools_before}
        value['buildToolsUnchanged'] = tools_before == {'builderSha256': provenance.sha(Path(__file__)), 'provenanceToolSha256': provenance.sha(ROOT/'desktop/packaging/provenance.py')}
        value['tool'] = {'executableSha256': provenance.sha(Path(command[0])),
                         'version': subprocess.check_output([command[0], '--version'], env=environment, text=True).strip()}
        if args.component == 'host':
            value['tool']['rustc'] = subprocess.check_output([str(cargo/'bin/rustc.exe'), '-vV'], env=environment, text=True).strip()
        if result.returncode == 0 and before == after and value['buildToolsUnchanged']:
            host_out_dir = provenance.host_build_output(ROOT, log) if args.component == 'host' else None
            if host_out_dir is not None:
                value['hostBuildScriptOutDir'] = str(host_out_dir)
            value['outputs'] = provenance.outputs(ROOT, args.component, host_out_dir)
        receipt.write_bytes(provenance.canonical(value))
        if result.returncode != 0 or before != after or not value['buildToolsUnchanged']:
            raise ValueError('Build failed or source changed during compilation; receipt retained')
        print(json.dumps({'component': args.component, 'sourceSha256': before['sha256'], 'outputSha256': value['outputs']['sha256']}))
    finally:
        if original is not None:
            next_env.write_bytes(original)


if __name__ == '__main__': main()
