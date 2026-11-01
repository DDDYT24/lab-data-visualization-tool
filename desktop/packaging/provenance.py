"""Bind build inputs to outputs before staging or freezing a desktop candidate."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def manifest(root, paths):
    paths = sorted(set(paths))
    for path in paths:
        for member in (path, *path.parents):
            if member.is_symlink() or member.is_junction():
                raise ValueError('Build provenance rejects linked paths')
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Build input escaped the source checkout')
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(sha, paths))
    files = {p.relative_to(root).as_posix(): digest for p, digest in zip(paths, hashes)}
    return {'files': files, 'sha256': hashlib.sha256(canonical(files)).hexdigest()}


def host_permissions(root):
    directory = root/'desktop/host/permissions'
    members = [directory, *directory.rglob('*')]
    if any(p.is_symlink() or p.is_junction() for p in members):
        raise ValueError('Linked host permissions are not supported')
    files = sorted(p for p in members if p.is_file())
    if [p.relative_to(directory).as_posix() for p in files] != ['acceptance-ready.toml']:
        raise ValueError('Unexpected or missing host permissions file; remove no evidence automatically')
    value = tomllib.loads(files[0].read_text(encoding='utf-8'))
    rows = value.get('permission', [])
    if set(value) != {'permission'} or not isinstance(rows, list) or len(rows) != 2:
        raise ValueError('Host permission declarations differ')
    expected = {'allow-acceptance-editor-ready': (['acceptance_editor_ready'], []),
                'deny-acceptance-editor-ready': ([], ['acceptance_editor_ready'])}
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('identifier'), str):
            raise ValueError('Invalid host permission identifier')
        identifier = row.get('identifier')
        if identifier in seen:
            raise ValueError('Duplicate host permission identifier')
        seen.add(identifier)
        commands = row.get('commands', {})
        if (not isinstance(commands, dict) or identifier not in expected or set(row) - {'identifier', 'description', 'commands'}
                or set(commands) - {'allow', 'deny'}
                or (commands.get('allow', []), commands.get('deny', [])) != expected[identifier]):
            raise ValueError('Host permission commands or scope differ')
    return files


def source(root, kind):
    if kind == 'host':
        paths = [p for name in ['src', 'boot', 'capabilities'] for p in (root/'desktop/host'/name).rglob('*') if p.is_file()]
        paths += [root/'desktop/host'/name for name in ['Cargo.toml', 'Cargo.lock', 'build.rs', 'tauri.conf.json']]
        paths += [root/'desktop/runtime/adapter.js', root/'V2.0/assets/labviz-logo.ico']
        paths += host_permissions(root)
    elif kind == 'web':
        web = root/'V2.0/web'
        paths = [p for name in ['src', 'public', 'messages', 'scripts'] for p in (web/name).rglob('*') if p.is_file()]
        paths += [p for p in web.iterdir() if p.is_file() and p.suffix in {'.ts', '.json', '.mjs', '.cjs', '.js'} and p.name != 'next-env.d.ts']
        paths += [web/'node_modules/.package-lock.json']
    elif kind == 'api':
        paths = [p for name in ['labviz_api', 'migrations', 'samples/v22'] for p in (root/'V2.0/api'/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        paths += [root/'V2.0/api'/name for name in ['requirements.txt', 'requirements.lock.txt', 'pyproject.toml', 'alembic.ini']]
        paths += [p for name in ['contracts', 'assets', 'packaging/windows-v23/bin'] for p in (root/'V2.0'/name).rglob('*') if p.is_file()]
    else:
        raise ValueError('Unknown build component')
    return manifest(root, paths)


def host_build_output(root, log):
    directories = set()
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if (isinstance(value, dict) and value.get('reason') == 'build-script-executed'
                and value.get('package_id', '').split('#')[-1].split('@')[0] == 'labviz-desktop'):
            directory = Path(value['out_dir']).absolute()
            boundary = (root/'desktop/prototypes/tauri/target/release/build').resolve()
            if (directory.name != 'out' or not directory.parent.name.startswith('labviz-desktop-')
                    or any(p.is_symlink() or p.is_junction() for p in (directory, *directory.parents))
                    or not directory.resolve().is_relative_to(boundary)):
                raise ValueError('Cargo host build output escaped its private target')
            directories.add(directory.resolve())
    if len(directories) != 1:
        raise ValueError('The actual host Cargo build-script output is missing or ambiguous')
    return next(iter(directories))


def outputs(root, kind, host_out_dir=None):
    if kind == 'host':
        paths = [root/'desktop/prototypes/tauri/target/release/LabViz.exe']
        if host_out_dir is None:
            raise ValueError('Host outputs require the actual Cargo build-script directory')
        paths += [Path(host_out_dir)/name for name in ['acl-manifests.json', 'capabilities.json']]
    elif kind == 'web':
        dist = root/'V2.0/web/.next-v23-package'
        paths = [p for name in ['standalone', 'static'] for p in (dist/name).rglob('*') if p.is_file()]
        paths += [dist/'BUILD_ID']
    else:
        raise ValueError('Unknown build output')
    return manifest(root, paths)


def validate(root, directory):
    receipts = {}
    for kind in ['host', 'web']:
        path = directory/(kind+'.json')
        value = json.loads(path.read_text(encoding='utf-8'))
        host_out_dir = host_build_output(root, directory/'host.log') if kind == 'host' else None
        if (value.get('format') != 'labviz-component-build-v1' or value.get('component') != kind
                or value.get('exitCode') != 0 or value.get('buildToolsUnchanged') is not True or value.get('sourceBefore') != value.get('sourceAfter')
                or value['sourceAfter'] != source(root, kind) or value.get('outputs') != outputs(root, kind, host_out_dir)
                or (kind == 'host' and value.get('hostBuildScriptOutDir') != str(host_out_dir))
                or value.get('builderSha256') != sha(root/'desktop/scripts/build_verified.py')
                or value.get('provenanceToolSha256') != sha(Path(__file__))
                or value.get('logSha256') != sha(directory/(kind+'.log'))):
            raise ValueError('Missing, stale or mismatched build provenance: '+kind)
        receipts[kind] = value
    return receipts


def git_identity(root):
    return {'head': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'dirty': bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']))}
