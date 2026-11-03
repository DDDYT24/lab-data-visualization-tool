"""Collect exact dependency notices, including LICENCE and version-pinned supplements."""
import csv
import email
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

from provenance import sha, canonical


def notice(path):
    return bool(re.match(r'^(licen[cs]e|copying|notice|copyright)(?:$|[._-])', path.name, re.I))


def web_supplement(package, metadata, supplements, source):
    matches = [item for item in source.get('webPackages', [])
               if (item['name'], item['version']) == (metadata['name'], metadata['version'])]
    if not matches:
        return [], None
    if len(matches) != 1:
        raise ValueError('Ambiguous web notice supplement')
    item = matches[0]
    if sha(package/'package.json') != item['packageMetadataSha256']:
        raise ValueError('Web supplement package metadata differs')
    for name, expected in item['implementationSha256'].items():
        path = package/name
        if path.is_symlink() or not path.resolve().is_relative_to(package.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError('Web supplement implementation differs: '+name)
    paths = []
    for name in item['noticeFiles']:
        path = supplements/name
        evidence = source['files'][name]
        if path.is_symlink() or not path.resolve().is_relative_to(supplements.resolve()) or not path.is_file() or sha(path) != evidence['sha256']:
            raise ValueError('Web supplement notice hash differs')
        paths.append(path)
    if not paths:
        raise ValueError('Empty web notice supplement')
    return paths, item


def python_inventory(bundle):
    site = bundle/'app/runtime/python/Lib/site-packages'
    records = []
    for path in sorted(site.glob('*.dist-info/METADATA')):
        info = email.message_from_bytes(path.read_bytes())
        candidates = {p for p in path.parent.rglob('*') if p.is_file() and notice(p)}
        record = path.parent/'RECORD'
        if record.exists():
            for row in csv.reader(record.read_text(encoding='utf-8').splitlines()):
                if not row or not notice(Path(row[0])):
                    continue
                item = (site/row[0]).resolve()
                if not item.is_relative_to(bundle.resolve()):
                    raise ValueError('Wheel notice escaped the candidate')
                if item.is_file(): candidates.add(item)
        for item in candidates:
            if item.is_symlink() or item.is_junction(): raise ValueError('Linked wheel notice')
        files = {p.relative_to(bundle).as_posix(): sha(p) for p in sorted(candidates)}
        if not files:
            raise ValueError('Python distribution has no collected notice: '+str(info['Name']))
        records.append({'name': str(info['Name']), 'version': str(info['Version']),
                        'license': str(info['License-Expression'] or info['License'] or ''),
                        'licenseFiles': files, 'status': 'bundled-notices'})
    return records


def rust_inventory(root, bundle, cargo):
    environment = {**os.environ, 'CARGO_HOME': str(cargo), 'RUSTUP_HOME': str(cargo.parent/'rustup')}
    command = [str(cargo/'bin/cargo.exe'), 'tree', '--locked', '--offline', '--target', 'x86_64-pc-windows-msvc',
               '--prefix', 'none', '--format', '{p}', '--manifest-path', str(root/'desktop/host/Cargo.toml')]
    def closure(edges):
        text = subprocess.check_output([*command, '--edges', edges], env=environment, text=True)
        return {(m.group(1), m.group(2)) for line in text.splitlines() if (m := re.match(r'^([\w-]+) v([\w.+-]+)', line))}
    all_packages, normal_packages = closure('normal,build'), closure('normal')
    supplements = root/'desktop/packaging/license-supplements'
    sources = json.loads((supplements/'sources.json').read_text(encoding='utf-8'))
    fallbacks = {(x['name'], x['version']): x for x in sources['packages']}
    result, sections = [], []
    for package in tomllib.loads((root/'desktop/host/Cargo.lock').read_text(encoding='utf-8'))['package']:
        key = package['name'], package['version']
        if not package.get('source', '').startswith('registry+') or key not in all_packages: continue
        hits = list((cargo/'registry/src').glob('*/'+key[0]+'-'+key[1]))
        if len(hits) != 1: raise ValueError('Missing exact Rust source: '+str(key))
        source = hits[0]; metadata = tomllib.loads((source/'Cargo.toml').read_text(encoding='utf-8'))
        notices = [p for p in source.iterdir() if p.is_file() and notice(p)]
        explicit = metadata['package'].get('license-file')
        if explicit:
            item = (source/explicit).resolve()
            if not item.is_relative_to(source.resolve()) or not item.is_file(): raise ValueError('Invalid crate license-file')
            notices.append(item)
        destination = bundle/'licenses/rust'/('-'.join(key)); destination.mkdir(parents=True, exist_ok=True)
        origin = {'repository': metadata['package'].get('repository'), 'crateChecksum': package.get('checksum')}
        collected = []
        for path in sorted(set(notices)):
            target = destination/path.name; shutil.copyfile(path, target); collected.append(target)
        if not collected:
            fallback = fallbacks.get(key)
            vcs = json.loads((source/'.cargo_vcs_info.json').read_text(encoding='utf-8'))['git']['sha1']
            if not fallback or vcs != fallback['vcsCommit']:
                raise ValueError('Unresolved Rust notice: '+str(key))
            file = fallback['file']; evidence = sources['files'][file]
            if sha(supplements/file) != evidence['sha256']: raise ValueError('License supplement hash differs')
            target = destination/file; shutil.copyfile(supplements/file, target); collected.append(target)
            origin.update(supplement=evidence, vcsCommit=vcs)
            if fallback.get('sourceNotice'):
                target = destination/fallback['sourceNotice']; shutil.copyfile(source/fallback['sourceNotice'], target); collected.append(target)
        files = {p.relative_to(bundle).as_posix(): sha(p) for p in collected}
        role = 'build-only' if key not in normal_packages else 'proc-macro' if metadata.get('lib', {}).get('proc-macro') else 'normal-dependency'
        item = {'name': key[0], 'version': key[1], 'license': metadata['package'].get('license'),
                'role': role, 'noticeFiles': files, 'origin': origin, 'status': 'version-pinned-upstream' if not notices else 'crate-notices'}
        result.append(item)
        sections.append('\n===== '+key[0]+' '+key[1]+'; '+str(item['license'])+'; '+role+' =====\n'+
                        '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in collected))
    (bundle/'licenses/RUST-THIRD-PARTY-LICENSES.txt').write_text('\n'.join(sections), encoding='utf-8')
    return result


def web_inventory(root, bundle):
    web = root/'V2.0/web'
    lock = json.loads((web/'package-lock.json').read_text(encoding='utf-8'))
    supplements = root/'desktop/packaging/license-supplements'
    source = json.loads((supplements/'sources.json').read_text(encoding='utf-8'))
    records, sections, issues = [], [], []
    for relative, info in lock['packages'].items():
        if not relative or info.get('dev'): continue
        package = web/relative
        if not package.exists():
            records.append({'path': relative, 'version': info.get('version'), 'status': 'not-installed', 'optional': info.get('optional', False), 'os': info.get('os')})
            continue
        metadata = json.loads((package/'package.json').read_text(encoding='utf-8'))
        paths = [p for p in package.iterdir() if p.is_file() and notice(p)]
        for sub in [p for p in package.iterdir() if p.is_dir() and re.fullmatch(r'licenses?|notices?|compiled', p.name, re.I)]:
            paths += [p for p in sub.rglob('*') if p.is_file() and notice(p)]
        status = 'package-notices'
        supplement = None
        if not paths:
            for file in package.iterdir():
                if file.is_file() and file.name.lower().startswith('readme'):
                    text = file.read_text(encoding='utf-8', errors='replace')
                    if 'Permission is hereby granted' in text and 'Copyright' in text:
                        paths.append(file)
            status = 'readme-notice'
        if not paths and metadata.get('name') in {'@next/env', '@next/swc-win32-x64-msvc'}:
            next_info = json.loads((web/'node_modules/next/package.json').read_text(encoding='utf-8'))
            if next_info['version'] != metadata['version']: raise ValueError('Next monorepo notice version differs')
            paths = [web/'node_modules/next/license.md']; status = 'same-release-monorepo-notice'
        if not paths and metadata.get('name') == '@schummar/icu-type-parser' and metadata['version'] == '1.21.5':
            name = 'icu-type-parser-1.21.5-LICENSE.txt'
            if sha(supplements/name) != source['files'][name]['sha256']: raise ValueError('Web license supplement differs')
            paths = [supplements/name]; status = 'version-pinned-upstream'
        if not paths:
            paths, supplement = web_supplement(package, metadata, supplements, source)
            if paths:
                status = 'implementation-matched-upstream-notice'
        destination = bundle/'licenses/web'/hashlib.sha256(relative.encode()).hexdigest()[:16]
        destination.mkdir(parents=True, exist_ok=True)
        copied = {}
        for index, path in enumerate(paths):
            target = destination/(str(index)+'-'+path.name); shutil.copyfile(path, target)
            copied[target.relative_to(bundle).as_posix()] = sha(target)
        if not paths:
            status = 'declared-license-only'
            issues.append({'path': relative, 'name': metadata.get('name'), 'version': metadata.get('version'),
                           'declaredLicense': metadata.get('license'), 'issue': 'No package-specific full notice found in this release; no copyright holder inferred'})
            shutil.copyfile(package/'package.json', destination/'package.json')
        item = {'path': relative, 'name': metadata.get('name'), 'version': metadata['version'],
                'license': metadata.get('license'), 'status': status, 'noticeFiles': copied,
                'packageMetadataSha256': sha(package/'package.json'), 'registryIntegrity': info.get('integrity')}
        if supplement:
            item['noticeSupplement'] = supplement
        records.append(item)
        sections.append('\n===== '+relative+' '+metadata['version']+'; '+status+' =====\n'+
                        ('\n'.join(path.read_text(encoding='utf-8', errors='replace') for path in paths)
                         if paths else 'Declared license: '+str(metadata.get('license'))+'. No package-specific full notice was supplied; see WEB-DEPENDENCIES.json.'))
    result = {'format': 'labviz-web-notices-v1', 'packages': records, 'reviewIssues': issues,
              'scope': 'Installed production lockfile closure, including browser-bundled code; absent optional/platform packages recorded separately',
              'publicReleaseAccepted': False}
    (bundle/'licenses/WEB-THIRD-PARTY-LICENSES.txt').write_text('\n'.join(sections), encoding='utf-8')
    (bundle/'licenses/WEB-DEPENDENCIES.json').write_bytes(canonical(result))
    return result
