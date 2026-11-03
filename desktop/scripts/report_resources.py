"""Report background measurements without accepting native desktop resource gates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def report(trial, bundle):
    samples = [json.loads(line) for line in (trial / 'resources.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (trial / 'headless-events.jsonl').read_text().splitlines()]
    build = json.loads((bundle / 'desktop-build.json').read_text())
    sizes = {}
    for path in bundle.rglob('*'):
        if path.is_symlink() or path.is_junction():
            raise ValueError('Resource measurements reject linked bundle paths')
        if path.is_file():
            relative = path.relative_to(bundle).as_posix()
            component = ('node' if relative.startswith('app/runtime/node/') else
                         'python' if relative.startswith('app/runtime/python/') else 'applicationAndHost')
            sizes[component] = sizes.get(component, 0) + path.stat().st_size
    phases = {}
    for sample in samples:
        name = sample.get('phase', 'unclassified')
        entry = phases.setdefault(name, {'samples': 0, 'peakPrivateCommitBytes': 0, 'peakWorkingSetBytes': 0, 'peakCpuPercentOneCore': 0})
        entry['samples'] += 1
        entry['peakPrivateCommitBytes'] = max(entry['peakPrivateCommitBytes'], sample['privateCommitBytes'])
        entry['peakWorkingSetBytes'] = max(entry['peakWorkingSetBytes'], sum(p['workingSetBytes'] for p in sample['processes']))
        entry['peakCpuPercentOneCore'] = max(entry['peakCpuPercentOneCore'], sample.get('cpuPercentOneCore') or 0)
    cycles = []
    hidden_work = []
    for index, event in enumerate(events):
        if event['event'] != 'client-hidden':
            continue
        end = next((value['utcSeconds'] for value in events[index+1:] if value['event'] == 'client-shown'), samples[-1]['utcSeconds'])
        hidden_work.extend(s for s in samples if event['utcSeconds'] <= s['utcSeconds'] < end and s.get('phase') == 'figure-generation')
    for index, event in enumerate(events):
        if event['event'] != 'services-suspended':
            continue
        end = next((value['utcSeconds'] for value in events[index+1:] if value['event'] in {'client-shown', 'headless-page-created'}), samples[-1]['utcSeconds'])
        stable = [s for s in samples if event['utcSeconds'] + 1 <= s['utcSeconds'] < end]
        if not stable:
            continue
        med = statistics.median(s['privateCommitBytes'] for s in stable)
        cycles.append({'index': len(cycles) + 1, 'samples': len(stable), 'medianPrivateCommitMiB': med / 2**20,
                       'maximumPrivateCommitMiB': max(s['privateCommitBytes'] for s in stable) / 2**20,
                       'ownedProcessCounts': sorted(set(len(s['processes']) for s in stable)),
                       'processNames': sorted(set(p['name'] for s in stable for p in s['processes']))})
    return {'format': 'labviz-headless-resource-report-v1', 'nativeAcceptance': False,
            'scope': 'Controller/services plus headless Chromium; excludes native host and WebView2. CPU uses one-core percentages.',
            'build': build, 'payloadManifestSha256': hashlib.sha256((bundle / 'app/payload-manifest.json').read_bytes()).hexdigest(),
            'previewLogicalBytes': sum(sizes.values()), 'componentLogicalBytes': sizes,
            'installerBytes': None, 'missingWebView2DownloadBytes': None,
            'sizeCaveat': 'An unfrozen preview directory is not an installed program or final installer.',
            'phases': phases, 'stableSuspendedCycles': cycles,
            'phaseCaveat': 'Driver phase labels can persist through later checks; phase peaks are not steady idle measurements.',
            'hiddenFigureWork': {'samples': len(hidden_work),
                                 'peakPrivateCommitBytes': max((s['privateCommitBytes'] for s in hidden_work), default=None),
                                 'peakWorkingSetBytes': max((s['workingSetBytes'] for s in hidden_work), default=None),
                                 'peakCpuPercentOneCore': max((s['cpuPercentOneCore'] for s in hidden_work), default=None)},
            'firstToLastMedianChangeMiB': cycles[-1]['medianPrivateCommitMiB'] - cycles[0]['medianPrivateCommitMiB'] if len(cycles) > 1 else None,
            'topologyDecision': 'Keep Next conditionally. Final native numeric gates and mandatory reports remain required; static migration has not been triggered.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = report(args.trial.resolve(), args.bundle.resolve())
    args.output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'nativeAcceptance': False, 'stableSuspendedCycles': len(value['stableSuspendedCycles']), 'previewLogicalBytes': value['previewLogicalBytes']}))


if __name__ == '__main__':
    main()
