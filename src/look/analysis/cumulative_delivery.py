"""Publish one cumulative view of a finite sequence, without starting training.

Only accepted per-configuration deliveries enter the metric table. Incomplete
cases remain visible in coverage. This publisher never reads test or chooses a
model, and does not recompute a cross-configuration statistical comparison.
"""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from look.runtime.state import file_sha256, stable_hash


def read(path):
    return json.loads(Path(path).read_text())


def collect(sequence, verify):
    config = read(sequence)
    cases, rows, seen = [], [], set()
    for item in config['tasks']:
        if file_sha256(item['config']) != item['sha256']:
            raise ValueError('Sequence configuration changed')
        task = read(item['config'])
        root = Path(task['task']['run_dir']).resolve()
        if str(root) in seen:
            raise ValueError('Duplicate run in sequence')
        seen.add(str(root))
        spec = read(root/'spec.json')
        if file_sha256(task['task']['spec']) != file_sha256(root/'spec.json'):
            raise ValueError('Run specification differs from registered specification')
        case = dict(run=str(root), spec_sha256=file_sha256(root/'spec.json'),
                    host=spec['host'], mode=spec['mode'],
                    factors=spec['spatial_factors'], dimensions=spec['latent_dims'],
                    state='awaiting_configuration_acceptance')
        if (root/'accepted.json').exists():
            verify(root, spec)
            case['state'] = 'awaiting_delivery_acceptance'
            delivery = root/'delivery'
            if (delivery/'accepted.json').exists():
                receipt = read(delivery/'accepted.json')
                if (receipt.get('state') != 'accepted' or receipt.get('test_access') is not False
                        or receipt['input_sha256'] != file_sha256(root/'accepted.json')):
                    raise ValueError('Delivery acceptance mismatch')
                required = {'results.json', 'paired_statistics.json', 'diagnostics.json',
                            'comparison.svg', 'README.zh-CN.md'}
                if not required.issubset(receipt['files']):
                    raise ValueError('Incomplete delivery')
                for name, digest in receipt['files'].items():
                    p = (delivery/name).resolve()
                    if not p.is_relative_to(delivery.resolve()) or file_sha256(p) != digest:
                        raise ValueError('Delivery evidence changed')
                case.update(state='accepted_delivery',
                            delivery_sha256=file_sha256(delivery/'accepted.json'))
                cells = set()
                for result in read(delivery/'results.json'):
                    cell = (result['pattern'], result['method'])
                    if cell in cells:
                        raise ValueError('Duplicate result cell')
                    cells.add(cell)
                    rows.append(dict(run=str(root), pattern=result['pattern'],
                                     method=result['method'],
                                     macro_f1=result['metrics']['macro_f1']))
                if cells != {(p, m) for p in ('oct_missing', 'cfp_missing') for m in ('host', 'search')}:
                    raise ValueError('Missing matched result cells')
        cases.append(case)
    return dict(schema='look_cumulative_delivery_v1', cases=cases, rows=rows,
                sequence_sha256=file_sha256(sequence), test_access=False,
                scope='registered_sequence_only; selected_development_results; not_test')


def render(snapshot, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = snapshot['rows']
    accepted = [c for c in snapshot['cases'] if c['state'] == 'accepted_delivery']
    if not accepted:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, len(accepted)*.7)), layout='constrained')
    for ax, pattern in zip(axes, ('oct_missing', 'cfp_missing')):
        for index, case in enumerate(accepted):
            scores = {r['method']:100*r['macro_f1'] for r in rows
                      if r['run'] == case['run'] and r['pattern'] == pattern}
            ax.plot([scores['host'], scores['search']], [index, index], color='grey')
            ax.scatter(scores['host'], index, marker='o', color='grey', label='Frozen host' if index == 0 else None)
            ax.scatter(scores['search'], index, marker='x', color='tab:blue', label='Selected LOOK' if index == 0 else None)
        ax.set_yticks(range(len(accepted)), [Path(c['run']).name for c in accepted])
        ax.set_xlim(0, 100)
        ax.set_xlabel('Development macro-F1 (%) — higher is better')
        ax.set_title(pattern)
        ax.legend()
    fig.savefig(folder/'comparison.svg')
    plt.close(fig)


def publish(snapshot, output, renderer=render):
    """Content-addressed releases + atomic current link; old view survives failure."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        digest = stable_hash(snapshot)
        target = root/'releases'/digest
        current = root/'current'
        if target.exists():
            receipt = read(target/'manifest.json')
            if receipt.get('snapshot') != digest or read(target/'snapshot.json') != snapshot:
                raise ValueError('Published identity changed')
            required = {'snapshot.json', 'results.csv', 'README.zh-CN.md'}
            if not required.issubset(receipt['files']):
                raise ValueError('Published manifest incomplete')
            for name, sha in receipt['files'].items():
                path = (target/name).resolve()
                if not path.is_relative_to(target.resolve()) or file_sha256(path) != sha:
                    raise ValueError('Published result changed')
            if current.is_symlink() and current.resolve() == target.resolve():
                return dict(state='unchanged', snapshot=digest)
            link = root/f'.current-{os.getpid()}'
            if link.is_symlink():
                link.unlink()
            link.symlink_to(target.relative_to(root))
            os.replace(link, current)
            return dict(state='recovered_publication', snapshot=digest)
        target.parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.publish-', dir=root))
        try:
            (temporary/'snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=['run', 'pattern', 'method', 'macro_f1'])
            writer.writeheader(); writer.writerows(snapshot['rows'])
            (temporary/'results.csv').write_text(stream.getvalue())
            renderer(snapshot, temporary)
            lines = ['# LOOK 累计配置交付', '',
                     '范围仅为登记序列，未接入历史末端案例；不是全项目总结果。',
                     '每行是同一运行的冻结宿主与选定LOOK。横轴为dev macro-F1%，越大越好；纵轴为运行ID。',
                     '各配置不同搜索策略/起点不得混称重复种子；此图不产生跨配置显著性结论。',
                     '逐配置区间与来源见原run/delivery；同一范围后续验收结果累计加入本图。', '',
                     '|运行|策略|状态|', '|---|---|---|']
            lines.extend(f"|{Path(c['run']).name}|{c['mode']}|{c['state']}|" for c in snapshot['cases'])
            if (temporary/'comparison.svg').exists():
                lines += ['', '![累计匹配结果](comparison.svg)']
            (temporary/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
            files = {p.name:file_sha256(p) for p in temporary.iterdir() if p.is_file()}
            (temporary/'manifest.json').write_text(json.dumps(dict(snapshot=digest, files=files), indent=2)+'\n')
            if target.exists():
                raise ValueError('Existing release requires explicit reconciliation')
            temporary.rename(target)
            link = root/f'.current-{os.getpid()}'
            link.symlink_to(target.relative_to(root))
            os.replace(link, current)
            return dict(state='published', snapshot=digest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sequence', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    from look.studies.search_case import verify_case
    print(json.dumps(publish(collect(a.sequence, verify_case), a.output)))


if __name__ == '__main__':
    main()
