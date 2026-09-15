"""Read-only, continuously refreshed terminal evidence; never selects a winner.

Checks accepted metadata and the report/spec digests on every refresh. The original
worker/registry performs full cache and prediction acceptance; this observer does
not replace that acceptance, read participant arrays or launch scientific work.
"""
import argparse
import csv
import fcntl
import io
import json
import math
from pathlib import Path
import statistics
import time

from look.analysis.linear_labels import METHOD_LABELS
from look.runtime.state import atomic_write_json, file_sha256, stable_hash

ARMS = tuple(METHOD_LABELS)
PATTERNS = ('oct_missing', 'cfp_missing')


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, receipt, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes run')
    if file_sha256(path) != receipt['files'].get(name):
        raise ValueError('Changed report artifact: ' + name)
    return read(path)


def collect(task):
    root = Path(task['run_dir'])
    receipt = read(root / 'accepted.json')
    spec = checked(root, receipt, 'spec.json')
    if (receipt.get('state') != 'accepted' or receipt.get('schema') != 'look_terminal_stage_v1'
        or receipt.get('identity') != stable_hash(spec) or receipt.get('test_access') is not False
        or receipt.get('profile') is not False or spec.get('test_access') is not False
        or any(receipt.get(k) is not True for k in ('full_development_mhd_replay', 'host_frozen', 'reload_exact'))):
        raise ValueError('Unaccepted or unsealed terminal evidence')
    if file_sha256(task['spec']) != task['spec_sha256'] or read(task['spec']) != spec:
        raise ValueError('Queue/spec identity mismatch')
    suite = checked(root, receipt, 'development/suite.json')
    stats = checked(root, receipt, 'report/paired_statistics.json')
    if suite.get('test_access') is not False or stats.get('test_access') is not False:
        raise ValueError('Only development evidence allowed')
    if stats.get('iterations') != 10000 or stats.get('family') != '21_terminal_stage_contrasts_not_global_LOOK':
        raise ValueError('Unmatched comparison family')
    selections = {p: checked(root, receipt, 'selection/' + p + '.json') for p in PATTERNS}
    rows = {}; h = spec['host']
    for record in suite['records']:
        key = (record['method'], record['scenario'])
        if key in rows:
            raise ValueError('Duplicate evaluation view')
        value = float(record['metrics']['macro_f1'])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid macro-F1')
        rows[key] = dict(h, run=root.name, method=key[0], scenario=key[1], macro_f1=value)
    needed = {(m, p) for m in (*ARMS, 'host') for p in PATTERNS}
    if not needed.issubset(rows):
        raise ValueError('Incomplete matched three-arm evidence')
    for row in rows.values():
        p = row['scenario']; baseline = rows.get(('host', p))
        row['delta_host_pp'] = 100 * (row['macro_f1'] - baseline['macro_f1']) if baseline else None
        if p in PATTERNS:
            row.update(rank_budget=selections[p]['rank'], ridge_lambda=selections[p]['lambda_'])
    return list(rows.values()), dict(host=h, run=root.name,
        receipt_sha256=file_sha256(root/'accepted.json'), comparison_family=stats,
        evidence_scope='report/spec digests verified; full arrays/cache acceptance owned by original registry')


def summarize(rows):
    views = {}; groups = {}
    for row in rows:
        h = tuple(row[k] for k in ('disease', 'architecture', 'position'))
        if row['method'] in ARMS and row['scenario'] != 'complete':
            views.setdefault((*h, row['seed'], row['scenario']), {})[row['method']] = row['macro_f1']
        if row['method'] in ARMS and row['scenario'] in PATTERNS:
            groups.setdefault((*h, row['method'], row['scenario']), {})[row['seed']] = row['macro_f1']
    counts = {kind: {m: {str(i): 0 for i in (1, 2, 3)} for m in ARMS} for kind in ('pure', 'mixed')}
    for key, values in views.items():
        if set(values) != set(ARMS):
            raise ValueError('Incomplete ranking view')
        kind = 'mixed' if key[-1].startswith('mixed_') else 'pure'
        for m, v in values.items():
            rank = 1 + sum(x > v + 1e-12 for x in values.values())
            counts[kind][m][str(rank)] += 1
    complete = []
    for key, values in sorted(groups.items()):
        if set(values) == {3416, 3417, 3418}:
            complete.append(dict(zip(('disease','architecture','position','method','scenario'),key),
                mean=statistics.mean(values.values()), sample_sd=statistics.stdev(values.values()), seeds=values))
    return dict(descriptive_ranks=counts, complete_three_seed_rows=complete,
        rank_note='Competitive ranks with ties; correlated views, not independent replications or significance tests',
        selected_method=None)


def tick(feed, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    queue = read(feed)
    if queue.get('schema') != 'look_terminal_work_feed_v1' or queue.get('test_access') is not False:
        raise ValueError('Unsealed terminal feed')
    rows = []; evidence = []; errors = []; pending = []; seen = set()
    for task in queue['tasks']:
        name = str(Path(task['run_dir']).resolve())
        if name in seen:
            raise ValueError('Duplicate queue run')
        seen.add(name)
        if not (Path(name)/'accepted.json').exists():
            pending.append(task['id']); continue
        try:
            items, source = collect(task)
            identity = tuple(source['host'][k] for k in ('disease','architecture','position','seed'))
            if any(tuple(e['host'][k] for k in ('disease','architecture','position','seed')) == identity for e in evidence):
                raise ValueError('Duplicate host/seed evidence')
            rows.extend(items); evidence.append(source)
        except Exception as exc:
            errors.append(dict(task=task['id'], error=repr(exc)))
    result = dict(updated_at=time.time(), accepted_cases=len(evidence), expected_hosts=queue['expected_hosts'],
        pending=pending, dependency_waiting=queue.get('waiting', {}), errors=errors,
        registry_errors=queue.get('errors', []), feed_stale=time.time()-queue['updated_at'] > 1800,
        test_access=False, scope='terminal_only_not_full_progressive_LOOK', **summarize(rows))
    result['complete'] = (len(evidence)==queue['expected_hosts'] and not errors and not result['registry_errors']
                          and not result['feed_stale'] and not pending and not result['dependency_waiting'])
    atomic_write_json(result, out/'summary.json'); atomic_write_json(evidence, out/'evidence.json')
    atomic_write_json(rows, out/'results.json')
    fields = ['disease','architecture','position','seed','run','method','scenario','macro_f1','delta_host_pp','rank_budget','ridge_lambda']
    buf=io.StringIO(); writer=csv.DictWriter(buf, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (out/'results.csv.partial').write_text(buf.getvalue()); (out/'results.csv.partial').replace(out/'results.csv')
    lines=['# LOOK 三种残差修正：自动开发集汇总','',
        f"已接受末级案例：{len(evidence)}/{queue['expected_hosts']}；完整逐层研究不能按这个数字计数。",
        f"本次读取异常：{len(errors)}；注册器异常：{len(result['registry_errors'])}；清单过期：{result['feed_stale']}。",
        '三组同秩、同正则。自由均值是重点候选，非预定赢家；不据排名自动删方法或改配置。',
        'F1单位为%；相对宿主差值单位为百分点，正值更好，负值为反例。全部是dev，test未访问。',
        '完整三种子均值/样本标准差仅在3416–3418齐全后生成；现有单案例区间见evidence.json，不冒充全局区间。',
        '', '| 疾病/架构/位置/种子 | 缺失 | 方法 | F1 (%) | 相对宿主 (pp) |','|---|---|---|---:|---:|']
    for r in rows:
        if r['method'] in ARMS and r['scenario'] in PATTERNS:
            lines.append(f"| {r['disease']}/{r['architecture']}/{r['position']}/{r['seed']} | {r['scenario']} | {METHOD_LABELS[r['method']]} | {100*r['macro_f1']:.3f} | {r['delta_host_pp']:+.3f} |")
    (out/'README.zh-CN.md.partial').write_text('\n'.join(lines)+'\n'); (out/'README.zh-CN.md.partial').replace(out/'README.zh-CN.md')
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--feed',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--watch',action='store_true'); args=parser.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                result=tick(args.feed,out)
                atomic_write_json(dict(state='needs_review' if result['errors'] or result['registry_errors'] or result['feed_stale'] else 'healthy',
                                       updated_at=time.time(),complete=result['complete']),out/'observer_status.json')
                if not args.watch or result['complete']:return
            except Exception as exc:
                atomic_write_json(dict(state='needs_review',error=repr(exc),updated_at=time.time()),out/'observer_status.json')
                if not args.watch:raise
            if (out/'stop.json').exists():return
            time.sleep(300)


if __name__=='__main__':
    main()
