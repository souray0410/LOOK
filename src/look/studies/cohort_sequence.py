"""Finite configuration order around cohort_delivery; no resource allocator.

One existing pipeline at a time, matched fitting arms inside that pipeline.
A failed configuration is isolated and recorded; no blind retry or seed expansion.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.analysis.cohort_publication import publish, read, PATTERNS, backbone_label


def validate_sequence(plan):
    if plan['schema']!='look_cohort_sequence_v1' or plan['test_access'] is not False:
        raise ValueError('Unregistered sequence')
    specs=[read(r['spec']) for r in plan['tasks']]
    if len({s['run_id'] for s in specs})!=len(specs) or len({s['output'] for s in specs})!=len(specs):
        raise ValueError('Duplicate run')
    fixed=('data_audit_sha256','data_root','training','seed','factor','rank','position','arms','search','framework_commit')
    for row,s in zip(plan['tasks'],specs):
        if file_sha256(row['spec'])!=row['spec_sha256']:raise ValueError('Spec changed')
        if s['seed']!=3416 or s['test_access'] is not False:raise ValueError('No repeat seed or test')
        if any(s[k]!=specs[0][k] for k in fixed):raise ValueError('Unmatched fixed factors')
        if row['role'] not in ('accepted_reference','execute'):raise ValueError('Unknown role')
    return specs


def refresh(plan, statuses):
    out=Path(plan['publication']);out.mkdir(parents=True,exist_ok=True)
    publications=[]
    for row in plan['tasks']:
        s=read(row['spec']);p=publish(s['output'],out/'configs'/s['run_id'])
        publications.append(p)
    # Only allowlisted aggregate content is exportable. Paths and commands stay onsite.
    current=dict(schema='look_cohort_sequence_publication_v1',sequence_id=plan['sequence_id'],
        test_used=False,order=[p['run_id'] for p in publications],
        configurations=publications,execution=statuses,
        next_question='external_method_A_vs_A_plus_LOOK_requires_adapter_acceptance')
    target=out/'current.json'
    if not target.exists() or read(target)!=current:atomic_write_json(current,target)
    has_external=any('mmtm' in p for p in publications)
    def host_label(p):return backbone_label(p['architecture'])+('＋MMTM适配宿主' if 'mmtm' in p else '')
    lines=['# LOOK 小队列累计进展：先逐配置跑通，再补重复种子','',
        '青光眼同一小队列：1,264 train / 296 dev，seed 3416，test封存。每个配置均包含共同新宿主、PCA与自由低秩残差两方法、两种缺失和完整正收益树。',
        '按配置顺序推进；LOOK固定一张卡，配置内两种拟合也顺序执行；另一张卡留给Radon_Bridge。已完成的ResNet50复用，不重训。当前这些是跨骨干验证，不能替代外部方法A与A+LOOK。','',
        '|顺序|骨干|状态|完整配置与路径|','|---:|---|---|---|']
    for i,p in enumerate(publications):
        state=statuses.get(p['run_id'],{}).get('state','not_started')
        label={'accepted':'已验收','running':'执行中','not_started':'未启动','needs_review':'故障待修复','accepted_reference':'已验收复用'}.get(state,state)
        progress=p.get('host_progress',{})
        if state=='running' and progress:label+=f'；宿主epoch {progress.get("epoch","—")} / 更新{progress.get("updates","—")}'
        lines.append(f'|{i+1}|{host_label(p)}|{label}|[配置、指标、树路径](configs/{p["run_id"]}/README.md)|')
    lines+=['','## 累计结果（Macro-F1，百分比）','',
        '只展示已完整匹配验收的配置；每一行应横向比较，跨骨干不把某个最高数值直接叫稳定最佳。','',
        '|骨干|缺失状态|不修正|PCA＋树|自由低秩残差＋树|残差−PCA（百分点）|','|---|---|---:|---:|---:|---:|']
    for p in publications:
        if p['matched_package']!='accepted':continue
        results={(r['method'],r['scenario']):r['metrics']['macro_f1'] for r in p['results']}
        for pattern,label in PATTERNS.items():
            a,b,c=(100*results[(m,pattern)] for m in ('host','pca_free_mean','residual_rrr'))
            lines.append(f'|{host_label(p)}|{label}|{a:.2f}|{b:.2f}|{c:.2f}|{c-b:+.2f}|')
    lines+=['','各配置详情含区间、AUROC/NLL/Brier反例、原模型停止轮次、初始化、秩/λ/缩放规则、实际路径及缓存计数。',
        '单种子与开发集选择结果只能支持探索性结论；没有稳定优越性或独立test结论。',
        '服务器自动汇总；GitHub由既有定时维护核验后同步。配置详情分别保留结果证据截止与发布核验时间。',
        '下一科学问题是已有方法A在加LOOK前后是否受益；MMTM适配审计尚未通过，不将它标为已运行。']
    if has_external:
        lines=[line.replace('当前这些是跨骨干验证，不能替代外部方法A与A+LOOK。','前两配置是已接受跨骨干参考；MMTM配置单独回答同一宿主A与A+LOOK。').replace('下一科学问题是已有方法A在加LOOK前后是否受益；MMTM适配审计尚未通过，不将它标为已运行。','MMTM配置包含Stage3作者门控适配；其实际阶段见上表，未验收结果不排名。每个配置内部严格使用同一宿主比较A和A+LOOK；不是原论文完整系统复现。') for line in lines]
    lines+=['','阅读入口：[研究问题、批准范围与术语说明](../reading_guide.zh-CN.md)。']
    text='\n'.join(lines)+'\n';p=out/'README.md'
    if not p.exists() or p.read_text()!=text:
        temp=out/'.README.tmp';temp.write_text(text);temp.replace(p)
    return current


def run(plan, *, launch=None, interval=60):
    specs=validate_sequence(plan);root=Path(plan['root']);root.mkdir(parents=True,exist_ok=True)
    if launch is None:
        def launch(s, row):
            log=(Path(s['output'])/'sequence_pipeline.log').open('a')
            child=subprocess.Popen([sys.executable,'-m','look.studies.cohort_delivery','--spec',row['spec'],'--stage','pipeline'],stdout=log,stderr=subprocess.STDOUT)
            log.close();return child
    with (root/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read(root/'status.json') if (root/'status.json').exists() else {}
        identity=stable_hash(plan)
        if previous and previous.get('identity')!=identity:raise ValueError('Sequence identity changed')
        statuses=previous.get('tasks',{})
        def save():
            atomic_write_json(dict(identity=identity,pid=os.getpid(),updated_at=time.time(),tasks=statuses),root/'status.json')
            refresh(plan,statuses)
        for row,s in zip(plan['tasks'],specs):
            rid=s['run_id'];runroot=Path(s['output']);runroot.mkdir(parents=True,exist_ok=True)
            if (runroot/'delivery/accepted.json').exists():
                published=publish(runroot,Path(plan['publication'])/'configs'/rid)
                if published['matched_package']!='accepted':raise ValueError('Incomplete reference')
                statuses[rid]=dict(state='accepted_reference' if row['role']=='accepted_reference' else 'accepted');save();continue
            if row['role']=='accepted_reference':raise ValueError('Reference not accepted')
            if statuses.get(rid,{}).get('state')=='needs_review':continue
            statuses[rid]=dict(state='running',started_at=time.time());save()
            child=launch(s,row)
            while child.poll() is None:
                # Publication faults never kill or restart healthy training.
                try:save()
                except Exception as e:
                    atomic_write_json(dict(state='needs_review',error=repr(e),time=time.time()),root/'publication_error.json')
                try:child.wait(timeout=interval)
                except subprocess.TimeoutExpired:pass
            try:
                if child.returncode:raise RuntimeError(f'Pipeline exit {child.returncode}')
                p=publish(runroot,Path(plan['publication'])/'configs'/rid)
                if p['matched_package']!='accepted':raise ValueError('Pipeline has no verified delivery')
                statuses[rid]=dict(state='accepted',finished_at=time.time())
            except Exception as e:
                statuses[rid]=dict(state='needs_review',error_type=type(e).__name__,finished_at=time.time())
                atomic_write_json(dict(state='needs_review',run_id=rid,error=repr(e),time=time.time()),root/'incidents'/f'{rid}.json')
            save()
        save()
        atomic_write_json(dict(identity=identity,state='accepted' if all(s['state'] in ('accepted','accepted_reference') for s in statuses.values()) else 'needs_review',time=time.time()),root/'completion.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);a=parser.parse_args();run(read(a.plan))

if __name__=='__main__':main()
