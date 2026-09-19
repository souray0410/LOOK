"""Finite configuration order around cohort_delivery; no resource allocator.

One existing pipeline at a time, matched fitting arms inside that pipeline.
A failed configuration is isolated and recorded; no blind retry or seed expansion.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.analysis.cohort_publication import publish, read, PATTERNS, backbone_label



FUSION_EXTENSION_SCHEMA='look_small_cohort_fusion_stage_extension_v1'
FUSION_POSITION_LABELS={
    'middle':'中层特征图融合（Stage 2后）',
    'deep':'深层特征图融合（Stage 4后）',
    'features':'每眼特征向量融合',
}


def validate_fusion_extension(extension,current):
    if extension.get('schema')!=FUSION_EXTENSION_SCHEMA:
        raise ValueError('Unknown small-cohort fusion extension')
    if extension.get('state')!='scientifically_accepted' or extension.get('test_used') is not False:
        raise ValueError('Fusion extension is not scientifically accepted')
    deep=extension.get('deep_reference',{})
    if deep.get('role')!='strict_reuse_no_retraining':
        raise ValueError('Fusion extension deep reference changed')
    deep_matches=[p for p in current['configurations'] if p.get('run_id')==deep.get('run_id')]
    if len(deep_matches)!=1 or deep_matches[0].get('position')!='deep':
        raise ValueError('Fusion extension must reuse exactly one registered deep configuration')
    configs=extension.get('configurations',[])
    if [(p.get('position'),p.get('run_id')) for p in configs] != [
            ('middle','2026_09_19_02_40_42_281912_middle'),
            ('features','2026_09_19_02_40_42_281912_features')]:
        raise ValueError('Fusion extension must register exactly middle/features once')
    if any(p.get('architecture')!='resnet18' or p.get('matched_package')!='accepted' for p in configs):
        raise ValueError('Fusion extension configuration is not accepted ResNet18')
    if any(p.get('run_id') in {q.get('run_id') for q in current['configurations']} for p in configs):
        raise ValueError('Fusion extension duplicated an existing configuration')
    return extension


def register_fusion_extension(current,fusion_current,left_audit_sha256):
    if (fusion_current.get('schema')!='look_fusion_stage_publication_v1'
            or fusion_current.get('complete') is not True or fusion_current.get('test_used') is not False
            or fusion_current.get('order')!=['deep','middle','features']):
        raise ValueError('Fusion publication is not complete')
    by_position={p['position']:p for p in fusion_current['configurations']}
    extension=dict(schema=FUSION_EXTENSION_SCHEMA,state='scientifically_accepted',test_used=False,
        task_id=fusion_current['task_id'],sequence_id=fusion_current['sequence_id'],
        deep_reference=dict(run_id=by_position['deep']['run_id'],role='strict_reuse_no_retraining'),
        configurations=[by_position['middle'],by_position['features']],
        execution={by_position['middle']['run_id']:{'state':'accepted'},
                   by_position['features']['run_id']:{'state':'accepted'}},
        fusion_publication_sha256=stable_hash(fusion_current),
        left_independent_audit_sha256=left_audit_sha256,
        conclusion_scope='accepted_single_seed_dev_mechanism_not_external_A_B_C_completion')
    return validate_fusion_extension(extension,current)


def _carry_fusion_extension(previous,current):
    extension=previous.get('fusion_stage_extension')
    if extension is not None:
        current['fusion_stage_extension']=validate_fusion_extension(extension,current)
    return current


def _display_publications(current):
    extension=current.get('fusion_stage_extension')
    return [*current['configurations'],*(extension.get('configurations',[]) if extension else [])]


def _display_execution(current):
    result=dict(current.get('execution',{}))
    extension=current.get('fusion_stage_extension')
    if extension:
        result.update(extension.get('execution',{}))
    return result


def _publication_science_projection(publication):
    value=copy.deepcopy(publication)
    value.pop('publication_verified_at_utc',None)
    value.pop('source_evidence_sha256',None)
    for row in value.get('search_details',[]):
        row.pop('feature_costs_provenance',None)
    return value


def _preserve_accepted_config_records(previous,publications):
    old={p.get('run_id'):p for p in previous.get('configurations',[])}
    result=[]
    for publication in publications:
        prior=old.get(publication.get('run_id'))
        if (prior is not None and prior.get('matched_package')=='accepted'
                and publication.get('matched_package')=='accepted'
                and _publication_science_projection(prior)==_publication_science_projection(publication)):
            result.append(prior)
        else:
            result.append(publication)
    return result


def validate_sequence(plan):
    if plan.get('study_kind')=='fusion_stage_v1':
        from look.studies.cohort_fusion_stage import validate_sequence as validate_fusion_sequence
        return validate_fusion_sequence(plan)
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
    if plan.get('study_kind')=='fusion_stage_v1':
        from look.analysis.fusion_stage_publication import refresh as refresh_fusion_stage
        return refresh_fusion_stage(plan,statuses)
    out=Path(plan['publication']);out.mkdir(parents=True,exist_ok=True)
    target=out/'current.json'
    previous=read(target) if target.exists() else {}
    publications=[]
    for row in plan['tasks']:
        s=read(row['spec']);p=publish(s['output'],out/'configs'/s['run_id'])
        publications.append(p)
    publications=_preserve_accepted_config_records(previous,publications)
    # Only allowlisted aggregate content is exportable. Paths and commands stay onsite.
    current=dict(schema='look_cohort_sequence_publication_v1',sequence_id=plan['sequence_id'],
        test_used=False,order=[p['run_id'] for p in publications],
        configurations=publications,execution=statuses,
        next_question='external_method_A_vs_A_plus_LOOK_requires_adapter_acceptance')
    current=_carry_fusion_extension(previous,current)
    if not target.exists() or previous!=current:atomic_write_json(current,target)
    displayed=_display_publications(current);display_execution=_display_execution(current)
    has_external=any('mmtm' in p for p in displayed)
    def host_label(p):
        base=backbone_label(p['architecture'])
        if 'mmtm' in p:
            return base+'＋MMTM适配宿主／'+FUSION_POSITION_LABELS.get(p.get('position','deep'),'融合位置未登记')
        if p.get('architecture')=='resnet18' and p.get('position') in FUSION_POSITION_LABELS:
            return base+'／'+FUSION_POSITION_LABELS[p['position']]
        return base
    lines=['# LOOK 小队列累计进展：先逐配置跑通，再补重复种子','',
        '青光眼同一小队列：1,264 train / 296 dev，seed 3416，test封存。每个配置均包含共同新宿主、PCA与自由低秩残差两方法、两种缺失和完整正收益树。',
        '原三配置保持原顺序和原证据截止；随后注册已独立验收的R18 middle/features融合位置机制扩展。deep只引用原R18 deep记录，不重复计为新宿主。当前累计表同时覆盖跨骨干、MMTM适配A及融合位置机制，但仍不能替代尚未完成的外部A/B/C全套匹配。','',
        '|顺序|骨干|状态|完整配置与路径|','|---:|---|---|---|']
    for i,p in enumerate(displayed):
        state=display_execution.get(p['run_id'],{}).get('state','not_started')
        label={'accepted':'已验收','running':'执行中','not_started':'未启动','needs_review':'故障待修复','accepted_reference':'已验收复用'}.get(state,state)
        progress=p.get('host_progress',{})
        if state=='running' and progress:label+=f'；宿主epoch {progress.get("epoch","—")} / 更新{progress.get("updates","—")}'
        lines.append(f'|{i+1}|{host_label(p)}|{label}|[配置、指标、树路径](configs/{p["run_id"]}/README.md)|')
    lines+=['','## 累计结果（Macro-F1，百分比）','',
        '只展示已完整匹配验收的配置；每一行应横向比较，跨骨干不把某个最高数值直接叫稳定最佳。','',
        '|骨干|缺失状态|不修正|PCA＋树|自由低秩残差＋树|残差−PCA（百分点）|','|---|---|---:|---:|---:|---:|']
    for p in displayed:
        if p['matched_package']!='accepted':continue
        results={(r['method'],r['scenario']):r['metrics']['macro_f1'] for r in p['results']}
        for pattern,label in PATTERNS.items():
            a,b,c=(100*results[(m,pattern)] for m in ('host','pca_free_mean','residual_rrr'))
            lines.append(f'|{host_label(p)}|{label}|{a:.2f}|{b:.2f}|{c:.2f}|{c-b:+.2f}|')
    lines+=['','各配置详情含区间、AUROC/NLL/Brier反例、原模型停止轮次、初始化、秩/λ/缩放规则、实际路径及缓存计数。',
        '单种子与开发集选择结果只能支持探索性结论；没有稳定优越性或独立test结论。',
        '服务器自动汇总；GitHub由既有定时维护核验后同步。配置详情分别保留结果证据截止与发布核验时间。',
        'MMTM适配A的A/A+LOOK已完成；外部EyeMoSt+/EDRL等候选仍需左侧另行锁定适配范围。融合位置机制包已完成，不据此继续堆融合位置/骨干/种子。']
    if has_external:
        lines=[line.replace('当前累计表同时覆盖跨骨干、MMTM适配A及融合位置机制，但仍不能替代尚未完成的外部A/B/C全套匹配。','前两旧配置是跨骨干参考；MMTM配置回答一个适配A与A+LOOK；fusion extension补充R18融合位置机制。三类问题分开解释，不把MMTM冒充完整A/B/C。') for line in lines]
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
            if statuses.get(rid,{}).get('state')=='needs_review':
                if plan.get('study_kind')!='fusion_stage_v1':
                    continue
                from look.studies.cohort_fusion_stage import validate_repair_resume
                marker=validate_repair_resume(plan,row,s,runroot)
                if marker is None:
                    continue
                incidents=root/'incidents';incidents.mkdir(exist_ok=True)
                atomic_write_json(dict(run_id=rid,previous=statuses[rid],repair_marker=marker,time=time.time()),
                    incidents/f'{rid}_before_repair_{time.time_ns()}.json')
                statuses[rid]=dict(state='repair_resuming',started_at=time.time(),
                    repair_packet_sha256=marker['repair_packet_sha256'])
                save()
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
