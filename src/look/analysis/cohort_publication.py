"""Publish an allowlisted aggregate view; never export predictions or run bindings."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from look.runtime.state import atomic_write_json, file_sha256, stable_hash


def read(p):return json.loads(Path(p).read_text())


def publish(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    s=read(root/'spec.json');identity=stable_hash(s)
    public=dict(schema='look_cohort_publication_v1',run_id=s['run_id'],configuration_id=identity,
        source_commit=s['source_commit'],framework_commit=s['framework_commit'],
        data=dict(name='ukb_small_20260909_v1',disease='glaucoma',train=1264,development=296,test_used=False),
        architecture=s['architecture'],position=s['position'],seed=s['seed'],factor=s['factor'],rank=s['rank'],
        search=s['search'],arms=s['arms'],training=s['training'],
        initialization={k:v for k,v in s['initialization'].items() if k!='path'},
        states={},results=[],limitations=['single_seed_development_selection',
          'public_initialization_fresh_host_not_Ibex_medical_parent_composition',
          'small_balanced_cohort_not_large_cohort_replacement'])
    for stage in ('profile','host','pca',*s['arms']):
        p=root/(stage+'_status.json');v=read(p) if p.exists() else {'state':'not_started'}
        public['states'][stage]={k:v[k] for k in ('state','time','gpu_peak_reserved_bytes') if k in v}
    p=root/'host/status.json'
    if p.exists():public['host_progress']={k:v for k,v in read(p).items() if k in ('epoch','updates','offset','state','updated_at')}
    assets=[]
    for role,path in [('host',root/'host'),*((arm,root/arm) for arm in s['arms'])]:
        p=path/'accepted.json'
        if not p.exists():continue
        r=read(p)
        if r['state']!='accepted' or r['identity']!=identity:raise ValueError('Unaccepted or mismatched source')
        if role=='host':
            public['host_metrics']=r['metrics']
            for name,sha in r['files'].items():
                if file_sha256(path/name)!=sha:raise ValueError('Host archive changed')
            for name in ('best.pt','last.pt'):
                assets.append(dict(role='selected_host' if name=='best.pt' else 'full_resume',
                    artifact_ref=f'LOOK/{s["run_id"]}/host/{name}',sha256=r['files'][name],
                    access='restricted',availability='available',github_payload=False))
        else:
            for row in r['records']:
                if file_sha256(row['path'])!=row['sha256']:raise ValueError('Prediction changed')
                public['results'].append({k:row[k] for k in ('method','scenario','metrics')})
            for pattern in ('oct_missing','cfp_missing'):
                p=path/'corrections'/pattern/'bank.pt'
                assets.append(dict(role='correction_bank',artifact_ref=f'LOOK/{s["run_id"]}/{role}/{pattern}/bank.pt',
                    sha256=file_sha256(p),access='restricted',availability='available',github_payload=False))
    unique={}
    for row in public['results']:
        key=(row['method'],row['scenario'])
        if key in unique and unique[key]!=row:raise ValueError('Conflicting duplicate result')
        unique[key]=row
    public['results']=list(unique.values())
    public['artifacts']=assets
    delivery=root/'delivery/accepted.json'
    if delivery.exists():
        r=read(delivery)
        if r['identity']!=identity or r['state']!='accepted':raise ValueError('Delivery identity')
        for name,sha in r['files'].items():
            if file_sha256(root/'delivery'/name)!=sha:raise ValueError('Delivery changed')
        if any(not (root/arm/'accepted.json').exists() for arm in s['arms']):
            raise ValueError('Delivery without accepted methods')
        public['matched_results']=read(root/'delivery/results.json')
        rows=public['matched_results']['results']
        if sorted(rows,key=lambda r:(r['method'],r['scenario']))!=sorted(public['results'],key=lambda r:(r['method'],r['scenario'])):
            raise ValueError('Delivery and method results disagree')
        public['matched_package']='accepted'
    else:public['matched_package']='incomplete'
    source_times=[v.get('time',0) for v in public['states'].values()]
    if delivery.exists():source_times.append(delivery.stat().st_mtime)
    public['evidence_cutoff_utc']=datetime.fromtimestamp(max(source_times),timezone.utc).isoformat()
    public['source_evidence_sha256']=stable_hash(public)
    previous=read(out/'current.json') if (out/'current.json').exists() else {}
    public['publication_verified_at_utc']=(previous.get('publication_verified_at_utc')
        if previous.get('source_evidence_sha256')==public['source_evidence_sha256'] else None) or datetime.now(timezone.utc).isoformat()
    public['publication_mode']='scheduled_verified_sync_not_realtime'

    lines=['# LOOK 小队列当前交付','','同一项目、独立小队列；状态与结果按下表分别验收。',
        f'运行 {s["run_id"]}；青光眼1264 train / 296 dev；ResNet50深层融合；3416；x16/q32。',
        '当前用途：先验证两种自由均值拟合方法；不是Ibex大队列或独立test结论。','',
        f'结果／阶段证据截止（UTC）：{public["evidence_cutoff_utc"]}。',
        f'本版发布核验（UTC）：{public["publication_verified_at_utc"]}。',
        '服务器更新后，由定时维护核验并同步GitHub；本页不是实时状态。没有新证据时不刷新上述时间。','',
        '|阶段|实际状态|','|---|---|']
    stage_names={'profile':'资源与恢复预检','host':'共同宿主训练','pca':'完整训练集PCA基','residual_rrr':'自由低秩残差','pca_free_mean':'PCA方向约束、自由均值'}
    states={'running':'执行中','completed':'该步骤已完成','not_started':'尚未开始','needs_review':'失败待修复'}
    for stage,v in public['states'].items():lines.append(f'|{stage_names[stage]}|{states.get(v["state"],v["state"])}|')
    if 'host_progress' in public:
        h=public['host_progress'];lines.append(f'\n原模型累计完成{h.get("updates",0)}次参数更新；这是训练进度，不是修正收益。')
    if 'host_metrics' in public:
        m=public['host_metrics'];lines.append(f'\n完整输入共同宿主：Macro-F1 {100*m["macro_f1"]:.3f}%，AUROC {100*m["macro_auroc_ovr"]:.3f}%；已通过重放。这不是缺失矫正增益。')
    lines+=['','## 已验收结果','']
    if not public['results']:lines.append('尚无已验收的修正方法结果，不把宿主训练中的分数作为方法增益。')
    else:
        lines+=['|方法|状态|Macro-F1 (%)|AUROC (%)|','|---|---|---:|---:|']
        seen=set()
        for row in public['results']:
            key=(row['method'],row['scenario'])
            if key in seen:continue
            seen.add(key);m=row['metrics']
            names={'host':'不修正','residual_rrr':'自由低秩残差','pca_free_mean':'PCA方向约束（自由均值）','oct_missing':'缺OCT','cfp_missing':'缺眼底照片'}
            lines.append(f'|{names[key[0]]}|{names[key[1]]}|{100*m["macro_f1"]:.3f}|{100*m["macro_auroc_ovr"]:.3f}|')
    lines+=['','## 下一步与阻塞','',
        '本配置计算与匹配报告已完成。下一步优先准备具体外部方法与加LOOK后的匹配验证；该后继尚未由本回执证明已运行。' if public['matched_package']=='accepted' else '继续完成尚未验收步骤，再生成两种方法的匹配报告；尚无完整比较不代表方法无效。',
        '本页完成状态只覆盖当前配置。历史故障及修复见[执行记录](../../../handoff/ws02_cohort_delivery_20260918.md)；其他队列状态见[项目累计入口](../README.md)。']
    lines+=['','权重、恢复断点、参与者预测不上传；current.json中保存验收后模型与修正模块的逻辑索引及SHA。',
        '两个完整方法均接受后才形成匹配结论；dev选择及小样本限制保留，失败历程见交接文档。']
    if public['matched_package']=='accepted':
        lines+=['','## 两种拟合的匹配比较','',
            '本配置的两个方法、两种缺失状态与配对统计均已完成；不等于整个周研究包完成。',
            '下表是自由低秩残差减去PCA方法的Macro-F1差值，单位为百分点。','',
            '|缺失状态|差值|普通95%区间|六项比较同时95%区间|','|---|---:|---|---|']
        m=public['matched_results']
        for c,v in zip(m['comparisons'],m['statistics']['contrasts']):
            if c['reference']!='pca_free_mean':continue
            interval=lambda k:'['+', '.join(f'{100*x:.3f}' for x in v[k])+']'
            label='缺OCT' if c['pattern']=='oct_missing' else '缺眼底照片'
            lines.append(f'|{label}|{100*v["difference"]:+.3f}|{interval("ordinary_95")}|{interval("simultaneous_95")}|')
        lines+=['','区间基于296名参与者的10,000次配对重采样；同一开发集也用于选择路径，区间不消除选择偏差，不能据单种子的小差异认定一种拟合普遍更好。']
    content='\n'.join(lines)+'\n'
    # All validation/rendering happens before replacing the last valid publication.
    if previous!=public:atomic_write_json(public,out/'current.json')
    target=out/'README.md'
    if not target.exists() or target.read_text()!=content:
        temporary=out/'.README.md.tmp';temporary.write_text(content);temporary.replace(target)
    return public

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();publish(a.run,a.output)
