"""Publish an allowlisted aggregate view; never export predictions or run bindings."""
import argparse
import json
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
    public['artifacts']=assets
    delivery=root/'delivery/accepted.json'
    if delivery.exists():
        r=read(delivery)
        if r['identity']!=identity or r['state']!='accepted':raise ValueError('Delivery identity')
        for name,sha in r['files'].items():
            if file_sha256(root/'delivery'/name)!=sha:raise ValueError('Delivery changed')
        public['matched_results']=read(root/'delivery/results.json')
        public['matched_package']='accepted'
    else:public['matched_package']='incomplete'
    atomic_write_json(public,out/'current.json')
    lines=['# LOOK 小队列当前交付','','同一项目、独立小队列；状态与结果按下表分别验收。',
        f'运行 {s["run_id"]}；青光眼1264 train / 296 dev；ResNet50深层融合；3416；x16/q32。',
        '当前用途：先验证两种自由均值拟合方法；不是Ibex大队列或独立test结论。','',
        '|阶段|实际状态|','|---|---|']
    stage_names={'profile':'资源与恢复预检','host':'共同宿主训练','pca':'完整训练集PCA基','residual_rrr':'自由低秩残差','pca_free_mean':'PCA方向约束、自由均值'}
    states={'running':'执行中','completed':'该步骤已完成','not_started':'尚未开始','needs_review':'失败待修复'}
    for stage,v in public['states'].items():lines.append(f'|{stage_names[stage]}|{states.get(v["state"],v["state"])}|')
    if 'host_progress' in public:lines.append('\n宿主进度（不是方法结果）：'+json.dumps(public['host_progress'],ensure_ascii=False))
    lines+=['','## 已验收结果','']
    if not public['results']:lines.append('尚无已验收的修正方法结果，不把宿主训练中的分数作为方法增益。')
    else:
        lines+=['|方法|状态|Macro-F1 (%)|AUROC (%)|','|---|---|---:|---:|']
        seen=set()
        for row in public['results']:
            key=(row['method'],row['scenario'])
            if key in seen:continue
            seen.add(key);m=row['metrics']
            lines.append(f'|{key[0]}|{key[1]}|{100*m["macro_f1"]:.3f}|{100*m["macro_auroc_ovr"]:.3f}|')
    lines+=['','权重、恢复断点、参与者预测不上传；current.json中保存验收后模型与修正模块的逻辑索引及SHA。',
        '两个完整方法均接受后才形成匹配结论；dev选择及小样本限制保留，失败历程见交接文档。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return public

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();publish(a.run,a.output)
