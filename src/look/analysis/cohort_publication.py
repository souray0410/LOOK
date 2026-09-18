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
    if 'mmtm' in s:
        public['mmtm']=s['mmtm']
        public['limitations'].append('author_gate_module_adaptation_not_original_paper_system')
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
            public['host_training_result']={k:r[k] for k in ('stop_epoch','best_epoch','seconds','plateau') if k in r}
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
    public['search_details']=[]
    for arm in s['arms']:
        for pattern in ('oct_missing','cfp_missing'):
            folder=root/arm/'corrections'/pattern
            selection=folder/'selection.json'
            if not selection.exists():continue
            tree=read(selection);contract=read(folder/'contract.json')
            if file_sha256(folder/'contract.json')!=tree['contract_sha256']:
                raise ValueError('Search contract changed')
            if contract['identity']['identity']!=identity or tree['mode']!=s['search']:
                raise ValueError('Search identity or mode differs')
            chosen=[];candidates={}
            branches=[]
            for decision in tree['decisions']:
                baseline=decision['baseline']['score']
                row=dict(prefix=[n['node'] for n in decision['path']],baseline_macro_f1=baseline,candidates=[])
                for c in decision['candidates']:
                    candidates[(c['node'],c['sha256'])]=c
                    row['candidates'].append(dict(node=c['node'],macro_f1=c['score'],
                        delta_pp=100*(c['score']-baseline),positive=c['score']>baseline))
                branches.append(row)
            for n in tree['selected_path']:
                if file_sha256(folder/n['artifact'])!=n['sha256']:raise ValueError('Selected search artifact changed')
                c=candidates[(n['node'],n['sha256'])]
                chosen.append(dict(node=n['node'],index=n['index'],sha256=n['sha256'],macro_f1=c['score']))
            if (arm,pattern) in unique and tree['final']['metrics']!=unique[(arm,pattern)]['metrics']:
                raise ValueError('Search final differs from accepted prediction')
            public['search_details'].append(dict(method=arm,scenario=pattern,mode=tree['mode'],sites=contract['sites'],
                candidates=contract['identity']['candidates'],penalty_policy=contract['identity']['penalty_policy'],
                selected_path=chosen,candidate_evaluations=tree['candidate_evaluations'],prefix_count=tree['prefix_count'],
                site_attempts=tree['site_attempts'],feature_costs=read(folder/'feature_costs.json'),
                selection_sha256=file_sha256(selection),contract_sha256=file_sha256(folder/'contract.json'),branches=branches))
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

    lines=render(public)
    content='\n'.join(lines)+'\n'
    # All validation/rendering happens before replacing the last valid publication.
    if previous!=public:atomic_write_json(public,out/'current.json')
    target=out/'README.md'
    if not target.exists() or target.read_text()!=content:
        temporary=out/'.README.md.tmp';temporary.write_text(content);temporary.replace(target)
    return public


NAMES={'host':'不修正','pca_free_mean':'PCA方向约束','residual_rrr':'自由低秩残差'}
PATTERNS={'oct_missing':'缺OCT（仅眼底照片）','cfp_missing':'缺眼底照片（仅OCT）'}
SITES={'joint_input':'①双分支输入','joint_stem':'②初始卷积后','joint_stage1':'③Stage 1后',
       'joint_stage2':'④Stage 2后','joint_stage3':'⑤Stage 3后','joint_stage4':'⑥Stage 4后、融合前',
       'fusion_stage4':'⑦融合后特征图','fusion_features':'⑧每眼特征向量','fusion_participant_feature':'⑨双眼汇总后的参与者特征'}


def render(p):
    site_names=dict(SITES)
    if p['architecture']=='densenet121':
        site_names={k.replace('stage','denseblock'):v.replace('Stage ','Dense block ') for k,v in SITES.items()}
    rows={(r['method'],r['scenario']):r['metrics'] for r in p['results']}
    complete=p['matched_package']=='accepted'
    lines=['# LOOK 小队列：两种拟合方法，在同一正收益树规则下比较','',
        '**本页比较的是拟合方法：PCA方向约束与自由低秩残差；两者都允许自由均值，都运行了各自的正收益树搜索。**',
        '阅读顺序：先看结果，再看共同配置和最终修正路径。不是固定单层比较，也不是把两种方法套在同一条已选路径上。','',
        '## 1．先看结果：同一行横向比较','',
        '主指标为Macro-F1（越高越好，不是准确率）。括号内为相对同一缺失状态“不修正”的变化，单位为百分点。','',
        '|输入情况|不修正|PCA方向约束＋树|自由低秩残差＋树|','|---|---:|---:|---:|']
    for pattern,label in PATTERNS.items():
        values=[];base=rows.get(('host',pattern),{}).get('macro_f1')
        for arm in ('host','pca_free_mean','residual_rrr'):
            m=rows.get((arm,pattern))
            values.append('未验收' if not m else f'{100*m["macro_f1"]:.2f}%'+('' if arm=='host' or base is None else f'（{100*(m["macro_f1"]-base):+.2f}）'))
        lines.append('|'+label+'|'+'|'.join(values)+'|')
    if 'host_metrics' in p:
        m=p['host_metrics'];lines+=['',f'完整输入（两种影像都在）的同一原模型：Macro-F1 **{100*m["macro_f1"]:.2f}%**，AUROC **{100*m["macro_auroc_ovr"]:.2f}%**。它只是参考，表中增益均相对对应缺失基线计算。']
    lines+=['','**如何理解：**先在同一缺失状态、同一原模型内比较修正增益，再看两种拟合之差及区间。单种子结果不支持稳定优越性。' if complete else '完整匹配组尚未齐全，暂不判断哪种拟合更好。',
        '同一开发集参与了模型和路径选择；缺OCT的低基线并不代表模型完全没有区分能力，须同时看AUROC，不能只根据F1涨幅解释机制。','',
        '## 2．这次到底用了什么配置','',
        '|项目|本次配置|','|---|---|',
        '|任务与数据|青光眼二分类；旧小队列1,264名训练参与者、296名开发集参与者；test未使用|',
        '|输入|每眼224×224；CFP眼底照片与旧数据导出的OCT二维图像（三通道）；不是R&B的32层三维输入|',
        f'|原网络|两分支{p["architecture"]}，公开ImageNet初始化后重新训练同一共同模型；没有复用历史医学宿主|',
        '|两个分支在哪里融合|各自走完Stage 4后，按通道拼接，经1×1卷积与BN投影，再生成每眼特征、汇总双眼特征、分类|',
        '|拟合时哪些会变|原网络参数及BN固定；只拟合LOOK变换，不反向训练原网络|',
        '|缺失如何模拟|整种模态用标准化空间的零值替代（normalized_mean）；不是删除某个病人|',
        f'|随机种子|{p["seed"]}，目前只有一个种子|',
        f'|x{p["factor"]}是什么意思|对LOOK拟合用的二维特征图，高和宽各做{p["factor"]}倍双线性降采样，最小1×1；原网络输入仍224×224，向量节点不缩空间|',
        f'|q{p["rank"]}是什么意思|修正的秩/子空间维数固定为{p["rank"]}；这次没有搜索其他秩|',
        '|两个方法的共同点|自由均值、同一原模型/数据/空间缩放/秩/评分规则；各自独立搜索修正路径|',
        '|正则强度λ|每个前缀、位置用训练集的PCA投影GCV规则选取；不是λ=0，也不是每种方法各自穷尽最优λ|',
        '|训练与选择分工|训练集拟合统计、PCA和修正；开发集Macro-F1选择原模型及树路径；test封存|']
    t=p['training'];h=p.get('host_training_result',{})
    lines+=['','<details>','<summary>展开：原网络训练配方及两种拟合的区别</summary>','',
        f'原网络：AdamW、FP32、无类别权重交叉熵；每次{t.get("microbatch",16)}名参与者，累积到{t.get("effective_batch",128)}名更新；预训练部分学习率{t.get("pretrained_lr",.0001)}，新层{t.get("new_layer_lr",.001)}；weight decay={t.get("weight_decay",.0001)}，梯度裁剪={t.get("clip",5)}。',
        f'最多{t.get("epochs",100)}轮、至少{t.get("minimum_epochs",8)}轮、连续{t.get("patience",15)}轮开发集无改善停止，warmup={t.get("warmup_epochs",5)}轮。',
        f'本次实际停止轮次：{h.get("stop_epoch","未验收")}；最佳权重轮次：{h.get("best_epoch","未验收")}。早停达到预定规则，不代表数学上证明全局收敛。',
        '',
        'PCA方向约束：先从完整训练特征找主要变化方向，修正的线性部分受该子空间限制。自由低秩残差：仍限制秩，但修正方向由完整与缺失特征之间的残差拟合，不强制落在上述PCA子空间。二者都允许自由截距，因此不是“自由均值与固定均值”的对比。',
        '', '</details>','',
        '## 3．树搜索具体在做什么','',
        '从不修正出发，在9个合法位置分别尝试修正；保留所有让当前路径F1严格提高的扩展。每条保留分支再尝试其后方的位置。候选使用该分支此前修正后的特征；同分关闭、负收益不继续向后扩展。最后在已探索路径中选F1最高的路径；同分优先更少位置。',
        '这里每个位置只有一个固定秩候选，λ由训练规则确定。**保留全部正扩展，不是每轮只保留一个最好位置；也不会探索先降分、后面再补回来的组合。**','',
        '```text',
        'CFP：输入 → 初始卷积 → Stage 1 → Stage 2 → Stage 3 → Stage 4 ┐',
        'OCT：输入 → 初始卷积 → Stage 1 → Stage 2 → Stage 3 → Stage 4 ┘',
        '        ①       ②         ③         ④         ⑤         ⑥',
        '                   ↓ 两分支拼接＋投影（原网络的融合位置）',
        '              ⑦融合后特征图 → ⑧每眼特征 → ⑨双眼汇总 → 分类',
        '```','',
        '①—⑥是LOOK同时读取两分支对应特征、拼接后拟合修正、再拆回两分支的位置；它们不是原网络已经完成融合。因而“原网络深层融合”不等于“LOOK只能从深层开始”。','',
        '## 4．最终选中了哪些位置，实际搜索了多少','',
        '|缺失状态|拟合方法|最终启用路径（按前向顺序）|候选评价数|已探索前缀数|',
        '|---|---|---|---:|---:|']
    for pattern,label in PATTERNS.items():
        for arm in ('pca_free_mean','residual_rrr'):
            d=next((d for d in p.get('search_details',[]) if d['method']==arm and d['scenario']==pattern),None)
            if not d:lines.append(f'|{label}|{NAMES[arm]}|尚无核验路径|—|—|');continue
            path=' → '.join(site_names.get(n['node'],n['node']) for n in d['selected_path']) or '不启用修正'
            lines.append(f'|{label}|{NAMES[arm]}|{path}|{d["candidate_evaluations"]}|{d["prefix_count"]}|')
    lines+=['','“前缀”就是一条已接受的部分修正路径，计数包含空路径和终端路径；候选评价数不是训练轮数。最终只留一个位置，不代表只测了一个位置。两方法搜索规则相同，但正收益分支不同，因此工作量不同。','',
        '<details>','<summary>展开：第一轮独立修正9个位置，各自提升多少</summary>','',
        '以下是从“不修正”出发、每次只开启这一个位置的F1变化（百分点），不是最终多位置组合结果。正值进入下一层树，零或负值不扩展。','',
        '|位置|缺OCT／PCA|缺OCT／低秩残差|缺眼底照片／PCA|缺眼底照片／低秩残差|',
        '|---|---:|---:|---:|---:|']
    roots={}
    for d in p.get('search_details',[]):
        root=next((b for b in d['branches'] if not b['prefix']),None)
        if root:roots[(d['scenario'],d['method'])]={c['node']:c['delta_pp'] for c in root['candidates']}
    for node,label in site_names.items():
        vals=[]
        for pattern in PATTERNS:
            for arm in ('pca_free_mean','residual_rrr'):
                value=roots.get((pattern,arm),{}).get(node);vals.append('未记录' if value is None else f'{value:+.3f}')
        lines.append('|'+label+'|'+'|'.join(vals)+'|')
    lines+=['','</details>','',
        '<details>','<summary>展开：每条最终路径逐步得到的F1，以及缓存/计算记录</summary>','']
    for d in p.get('search_details',[]):
        baseline=rows.get(('host',d['scenario']),{}).get('macro_f1',0)
        curve=f'不修正 {100*baseline:.2f}%'
        for n in d['selected_path']:curve+=f' → {site_names.get(n["node"],n["node"])} {100*n["macro_f1"]:.2f}%'
        lines +=[f'- **{PATTERNS[d["scenario"]]}／{NAMES[d["method"]]}：** {curve}。']
    lines+=['','所有分支和首轮9位置的分数、保留/拒绝依据见[current.json](current.json)的`search_details[].branches`，不只保留最终胜出路径。','',
        '|方法／缺失状态|完整输入特征前向批次|缺失输入特征前向批次|完整统计缓存命中|恢复跳过批次|',
        '|---|---:|---:|---:|---:|']
    for d in p.get('search_details',[]):
        c=d['feature_costs'];lines.append(f'|{NAMES[d["method"]]}／{PATTERNS[d["scenario"]]}|{c["full_forwards"]}|{c["missing_forwards"]}|{c["completed_hits"]}|{c["skipped_batches"]}|')
    lines+=['','缓存边界：共同原模型和完整训练PCA基复用；同一前缀一次扫描收集多个下游位置的统计，已提交统计可恢复。上表记录这次首次运行的实际命中，不把“有缓存机制”写成“每次都命中”。本次完整统计命中为0，完整侧仍有重复前向；不能声称跨前缀完整特征已经全部缓存或这是最优速度。上述计数不包含全部开发集评价，不能直接换算总耗时。','',
        '</details>','',
        '## 5．其他指标和不确定性','',
        '下面仍按同一缺失状态比较，AUROC越高越好；NLL和Brier越低越好。','',
        '|缺失状态／指标|不修正|PCA方向约束＋树|自由低秩残差＋树|','|---|---:|---:|---:|']
    for pattern,label in PATTERNS.items():
        for metric,name,mult in [('macro_auroc_ovr','AUROC (%)',100),('negative_log_likelihood','NLL',1),('multiclass_brier','Brier',1)]:
            vals=[f'{mult*rows[(arm,pattern)][metric]:.3f}' if metric in rows.get((arm,pattern),{}) else '未验收' for arm in ('host','pca_free_mean','residual_rrr')]
            lines.append('|'+label+'／'+name+'|'+'|'.join(vals)+'|')
    if complete:
        lines+=['','两方法差值定义为**自由低秩残差−PCA**，单位为Macro-F1百分点。','',
            '|缺失状态|差值|普通95%区间|六项比较同时95%区间|','|---|---:|---|---|']
        m=p['matched_results']
        for c,v in zip(m['comparisons'],m['statistics']['contrasts']):
            if c['reference']!='pca_free_mean':continue
            interval=lambda k:'['+', '.join(f'{100*x:.3f}' for x in v[k])+']'
            lines.append(f'|{PATTERNS[c["pattern"]]}|{100*v["difference"]:+.3f}|{interval("ordinary_95")}|{interval("simultaneous_95")}|')
        lines+=['','区间跨零时不能确定胜者，也不能据此认定等效。请同时检查上表AUROC、NLL及Brier；F1提高不等于所有指标变好。']
    lines+=['','统计使用296人的10,000次参与者级配对bootstrap；仅一个种子。开发集同时用于原模型和路径选择，区间不消除选择偏差；不与Ibex不同疾病/队列混合排名。','',
        '## 6．完成范围、下一步与来源','',
        '本配置的共同模型、两种拟合、两种缺失、重放与匹配报告已完成；不是整个周研究包或独立test验证完成。' if complete else '当前配置仍有未验收步骤；已有局部结果不冒充完整比较。',
        '下一步优先准备具体外部方法与加LOOK后的匹配验证；本页不声称这个后继已经开跑。',
        f'结果／阶段证据截止（UTC）：{p["evidence_cutoff_utc"]}；本版发布核验（UTC）：{p["publication_verified_at_utc"]}。',
        '服务器累计生成，再由定时维护核验并同步GitHub；不是实时仪表盘。相同证据重复发布不刷新时间。',
        f'运行：`{p["run_id"]}`；科学源：`{p["source_commit"]}`；框架：`{p["framework_commit"]}`。',
        f'配置指纹：`{p["configuration_id"]}`。',
        '完整安全汇总和路径在[current.json](current.json)。权重、参与者预测及特征保留在授权服务器，不上传GitHub。']
    if 'mmtm' in p:
        v=p['mmtm']
        lines[2:2]=['',f'**本配置的A是MMTM适配宿主**：两路Stage 3后加入作者门控（ratio={v["ratio"]}，scale={v["gate_scale"]}、随机Linear初始化），再进入Stage 4和原深融合分类头。上图需在Stage 3后加入双向门控；它不是原论文视频/骨骼完整系统。', '不修正列就是A；两拟合列是冻结同一A后加LOOK，比较不另训一套A。','']
    if p['architecture']=='densenet121':
        lines=[line.replace('Stage 4','Dense block 4').replace('Stage 1','Dense block 1').replace('Stage 2','Dense block 2').replace('Stage 3','Dense block 3') for line in lines]
    return lines

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();publish(a.run,a.output)
