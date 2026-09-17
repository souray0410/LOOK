"""Complete one accepted configuration before the serial queue advances."""
import argparse
from pathlib import Path
import numpy as np
from look.runtime.state import atomic_write_json, file_sha256
from look.studies.project_case import read
from look.studies.search_protocol import PATTERNS
from look.analysis.observed_report import simultaneous_bootstrap


def validate_predictions(search, host):
    for key in ('participant_ids', 'labels'):
        if not np.array_equal(search[key], host[key]):
            raise ValueError('Unpaired report participants or labels')
    if len(np.unique(search['participant_ids'])) != len(search['participant_ids']):
        raise ValueError('Duplicate participant in delivery')


def report(root):
    from look.studies.search_case import verify_case, check_files
    root=Path(root);spec=read(root/'spec.json');verify_case(root,spec)
    out=root/'delivery';out.mkdir(exist_ok=True)
    digest=file_sha256(root/'accepted.json')
    if (out/'accepted.json').exists():
        r=read(out/'accepted.json')
        if r['input_sha256']!=digest:raise ValueError('Delivery input changed')
        check_files(out,r['files']);return r
    rows=read(root/'development/suite.json')['records'];results=[];diagnostics=[];arrays={}
    for row in rows:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Prediction changed')
        with np.load(row['path'],allow_pickle=False) as f:arrays[row['scenario'],row['method']]={k:f[k] for k in f.files}
        results.append(dict(pattern=row['scenario'],method=row['method'],metrics=row['metrics']))
    reference=None
    predictions=[];weights=[]
    for i,pattern in enumerate(PATTERNS):
        a,b=arrays[pattern,'search'],arrays[pattern,'host'];validate_predictions(a,b)
        if reference is not None:validate_predictions(a,reference)
        reference=a
        predictions.extend([a['logits'].argmax(1),b['logits'].argmax(1)])
        w=np.zeros(2*len(PATTERNS));w[2*i]=1;w[2*i+1]=-1;weights.append(w)
        folder=root/'corrections'/pattern
        diagnostics.append(dict(pattern=pattern,selection=read(folder/'factor_selection.json'),
            traces=[read(p) for p in sorted(folder.glob('factors/*/selection.json'))]))
    stats=simultaneous_bootstrap(reference['labels'],np.stack(predictions),weights,10000,7341618)
    stats.update(patterns=list(PATTERNS),direction='search_minus_host',scope='one_selected_dev_configuration_not_independent_confirmation')
    atomic_write_json(results,out/'results.json');atomic_write_json(stats,out/'paired_statistics.json')
    atomic_write_json(dict(search=diagnostics,costs=read(root/'costs.json')),out/'diagnostics.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(PATTERNS),figsize=(9,4),layout='constrained')
    for ax,pattern in zip(axes,PATTERNS):
        values={r['method']:100*r['metrics']['macro_f1'] for r in results if r['pattern']==pattern}
        ax.bar(['Frozen host','LOOK'],[values['host'],values['search']]);ax.set_ylim(0,100)
        ax.set_title(pattern);ax.set_ylabel('Development macro-F1 (%)')
    fig.savefig(out/'comparison.svg');plt.close(fig)
    text=['# LOOK 单配置全流程交付','',f"配置：{spec['host']}；策略 {spec['mode']}；空间因子 {spec.get('spatial_factors')}；潜在维度 {spec.get('latent_dims')}。",
        '横轴：未修正冻结宿主与LOOK；纵轴：开发集macro-F1百分比，越高越好。两个面板分别表示缺OCT和缺CFP。',
        '误差分析、实际开启位置和候选评价数见diagnostics.json；差值及普通/本配置族同时区间见paired_statistics.json。',
        '首种子、开发集选优后的探索结果；区间未消除选择偏差，不是独立test或多种子复现。',
        '空间因子16指二维特征的高宽各除16；原网络输入未缩小。向量节点不进行空间插值。',
        '先扫描剩余下游位置，选择最大正收益；后续拟合使用此前已接受修正。无正收益保留当前模型并停止。','',
        '![单配置结果](comparison.svg)']
    (out/'README.zh-CN.md').write_text('\n\n'.join(text)+'\n')
    names=['results.json','paired_statistics.json','diagnostics.json','comparison.svg','README.zh-CN.md']
    receipt=dict(state='accepted',test_access=False,input_sha256=digest,files={n:file_sha256(out/n) for n in names})
    atomic_write_json(receipt,out/'accepted.json');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();report(a.run)
