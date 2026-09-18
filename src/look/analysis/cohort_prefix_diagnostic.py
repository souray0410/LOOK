"""Reuse accepted root-prefix evidence to separate fitting from path selection."""
import argparse
import json
from pathlib import Path
from look.runtime.state import atomic_write_json, stable_hash
from look.analysis.cohort_publication import PATTERNS, SITES, backbone_label


def diagnose(source, output):
    source=Path(source); doc=json.loads(source.read_text()); rows=[]
    assert doc['schema']=='look_cohort_sequence_publication_v1' and doc['test_used'] is False
    for config in doc['configurations']:
        assert config['matched_package']=='accepted'
        details={(x['method'],x['scenario']):x for x in config['search_details']}
        for scenario in ('oct_missing','cfp_missing'):
            a,b=(details[(m,scenario)] for m in ('pca_free_mean','residual_rrr'))
            assert a['sites']==b['sites'] and a['penalty_policy']==b['penalty_policy']
            ar=[x for x in a['branches'] if x['prefix']==[]];br=[x for x in b['branches'] if x['prefix']==[]]
            assert len(ar)==len(br)==1 and ar[0]['baseline_macro_f1']==br[0]['baseline_macro_f1']
            aa={r['node']:r for r in ar[0]['candidates']};bb={r['node']:r for r in br[0]['candidates']}
            assert aa.keys()==bb.keys() and len(aa)==len(a['sites'])
            for node in aa:
                p,r=aa[node]['macro_f1'],bb[node]['macro_f1']
                rows.append(dict(configuration_id=config['configuration_id'],backbone=config['architecture'],scenario=scenario,node=node,
                    baseline=ar[0]['baseline_macro_f1'],pca=p,residual=r,residual_minus_pca_pp=100*(r-p)))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    result=dict(schema='look_fixed_empty_prefix_diagnostic_v1',source_sha256=stable_hash(doc),test_used=False,
        scope='same accepted host and empty correction prefix; same node/rank/penalty selection policy; each family retains its own fitted slope and selected lambda',
        new_gpu_fits=0,new_training=0,rows=rows,
        limitations=['not_fixed_lambda_or_slope','not_nonempty_prefix_mechanism','selected_dev_descriptive_no_new_confirmatory_p_values'])
    atomic_write_json(result,out/'current.json')
    lines=['# 同一个位置、还没开始走树：两种拟合有何不同？','',
        '复用已验收的第一轮候选，不新增训练或GPU拟合。固定同一宿主、空修正前缀、位置与秩，比较PCA与自由残差各自拟合后的F1。',
        '这排除了此前修正路径不同的影响，但λ仍按各方法自身的既定规则选取，不能叫“固定λ/斜率”的机制消融。负候选也完整展示；这里是强制启用候选，不是最终关闭开关后的性能。','',
        '|骨干|缺失|位置|不修正|PCA|残差|残差−PCA(pp)|','|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'|{backbone_label(r["backbone"])}|{PATTERNS[r["scenario"]]}|{SITES[r["node"]]}|{100*r["baseline"]:.2f}|{100*r["pca"]:.2f}|{100*r["residual"]:.2f}|{r["residual_minus_pca_pp"]:+.2f}|')
    lines+=['','各单元只描述同一个已看过的dev，不新增确认性显著性结论。原树最终结果仍见[累计入口](../small_cohort/README.md)。']
    (out/'README.md').write_text('\n'.join(lines)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();diagnose(a.source,a.output)
