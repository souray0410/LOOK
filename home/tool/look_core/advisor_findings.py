"""Data-derived findings, including unfavorable metrics, for advisor snapshots."""
from collections import Counter
from pathlib import Path
import numpy as np
from .state import atomic_write_text,atomic_write_json


def main_pairs(rows):
    index={(r['fusion'],r['filling'],r['seed'],r['scenario'],r['metric']):r for r in rows if r['method']=='filling'}
    result=[]
    for r in rows:
        key=tuple(r[k] for k in ('fusion','filling','seed','scenario','metric'))
        if r['method']=='original_LOOK' and key in index:
            result.append(dict(r,baseline=index[key]['value'],delta=r['value']-index[key]['value']))
    return result


def finding_sections(rows,agg):
    pairs=main_pairs(rows);fixed=[r for r in pairs if r['scenario'] in ('oct_missing','cfp_missing')]
    f1=[r for r in fixed if r['metric']=='macro_f1'];random=[r for r in pairs if r['metric']=='macro_f1' and r['scenario'].startswith('random_')]
    count=lambda rr: (sum(r['delta']>0 for r in rr),sum(r['delta']==0 for r in rr),sum(r['delta']<0 for r in rr))
    n=len({(r['fusion'],r['filling'],r['seed']) for r in f1})
    if not n:return [('Findings awaiting verified results',['No completed main LOOK case is available. No performance conclusion is drawn.'])]
    positive,tied,negative=count(f1);rp,rt,rn=count(random)
    summary=[f'{n}/27 main LOOK cases complete. Fixed missing directions: F1 improves in {positive}/{len(f1)}, ties in {tied}, decreases in {negative}. Random-ratio comparisons: {rp} improve, {rt} tie, {rn} decrease.',
        'These are correlated development comparisons on the same validation participants, selected for Macro-F1. They are not independent repetitions or evidence of test generalization.',
        'Complete-model fusion screening, missing-input robustness and correction benefit are different questions. A strong complete-model score does not establish the best missing-modality deployment.',
        'Mechanism comparisons remain pending until their own cases finish. Node activation counts and sequential gains alone do not establish causal necessity.']
    metric=[]
    for name,label,lower in [('macro_auroc_ovr','AUROC',False),('macro_auprc_ovr','AUPRC',False),('ece_15','ECE',True),('multiclass_brier','Brier',True),('negative_log_likelihood','NLL',True)]:
        rr=[r for r in fixed if r['metric']==name];p,t,m=count(rr)
        metric.append(f'{label}: {m if lower else p} improve, {t} tie, {p if lower else m} worsen, out of {len(rr)} fixed-direction comparisons.')
    nll=[r for r in fixed if r['metric']=='negative_log_likelihood']
    worst=max(nll,key=lambda r:r['delta']) if nll else None
    if worst:metric.append(f"Largest NLL increase: {worst['fusion']} / {worst['filling']} / seed {worst['seed']} / {worst['scenario']}: {worst['baseline']:.3f} -> {worst['value']:.3f}. Inspect selected corrections and logit magnitude; the cause is not established.")
    # Split quantitative comparisons into readable pages without omitting negative values.
    table=[]
    base={(r['fusion'],r['filling'],r['scenario']):r for r in agg if r['method']=='filling' and r['metric']=='macro_f1'}
    for r in agg:
        key=(r['fusion'],r['filling'],r['scenario'])
        if r['method']=='original_LOOK' and r['metric']=='macro_f1' and r['scenario'] in ('oct_missing','cfp_missing') and key in base:
            b=base[key];table.append(f"{r['fusion']} | {r['filling']} | {r['scenario']}: {b['mean']:.4f} -> {r['mean']:.4f} (LOOK SD {r['sd']:.4f}; delta {100*(r['mean']-b['mean']):+.2f} percentage points).")
    sections=[('What the completed results support',summary),('Metric tradeoffs and the open diagnostic',metric)]
    sections.extend((f'Three-seed comparisons ({i//5+1})',table[i:i+5]) for i in range(0,len(table),5))
    return sections


def write_speaker_notes(rows,agg,manifest,destination):
    pairs=main_pairs(rows)
    atomic_write_json(pairs,Path(destination)/'main_paired_effects.json')
    lines=['# LOOK 导师汇报讲稿',f"生成时间（UTC）：{manifest['generated_at_utc']}",
        f"完成状态：主流程 {manifest['parent_completed']}/52；起点组合 {manifest['suffix_completed']}/243；机制对照已完成 {manifest['methods_completed']}。尚未完成的实验不作结论。",
        '## 研究问题与方法',
        '目标是在冻结完整多模态模型的前提下，利用训练集完整/缺失特征配对拟合逐层校正，改善缺失模态时的任务性能。PCA 与 W/b 使用 train；validation 选择维度、空间因子、启用节点和起点；test 保持封存。',
        '融合位置决定网络在哪里融合，校正起点决定从哪里开始允许校正，两者独立。起点不是强制启用位置；只有严格改善验证 Macro-F1 才启用。',
        '## 当前能说与不能说',
        '目前是验证集上的开发证据。逐种子改善不是独立重复试验；三种子标准差不是参与者层面的置信区间；起点搜索包含原方法，验证最优分数不下降是集合包含关系。',
        'F1 改善不保证 AUROC、AUPRC、校准或 NLL 改善。下面按原始预测复算后的配对结果完整汇报：']
    for pattern in ('oct_missing','cfp_missing'):
        for metric,lower in [('macro_f1',False),('macro_auroc_ovr',False),('negative_log_likelihood',True)]:
            rr=[r for r in pairs if r['scenario']==pattern and r['metric']==metric]
            improve=sum(r['delta']<0 if lower else r['delta']>0 for r in rr)
            worsen=sum(r['delta']>0 if lower else r['delta']<0 for r in rr)
            lines.append(f'- {pattern}，{metric}：{len(rr)} 个已完成配对，{improve} 个改善、{worsen} 个变差。')
    lines+=['## 机制验证与下一步',
        '已有起点与单点消融检验位置；independent_fit 检验上游条件拟合；missing_only 检验是否需要校正保留分支；SSF 是相近的冻结网络适配对照。',
        '新增 bias_only 检验是否只需均值平移；logit_affine 检验是否只需输出边界与尺度调整。均固定 layer3 / normalized_mean / 三种子，并保留监督与搜索成本差异。',
        '当前概率异常需要追踪实际 ON 节点的放大量，不能用未启用候选的矩阵范数解释结果，也不能只挑好看的 F1。',
        '独立测试前冻结配置、主要比较、参与者配对不确定性分析和报告范围。核实两个 test cohort 的参与者重叠，不把重叠样本视为两次独立验证。',
        '当前只验证了同一 ResNet50 家族的融合位置，不据此宣称跨架构普适性或小设备部署优势。',
        '## 汇报材料索引',
        'advisor_report.pdf 为英文展示版；three_seed_summary.csv 为完整三种子均值与样本标准差；all_metrics.csv 与 report_manifest.json 提供预测和来源哈希；main_paired_effects.json 保留所有主方法配对差值。']
    atomic_write_text('\n\n'.join(lines)+'\n',Path(destination)/'导师汇报讲稿.md')


def tradeoff_figure(rows,destination):
    import matplotlib.pyplot as plt
    pairs=[r for r in main_pairs(rows) if r['scenario'] in ('oct_missing','cfp_missing')]
    index={tuple(r[k] for k in ('fusion','filling','seed','scenario','metric')):r for r in pairs}
    if not pairs:return None
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors={'normalized_mean':'#237C8B','raw_zero':'#C38A39','paired_cgan':'#9C6388'}
    for filling,color in colors.items():
        xs=[];ys=[];ns=[]
        for r in pairs:
            if r['metric']!='macro_f1' or r['filling']!=filling:continue
            key=tuple(r[k] for k in ('fusion','filling','seed','scenario'))
            auc=index.get((*key,'macro_auroc_ovr'));nll=index.get((*key,'negative_log_likelihood'))
            if not auc or not nll:continue
            xs.append(100*r['delta']);ys.append(100*auc['delta']);ns.append(np.log10(max(nll['value'],1e-12)/max(nll['baseline'],1e-12)))
        axes[0].scatter(xs,ys,s=30,color=color,label=filling,alpha=.75)
        axes[1].scatter(xs,ns,s=30,color=color,label=filling,alpha=.75)
    for ax in axes:
        ax.axhline(0,color='#777777',lw=.8);ax.axvline(0,color='#777777',lw=.8)
        ax.set_xlabel('Macro-F1 change (percentage points)');ax.legend(fontsize=8)
    axes[0].set(ylabel='AUROC change (percentage points)',title='Upper right: classification and ranking improve')
    axes[1].set(ylabel='log10(NLL after / NLL before)',title='Above zero: probability loss worsens')
    fig.suptitle('Every completed fixed-direction case | development validation | negative outcomes retained',fontsize=12)
    path=Path(destination)/'metric_tradeoffs.png'
    for ext in ('png','svg'):fig.savefig(path.with_suffix('.'+ext),dpi=160)
    plt.close(fig);return path


def workflow_figure(destination):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(12,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    blocks=[('TRAIN','Complete backbone\nthen freeze weights','#DDEBF0'),
            ('TRAIN','Complete PCA + paired W/b\ninherit accepted upstream','#DDEBF0'),
            ('VALIDATION','Select dimensions, factors,\nON/OFF and allowed start','#F4E8D5'),
            ('SEALED TEST','Apply frozen configurations\nno further selection','#E6E8EC')]
    for i,(tag,label,color) in enumerate(blocks):
        x=.025+i*.25
        ax.add_patch(FancyBboxPatch((x,.63),.21,.21,boxstyle='round,pad=0.01',facecolor=color,edgecolor='none'))
        ax.text(x+.105,.795,tag,ha='center',va='center',fontsize=10,weight='bold',color='#173C4C')
        ax.text(x+.105,.71,label,ha='center',va='center',fontsize=9,color='#263A47')
        if i<3:ax.annotate('',xy=(x+.24,.735),xytext=(x+.215,.735),arrowprops=dict(arrowstyle='->',color='#697381'))
    ax.text(.02,.48,'EXAMPLE: LAYER3 FUSION | 9 ORDERED CANDIDATE CORRECTION SITES',fontsize=10,weight='bold',color='#173C4C')
    nodes=['joint\ninput','joint\nstem','joint\nlayer1','joint\nlayer2','joint\nlayer3','fusion\nlayer3','fusion\nlayer4','fusion\nfeature','fusion\nparticipant\nfeature']
    for i,name in enumerate(nodes):
        x=.025+i*.11;color='#EEDDC2' if i==5 else '#E3EDF0'
        ax.add_patch(FancyBboxPatch((x,.24),.085,.15,boxstyle='round,pad=0.005',facecolor=color,edgecolor='none'))
        ax.text(x+.0425,.315,name,ha='center',va='center',fontsize=8,color='#173C4C')
        if i<8:ax.annotate('',xy=(x+.103,.315),xytext=(x+.088,.315),arrowprops=dict(arrowstyle='->',lw=.8,color='#697381'))
    ax.text(.02,.135,'Fusion location and correction start are distinct. A candidate site is enabled only for strict validation Macro-F1 improvement.',fontsize=9,color='#263A47')
    ax.text(.02,.065,'Missingness: each affected participant loses ONE modality in both eyes; direction is fixed across nested 20-100% ratios.',fontsize=9,color='#263A47')
    fig.tight_layout(pad=.8);path=Path(destination)/'method_overview.png'
    for ext in ('png','svg'):fig.savefig(path.with_suffix('.'+ext),dpi=160)
    plt.close(fig);return path
