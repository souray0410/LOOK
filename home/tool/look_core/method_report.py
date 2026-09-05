"""Traceable advisor report. Only complete, hash-checked validation results enter plots."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .state import atomic_write_json, atomic_write_text, utc_now
from .start_study import read_json, file_record, verify_file, source_result
from .start_report import write_csv
from .stable_metrics import logit_metrics

METRICS = ['macro_f1','macro_auroc_ovr','macro_auprc_ovr','ece_15','multiclass_brier','negative_log_likelihood']
SEEDS = [3407,3408,3409]


def aggregate(rows):
    groups={}
    for r in rows:
        key=tuple(r[k] for k in ('fusion','filling','method','scenario','metric'))
        groups.setdefault(key,[]).append(r)
    out=[]
    for key, rr in sorted(groups.items()):
        if sorted(r['seed'] for r in rr)!=SEEDS:continue
        out.append(dict(zip(('fusion','filling','method','scenario','metric'),key),n_seeds=3,
            mean=float(np.mean([r['value'] for r in rr])),sd=float(np.std([r['value'] for r in rr],ddof=1))))
    return out


def collect(summary):
    parent=read_json(Path(summary['parent_summary']));suffix=read_json(Path(summary['suffix_summary']))
    rows=[];sources=[]
    def add(path,fusion,filling,seed,method,translate=False):
        rec=file_record(Path(path));result=read_json(Path(path));verify_file(rec)
        if result.get('status')!='complete':return
        if result.get('test_access',False) or result.get('test'):raise ValueError('Report accepts sealed validation only')
        sources.append(rec)
        for scenario,values in result['validation'].items():
            original_scenario=scenario
            actual_method=method
            if translate:
                if scenario=='complete':actual_method='complete'
                elif scenario.startswith('fill_'):actual_method='filling';scenario=scenario[5:]
                elif scenario.startswith('look_after_fill_'):actual_method='original_LOOK';scenario=scenario[16:]
                else:continue
            elif scenario.startswith('look_after_fill_'):
                scenario=scenario[16:]
            elif method in ('terminal_only','selected_start_LOOK'):
                if scenario!='complete':continue
            prediction_path=Path(path).parent/'predictions'/f'validation__{original_scenario}.npz'
            prediction_record=file_record(prediction_path)
            with np.load(prediction_path,allow_pickle=False) as bundle:
                recomputed=logit_metrics(bundle['labels'],bundle['logits'])
            verify_file(prediction_record)
            for metric in METRICS:
                if metric not in values:continue
                if not np.isclose(values[metric],recomputed[metric],rtol=0,atol=1e-12,equal_nan=False):
                    raise ValueError(f'Report metric does not reproduce: {path}, {original_scenario}, {metric}')
                rows.append(dict(fusion=fusion,filling=filling,seed=seed,method=actual_method,
                    scenario=scenario,metric=metric,value=values[metric],source_path=rec['path'],source_sha256=rec['sha256'],
                    prediction_path=prediction_record['path'],prediction_sha256=prediction_record['sha256']))
    # Read exact completed main-case references; never glob provisional experiments.
    for filling in ('normalized_mean','raw_zero','paired_cgan'):
        for fusion in ('layer3','feature','layer2'):
            for seed in SEEDS:
                stage=f'{filling}_{fusion}_{seed}'
                if stage in parent.get('completed_stages',{}):
                    add(source_result(parent,Path(summary['parent_summary']).parents[2],stage),fusion,filling,seed,'original_LOOK',True)
    for r in suffix.get('completed_cases',{}).values():
        if r['start_ordinal']==9:
            verify_file(r['result']);add(r['result']['path'],r['fusion_position'],r['filling'],r['seed'],'terminal_only')
    for context,rec in suffix.get('selected_contexts',{}).items():
        original=next(r for r in suffix['completed_cases'].values() if r['context']==context)
        verify_file(rec);add(rec['path'],original['fusion_position'],original['filling'],original['seed'],'selected_start_LOOK')
    for rec in summary.get('completed_cases',{}).values():
        verify_file(rec);r=read_json(Path(rec['path']));c=r['case']
        add(rec['path'],c['fusion_position'],c['filling'],c['seed'],c['method'])
    return rows,sources,parent,suffix


def paired_comparisons(rows):
    index={(r['fusion'],r['filling'],r['seed'],r['method'],r['scenario'],r['metric']):r for r in rows}
    out=[]
    for r in rows:
        if r['method'] in ('complete','filling','original_LOOK'):continue
        key=(r['fusion'],r['filling'],r['seed'],'original_LOOK',r['scenario'],r['metric'])
        if key not in index:continue
        b=index[key]
        out.append(dict(r,original_value=b['value'],delta=r['value']-b['value'],
            original_source=b['source_path'],original_sha256=b['source_sha256']))
    return out


def write_method_report(summary,destination,*,pdf=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rows,sources,parent,suffix=collect(summary);agg=aggregate(rows);comparisons=paired_comparisons(rows)
    write_csv(rows,destination/'all_metrics.csv');write_csv(agg,destination/'three_seed_summary.csv')
    write_csv(comparisons,destination/'paired_method_differences.csv')
    diagnostics=[];costs=[]
    records=list(summary.get('reference_analyses',{}).values())
    for rec in summary.get('completed_cases',{}).values():
        verify_file(rec);records.append(read_json(Path(rec['path'])))
    for r in records:
        method=r.get('label',r.get('case',{}).get('method'));seed=r.get('seed',r.get('case',{}).get('seed'))
        for pattern,d in r['diagnostics'].items():diagnostics.append(dict(method=method,seed=seed,pattern=pattern,**d))
        costs.append(dict(method=method,seed=seed,costs=r['costs']))
    atomic_write_json(diagnostics,destination/'representation_diagnostics.json')
    atomic_write_json(costs,destination/'costs.json')
    atomic_write_json(parent,destination/'parent_summary_snapshot.json')
    atomic_write_json(suffix,destination/'suffix_summary_snapshot.json')
    audit=read_json(Path(sources[0]['path'])).get('data_audit',{}) if sources else {}
    manifest=dict(generated_at_utc=utc_now(),status=summary['status'],parent_completed=len(parent['completed_stages']),
        suffix_completed=len(suffix.get('completed_cases',{})),methods_completed=len(summary.get('completed_cases',{})),
        test_access=False,data_audit=audit,source_records=sources,summary_snapshots=[file_record(destination/'parent_summary_snapshot.json'),file_record(destination/'suffix_summary_snapshot.json')],
        verified_prediction_bundles=len({r['prediction_path'] for r in rows}),metric_reproduction_tolerance=1e-12,
        interpretation='Development validation; no independent test claim. SD across seeds is not a confidence interval.')
    atomic_write_json(manifest,destination/'report_manifest.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    figures=[]
    colors={'filling':'#999FA8','original_LOOK':'#237C8B','ssf':'#C38A39','independent_fit':'#9C6388','missing_only':'#677BB4','selected_start_LOOK':'#477842','terminal_only':'#B57964','bias_only':'#A06C32','logit_affine':'#7068A0'}
    names={'normalized_mean':'Normalized mean','raw_zero':'Raw zero','paired_cgan':'Paired cGAN'}
    for filling in ('normalized_mean','raw_zero','paired_cgan'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
        for ax,pattern in zip(axes,('oct_missing','cfp_missing')):
            for mi,method in enumerate(('filling','original_LOOK')):
                for fi,fusion in enumerate(('layer3','feature','layer2')):
                    subset=[r for r in rows if (r['filling'],r['fusion'],r['scenario'],r['metric'],r['method'])==(filling,fusion,pattern,'macro_f1',method)]
                    for r in subset:ax.scatter(fi+(mi-.5)*.22+(r['seed']-3408)*.035,r['value'],s=20,color=colors[method],alpha=.6)
                    a=[r for r in agg if (r['filling'],r['fusion'],r['scenario'],r['metric'],r['method'])==(filling,fusion,pattern,'macro_f1',method)]
                    if a:ax.errorbar(fi+(mi-.5)*.22,a[0]['mean'],yerr=a[0]['sd'],color=colors[method],fmt='s',capsize=5,label=method if fi==0 else None)
            ax.set(xticks=range(3),xticklabels=['layer3','feature','layer2'],xlim=(-.4,2.4),ylim=(0,1),ylabel='Validation Macro-F1',title=pattern.replace('_',' ').upper())
            ax.legend(handles=[Line2D([],[],marker='o',ls='',color=colors[m],label=label) for m,label in [('filling','Filling baseline'),('original_LOOK','Original LOOK')]],fontsize=9)
            for fi,fusion in enumerate(('layer3','feature','layer2')):
                if not any(r['fusion']==fusion and r['filling']==filling for r in rows):ax.text(fi,.08,'Pending',ha='center',color='#777777',fontsize=9)
        fig.suptitle(f'{names[filling]} | points: individual seeds; squares: mean ± SD only when all 3 finish',fontsize=12)
        name=f'{filling}__main_effects'
        for ext in ('png','svg'):fig.savefig(destination/f'{name}.{ext}',dpi=160)
        plt.close(fig);figures.append((f'{names[filling]}: completed cases',destination/f'{name}.png'))
    fig,axes=plt.subplots(1,3,figsize=(12,4.7),layout='constrained')
    for ax,fusion in zip(axes,('layer3','feature','layer2')):
        for method in ('filling','original_LOOK'):
            points=[]
            for ratio in (.2,.4,.6,.8,1.):
                a=[r for r in agg if (r['fusion'],r['filling'],r['method'],r['scenario'],r['metric'])==(fusion,'normalized_mean',method,f'random_{ratio:.1f}','macro_f1')]
                if a:points.append((ratio,a[0]['mean'],a[0]['sd']))
                rr=[r for r in rows if (r['fusion'],r['filling'],r['method'],r['scenario'],r['metric'])==(fusion,'normalized_mean',method,f'random_{ratio:.1f}','macro_f1')]
                for r in rr:ax.scatter(ratio+(r['seed']-3408)*.015,r['value'],color=colors[method],alpha=.45,s=10)
            if points:
                x,y,sd=zip(*points);ax.errorbar(x,y,yerr=sd,fmt='o-',ms=3,capsize=3,color=colors[method],label=method)
        ax.set(title=fusion,ylim=(0,1),xticks=[.2,.4,.6,.8,1.],xticklabels=['20','40','60','80','100'],xlabel='Participants missing ONE modality (%)',ylabel='Validation Macro-F1')
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    fig.suptitle('Normalized mean | nested missingness, fixed directions | mean ± SD only for 3 complete seeds',fontsize=12)
    for ext in ('png','svg'):fig.savefig(destination/f'random_missingness.{ext}',dpi=160)
    plt.close(fig);figures.append(('Random missingness: frozen direction configs',destination/'random_missingness.png'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    methods=['original_LOOK','selected_start_LOOK','terminal_only','ssf','independent_fit','missing_only','bias_only','logit_affine']
    for ax,pattern in zip(axes,('oct_missing','cfp_missing')):
        for i,method in enumerate(methods):
            rr=[r for r in rows if (r['fusion'],r['filling'],r['method'],r['scenario'],r['metric'])==('layer3','normalized_mean',method,pattern,'macro_f1')]
            for r in rr:ax.scatter(i+(r['seed']-3408)*.09,r['value'],color=colors[method],s=24)
            a=[r for r in agg if (r['fusion'],r['filling'],r['method'],r['scenario'],r['metric'])==('layer3','normalized_mean',method,pattern,'macro_f1')]
            if a:ax.errorbar(i,a[0]['mean'],yerr=a[0]['sd'],fmt='ks',capsize=4,ms=4)
            if not rr:ax.text(i,.06,'Pending',rotation=90,ha='center',color='#777777',fontsize=9)
        ax.set(xticks=range(len(methods)),xticklabels=[m.replace('_','\n') for m in methods],ylim=(0,1),ylabel='Validation Macro-F1',title=pattern.replace('_',' ').upper())
    fig.suptitle('Bounded method comparison | layer3 + normalized_mean | 3 seeds',fontsize=13)
    for ext in ('png','svg'):fig.savefig(destination/f'method_comparison.{ext}',dpi=160)
    plt.close(fig);figures.append(('Method evidence: available vs pending',destination/'method_comparison.png'))
    sections=[('Study status and evidence boundary',[
        f"Snapshot UTC: {manifest['generated_at_utc']}. Parent: {manifest['parent_completed']}/52; suffix: {manifest['suffix_completed']}/243 (includes 30 reuse slots); new controls: {manifest['methods_completed']}/{summary.get('expected_new_cases',9)}.",
        'All shown values are development validation results. Fusion, dimensions, factors, nodes and starts use this validation set for selection. They are not independent test evidence.',
        'Train fits PCA and paired Ridge W/b. SSF and the output-only control use train labels for fitting. Frozen original backbone and classifier are shared by all comparisons.',
        'Primary and natural test cohorts remain sealed. Check participant overlap before treating the two test reports as independent replications.']),
      ('Questions and fixed experimental scope',[
        'SSF: compare feature reconstruction with supervised scale-and-shift adaptation of the retained encoder and fusion operations; frozen head and backbone.',
        'Independent fit: fit each correction without upstream corrections; evaluate the resulting sequence with accepted upstream corrections. This tests conditional fitting.',
        'Missing-only writeback: joint regression sees both branches, but only the missing branch is replaced before fusion. This tests the value of correcting the retained branch.',
        f"{summary.get('expected_new_cases',9)} method cases, each with 2 missing directions and 5 frozen random ratios. Run after the parent and suffix queues. Terminal-only is reused."]),
      ('Selection, controls and reporting rules',[
        'At every allowed site: evaluate OFF, search dimensions 8,16,32,64,96,128,192,256,384,512, and enable only a strict validation Macro-F1 improvement. Ties stay OFF.',
        'Spatial reduction rho = 1/4, 1/8, 1/16; vector features use identity reduction. Dmax = 512 is independent of the search grid.',
        'Best start is chosen separately for OCT missing and CFP missing; exact ties choose the earlier start. Random ratios reuse these frozen direction configurations without another search.',
        'Start-selected validation improvement is partly guaranteed by inclusion of original LOOK. Only a later sealed-test comparison can establish added generalization value.']),
      ('Mechanism and resource interpretation',[
        'Export paired final-feature L2 errors, classifier margins, wrong-to-right and right-to-wrong counts, including cases whose representation gets closer but prediction worsens.',
        'For final linear head difference w: |delta margin| <= ||w||_2 ||delta feature||_2. A bound below the absolute complete-model margin preserves that model decision, not necessarily correctness.',
        'Report actual PCA, fitting/search and warm inference costs separately. Shared PCA is counted once; GPU peak allocation is measured, not inferred from parameter count.',
        'Sequential step gains are conditional on accepted upstream corrections. They are not independent node contributions. No small-device deployment claim is established here.']),
      ('Comparator specification and limitations',[
        'Reference: Reza et al., Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation. https://arxiv.org/abs/2310.03986',
        'SSF is a matched LOOK-task adaptation, not a reproduction of the authors original benchmark. Missing input uses the same normalized-mean filler; the missing branch remains frozen and unadapted.',
        'Prespecified AdamW learning rates 1e-5, 6e-5, 3e-4; weight decay 0.01; at most 100 epochs; patience 15; microbatch 8, effective batch 128; FP32. Identity epoch 0 is an allowed candidate.',
        'Model selection budgets differ between LOOK and SSF; report search cost and supervision differences. Three-seed SD measures seed variability, not participant-level confidence intervals.']),
      ('Advisor discussion and final-test freeze',[
        'Primary development comparisons: original LOOK vs SSF, vs independent fit, and vs missing-only, at fixed layer3 / normalized_mean. Other fillings and starts remain visible as context.',
        'Report Macro-F1 with AUROC, AUPRC, calibration and all unfavorable differences. Do not expand hypotheses after seeing favorable test subsets.',
        'Before any independent evaluation, freeze both method configuration lists, primary contrasts, uncertainty procedure and report scope. This queue does not authorize opening test data.',
        'Deliverables include all_metrics.csv, paired_method_differences.csv, three_seed_summary.csv, representation_diagnostics.json, costs.json, report_manifest.json and the existing suffix decision/gain reports.'])]
    from .advisor_findings import finding_sections, write_speaker_notes, tradeoff_figure, workflow_figure
    figures.insert(0,('Method and data roles',workflow_figure(destination)))
    findings=finding_sections(rows,agg)
    sections[1:1]=findings
    sections.append(('Two simple explanations tested',[
        'Bias-only: fix W = 0 and fit b = mean(train full latent - train missing latent). Keep the shared PCA, sites, dimensions, factors and strict validation acceptance. Downstream fitting inherits accepted upstream corrections.',
        'Output-only: fit positive a and offset c to train binary cross-entropy, s_new = a*s + c. Fixed identity-centered L2 = 1e-6; a >= 1e-8. Validation chooses fitted versus identity only; ties stay OFF.',
        'Positive slope preserves AUROC and AUPRC within a fixed missing direction. Mixing differently transformed directions at random ratios can change ranking. No intermediate features change.',
        'These six added cases were specified after examining development results. They are exploratory mechanism controls; test remains sealed. Existing SSF and mechanism settings are unchanged.']))
    tradeoff=tradeoff_figure(rows,destination)
    if tradeoff:figures.append(('Classification gains and probability tradeoffs',tradeoff))
    write_speaker_notes(rows,agg,manifest,destination)
    if audit:
        counts=audit['split_counts']
        sections[0][1].insert(1,f"Primary cohort: train {counts['train']}, validation {counts['validation']}, sealed test {counts['test']} participants. Recorded participant split leakage: {audit['participant_split_leakage']}. Counts are from the existing audit; no test inference was run.")
    text='# LOOK advisor report\n\n'
    for title,paragraphs in sections:text+=f'## {title}\n\n'+'\n\n'.join(paragraphs)+'\n\n'
    for title,path in figures:text+=f'## {title}\n\n![{title}]({path.name})\n\n'
    atomic_write_text(text,destination/'advisor_report.md')
    if pdf:render_pdf(destination/'advisor_report.pdf',sections,figures,manifest)
    return manifest


def render_pdf(path,sections,figures,manifest):
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from xml.sax.saxutils import escape
    from .state import durable_replace
    path=Path(path);temporary=path.with_suffix('.partial.pdf')
    w,h=960,540;c=Canvas(str(temporary),pagesize=(w,h))
    c.setTitle('LOOK | Advisor evidence report');c.setAuthor('Souray Meng')
    page=0
    def frame(title,tag):
        nonlocal page
        page+=1;c.setFillColor(HexColor('#F5F7FA'));c.rect(0,0,w,h,fill=1,stroke=0)
        c.setFillColor(HexColor('#173C4C'));c.rect(0,h-10,w,10,fill=1,stroke=0)
        c.setFont('Helvetica',10);c.drawString(40,h-37,'LOOK / FROZEN MULTIMODAL CORRECTION')
        c.setFont('Helvetica-Bold',23);c.drawString(40,h-75,title)
        c.setFont('Helvetica',9);c.setFillColor(HexColor('#697381'))
        c.drawString(40,22,f"{tag} | test sealed | {manifest['generated_at_utc'][:19]} UTC")
        c.drawRightString(w-40,22,f'{page:02d}')
    style=ParagraphStyle('body',fontName='Helvetica',fontSize=15,leading=21,textColor=HexColor('#263A47'))
    items=[('text',sections[0]),*[('figure',f) for f in figures],*[('text',s) for s in sections[1:]]]
    for kind,(title,paragraphs) in items:
        if kind=='figure':
            frame(title,'POINTS = SEEDS; ERROR BARS = THREE-SEED SD')
            c.drawImage(str(paragraphs),30,60,width=900,height=365,preserveAspectRatio=True,anchor='c',mask='auto')
            c.showPage();continue
        frame(title,'DEVELOPMENT EVIDENCE');y=h-112
        for i,p in enumerate(paragraphs):
            para=Paragraph(escape(p),style);pw,ph=para.wrap(w-118,1000)
            if y-ph<48:raise ValueError(f'PDF page overflow: {title}')
            c.setFillColor(HexColor('#237C8B'));c.circle(46,y-8,3,fill=1,stroke=0)
            para.drawOn(c,62,y-ph);y-=ph+20
        c.showPage()
    c.save();durable_replace(temporary,path)
