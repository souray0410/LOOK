"""Reproducible scientific plots from completed suffix cases; missing is never zero."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path

import numpy as np
from .state import atomic_write_json, atomic_write_text

COLORS = ['#27647B', '#D18D43', '#9375A7']
PATTERNS = ['oct_missing', 'cfp_missing']


def report_rows(summary):
    records = list(summary['completed_cases'].values())
    original = {r['context']:r for r in records if r['start_ordinal'] == 1}
    rows, nodes = [], []
    for r in records:
        for pattern in PATTERNS:
            bank = r['banks'][pattern]
            baseline = original.get(r['context'])
            rows.append(dict(case_id=r['case_id'], fusion=r['fusion_position'], filling=r['filling'], seed=r['seed'],
                pattern=pattern, start_ordinal=r['start_ordinal'], allowed_start=r['allowed_start'],
                first_enabled=bank['first_enabled'], enabled_count=bank['enabled_count'], spatial_ratio=bank['spatial_ratio'],
                macro_f1=bank['macro_f1'], delta_f1_pp=None if baseline is None else 100*(bank['macro_f1']-baseline['banks'][pattern]['macro_f1']),
                result_path=r['result']['path'], result_sha256=r['result']['sha256']))
            for d in bank['decisions']:
                nodes.append(dict(case_id=r['case_id'], fusion=r['fusion_position'], filling=r['filling'], seed=r['seed'],
                    pattern=pattern, start_ordinal=r['start_ordinal'], allowed_start=r['allowed_start'],
                    node=d['node'], ordinal=d['ordinal'], state=d['state'], factor=bank['selected_factor'],
                    chosen_dimension=d.get('chosen_dimension'), best_candidate_dimension=d.get('best_candidate_dimension'),
                    baseline_score=d.get('baseline_score'), selected_score=d.get('selected_score'), delta=d.get('delta')))
    return rows, nodes


def three_seed_rows(rows):
    groups = {}
    for r in rows:
        key = (r['fusion'],r['filling'],r['pattern'],r['start_ordinal'])
        groups.setdefault(key, []).append(r)
    result = []
    for key, values in groups.items():
        if sorted(v['seed'] for v in values) != [3407,3408,3409]:
            continue
        result.append(dict(fusion=key[0],filling=key[1],pattern=key[2],start_ordinal=key[3],n_seeds=3,
            mean_macro_f1=float(np.mean([v['macro_f1'] for v in values])),
            sd_macro_f1=float(np.std([v['macro_f1'] for v in values],ddof=1)),
            mean_delta_f1_pp=None if any(v['delta_f1_pp'] is None for v in values) else float(np.mean([v['delta_f1_pp'] for v in values])),
            sd_delta_f1_pp=None if any(v['delta_f1_pp'] is None for v in values) else float(np.std([v['delta_f1_pp'] for v in values],ddof=1))))
    return result


def write_csv(rows, path):
    if not rows:
        atomic_write_text('',path); return
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows); atomic_write_text(buffer.getvalue(),path)


def write_report(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    rows, nodes = report_rows(summary)
    aggregates = three_seed_rows(rows)
    write_csv(rows,destination/'start_effects.csv'); write_csv(nodes,destination/'node_decisions.csv')
    write_csv(aggregates,destination/'three_seed_summary.csv')
    comparisons=[]
    originals={r['context']:r for r in summary['completed_cases'].values() if r['start_ordinal']==1}
    for context, reference in summary.get('selected_contexts',{}).items():
        selected=json.loads(Path(reference['path']).read_text())
        original=originals[context]
        for scenario,value in selected['validation'].items():
            if scenario not in original['validation']: continue
            before=original['validation'][scenario]
            comparisons.append(dict(context=context,scenario=scenario,original_f1=before['macro_f1'],
                selected_start_f1=value['macro_f1'],delta_f1_pp=100*(value['macro_f1']-before['macro_f1']),
                original_result=original['result']['path'],selected_result=reference['path']))
    write_csv(comparisons,destination/'original_vs_selected_starts.csv')
    metrics = []
    for r in summary['completed_cases'].values():
        for scenario, values in r['validation'].items():
            metrics.append(dict(case_id=r['case_id'],scenario=scenario,metrics=values,result=r['result']))
    atomic_write_json(metrics,destination/'all_validation_metrics.json')
    atomic_write_json(dict(completed_cases=len(summary['completed_cases']),expected_cases=summary.get('expected_cases',243),
        three_seed_groups=len(aggregates),test_access=False,selected_contexts=summary.get('selected_contexts',{})), destination/'report_manifest.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'semibold','savefig.facecolor':'white','axes.grid':True,'grid.alpha':.16})
    def save(fig, name):
        fig.savefig(destination/f'{name}.png',dpi=170,bbox_inches='tight')
        fig.savefig(destination/f'{name}.svg',bbox_inches='tight')
        plt.close(fig)
    fillings = list(dict.fromkeys(r['filling'] for r in rows))
    fusions = list(dict.fromkeys(r['fusion'] for r in rows))
    for filling in fillings:
        fig, axes = plt.subplots(len(fusions),4,figsize=(19,3.7*len(fusions)),squeeze=False,layout='constrained')
        fig.suptitle(f'LOOK correction start | {filling}\nValidation-selected configurations • {len(summary['completed_cases'])}/{summary.get('expected_cases',243)} cases available • test sealed',fontsize=16)
        for i,fusion in enumerate(fusions):
            for j,pattern in enumerate(PATTERNS):
                subset=[r for r in rows if (r['filling'],r['fusion'],r['pattern'])==(filling,fusion,pattern)]
                for k,field in enumerate(['macro_f1','delta_f1_pp']):
                    ax=axes[i,2*j+k]
                    for seed,color in zip([3407,3408,3409],COLORS):
                        points=sorted([r for r in subset if r['seed']==seed and r[field] is not None],key=lambda r:r['start_ordinal'])
                        ax.plot([r['start_ordinal'] for r in points],[r[field] for r in points],'.-',color=color,alpha=.7,label=f'Seed {seed}',lw=1)
                    agg=sorted([r for r in aggregates if (r['filling'],r['fusion'],r['pattern'])==(filling,fusion,pattern)],key=lambda r:r['start_ordinal'])
                    valid_agg=[r for r in agg if r['mean_'+field] is not None]
                    if valid_agg:
                        ax.errorbar([r['start_ordinal'] for r in valid_agg],[r['mean_'+field] for r in valid_agg],yerr=[r['sd_'+field] for r in valid_agg],fmt='ko',ms=4,capsize=3,label='Mean ± SD (3 seeds)')
                    if field=='delta_f1_pp': ax.axhline(0,color='#48505A',lw=.8)
                    ax.set(xlim=(.7,9.3),xticks=range(1,10),xlabel='Allowed start (canonical ordinal)',
                        ylabel='Macro-F1' if k==0 else 'Δ Macro-F1 vs start 1 (pp)', title=f'Fusion: {fusion} | {pattern}')
                    if field=='macro_f1': ax.set_ylim(0,1)
                    if i==0 and j==0: ax.legend(fontsize=7)
        save(fig,f'{filling}__start_effects')
        for fusion in fusions:
            selected=[r for r in summary['completed_cases'].values() if r['filling']==filling and r['fusion_position']==fusion]
            if not selected: continue
            selected.sort(key=lambda r:(r['seed'],r['start_ordinal']))
            sites=selected[0]['candidate_sites']; fusion_index=sites.index(f'fusion_{fusion}')
            fig,axes=plt.subplots(1,2,figsize=(17,max(5,len(selected)*.25+2)),layout='constrained')
            fig.suptitle(f'Correction decisions | fusion: {fusion} | {filling}\nAllowed start ○ • first enabled ★ • dashed line: fusion output',fontsize=14)
            cmap=ListedColormap(['#E7E9EC','#BBC4CB','#28788B']); norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
            for ax,pattern in zip(axes,PATTERNS):
                states=np.array([[{'excluded':0,'off':1,'on':2}[d['state']] for d in r['banks'][pattern]['decisions']] for r in selected])
                ax.imshow(states,aspect='auto',cmap=cmap,norm=norm); ax.grid(False)
                ax.axvline(fusion_index-.5,color='#343C47',ls='--',lw=1)
                for y,r in enumerate(selected):
                    bank=r['banks'][pattern]
                    ax.scatter(r['start_ordinal']-1,y,s=75,facecolors='none',edgecolors='black',linewidths=.8)
                    if bank['first_enabled']: ax.scatter(sites.index(bank['first_enabled']),y,marker='*',s=60,color='#F7C66A',edgecolors='#333333',linewidths=.3)
                ax.set(xticks=range(9),xticklabels=sites,yticks=range(len(selected)),
                    yticklabels=[f"{r['seed']} / start {r['start_ordinal']} / ρ = 1/{r['banks'][pattern]['selected_factor']}" for r in selected],title=pattern)
                ax.tick_params(axis='x',labelrotation=55,labelsize=8)
            fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(cmap.colors,['Excluded before start','Eligible, OFF','Enabled, ON'])],loc='outside lower center',ncol=3)
            save(fig,f'{filling}__{fusion}__decisions')
            fig,axes=plt.subplots(3,2,figsize=(14,10),layout='constrained')
            fig.suptitle(f'Sequential conditional gains | fusion: {fusion} | {filling}\nEach trace uses its own accepted upstream corrections; not isolated node effects',fontsize=14)
            for i,seed in enumerate([3407,3408,3409]):
                for j,pattern in enumerate(PATTERNS):
                    ax=axes[i,j]
                    for r in [v for v in selected if v['seed']==seed]:
                        ds=[d for d in r['banks'][pattern]['decisions'] if d['state']!='excluded']
                        x=[ds[0]['ordinal']-1]+[d['ordinal'] for d in ds]
                        y=[ds[0]['baseline_score']]+[d['selected_score'] for d in ds]
                        ax.plot(x,y,'.-',lw=1,label=f"Start {r['start_ordinal']}")
                    ax.set(xticks=range(10),ylim=(0,1),xlabel='Canonical position (0 = filling baseline)',ylabel='Validation Macro-F1',title=f'Seed {seed} | {pattern}')
                    if ax.lines: ax.legend(ncol=3,fontsize=7)
            save(fig,f'{filling}__{fusion}__sequential_gains')
    atomic_write_text('''# LOOK suffix-start report

All scores are development validation results. Nodes, dimensions, spatial factors and starts use validation Macro-F1. Test remains sealed.

- Canonical start ordinals are architecture specific; see the full node names in decision plots and node_decisions.csv.
- An allowed start is not forced ON. Excluded, eligible OFF and enabled ON are distinct states.
- Spatial shrinkage is labelled rho = 1/4, 1/8 or 1/16; vector sites use factor 1.
- All completed individual seed points are retained. Means and sample SD (ddof=1) require exactly seeds 3407, 3408 and 3409; missing cases are never zero-filled.
- Sequential gains are conditional on accepted upstream corrections, not isolated causal node contributions.
- Negative differences are retained. all_validation_metrics.json includes secondary and calibration metrics and exact result provenance.
- The selected-start variant searches more configurations, so its selected validation improvement is not independent evidence of generalization.
''',destination/'README.md')
    write_advisor_report(summary,destination)
    return dict(completed_cases=len(summary['completed_cases']),plots=len(list(destination.glob('*.png'))))


def write_advisor_report(summary, destination):
    """A reviewable development report with matched controls and source citations."""
    from .start_study import read_json, source_result, file_record
    parent_ref = summary.get('identity', {}).get('parent_study')
    parent = read_json(parent_ref) if parent_ref else None
    parent_records = {}
    control_rows = []
    if parent:
        runs = Path(parent_ref).parents[2]
        for stage in parent['completed_stages']:
            path = source_result(parent, runs, stage)
            result = read_json(path)
            parent_records[stage] = result
            result_reference = file_record(path)
            for scenario, metric in result['validation'].items():
                control_rows.append(dict(stage=stage, fusion=result['selection']['fusion_position'],
                    filling=result['selection']['filling_strategy'],seed=result['selection']['seed'],scenario=scenario,
                    macro_f1=metric['macro_f1'],auroc=metric.get('macro_auroc_ovr'),auprc=metric.get('macro_auprc_ovr'),
                    ece=metric.get('ece_15'),brier=metric.get('multiclass_brier'),source=str(path),source_sha256=result_reference['sha256']))
    write_csv(control_rows,destination/'parent_controls.csv')
    records=list(summary['completed_cases'].values())
    originals={r['context']:r for r in records if r['start_ordinal']==1}
    for stage,r in parent_records.items():
        if stage.startswith(('normalized_mean_', 'raw_zero_', 'paired_cgan_')):
            originals.setdefault(stage, {'validation':r['validation']})
    selected={k:read_json(v['path']) for k,v in summary.get('selected_contexts',{}).items()}
    metrics=['macro_f1','macro_auroc_ovr','macro_auprc_ovr','ece_15','multiclass_brier']
    def display(values):
        if len(values)!=3:return f'Pending ({len(values)}/3 seeds)'
        return f'{np.mean(values):.4f} ± {np.std(values,ddof=1):.4f}'
    lines=['# LOOK: correction-start study — supervisor review', '',
        '**Evidence stage: development validation. Independent test remains sealed.**', '',
        f"Parent stages completed: {len(parent['completed_stages']) if parent else 'not loaded'}/52. "
        f"Suffix cases completed or referenced: {len(records)}/243. "
        f"Contexts with all nine starts selected and re-evaluated: {len(selected)}/27.", '',
        '## Research question and design', '',
        'Does allowing correction to begin later improve a frozen multimodal classifier under missing inputs, compared with the original input-start greedy LOOK procedure?', '',
        'The parent screens seven fusion locations with seed 3407 and replicates the selected three with seeds 3408 and 3409. This does not establish a globally optimal fusion location. The supplement tests nine ordered suffixes for every selected fusion, filling and seed. It fits each suffix independently, preserving accepted upstream corrections and optional OFF decisions.', '',
        'Train fits PCA and W/b. Validation selects checkpoint, dimensions, node activation, spatial factor and correction start. Missing directions are selected separately; random missingness reuses these fixed banks and nested participant masks. The primary test is contained within the natural-distribution test, so those two evaluations must not be described as independent cohorts.', '',
        '## Matched primary comparisons', '',
        'Each row uses the same fusion, filling, model seeds and validation participants. Values are mean ± sample SD across exactly three seeds, not confidence intervals. Partial seed sets are explicitly pending. Complete-input scores are contextual references, not assumed attainable upper bounds.', '',
        '| Fusion | Filling | Missing direction | Complete input F1 | Filling F1 | Original LOOK F1 | Selected-start LOOK F1 |',
        '|---|---|---|---:|---:|---:|---:|']
    for fusion in (parent['selected_fusions'] if parent else list(dict.fromkeys(r['fusion_position'] for r in records))):
        for filling in (['normalized_mean','raw_zero','paired_cgan'] if parent else list(dict.fromkeys(r['filling'] for r in records))):
            for pattern in PATTERNS:
                contexts=[f'{filling}_{fusion}_{seed}' for seed in (3407,3408,3409)]
                base=[originals[c] for c in contexts if c in originals]
                if not base:continue
                cells=[display([r['validation'][scenario]['macro_f1'] for r in base]) for scenario in ('complete',f'fill_{pattern}',f'look_after_fill_{pattern}')]
                cells.append(display([selected[c]['validation'][f'look_after_fill_{pattern}']['macro_f1'] for c in contexts if c in selected]))
                lines.append(f"| {fusion} | {filling} | {pattern} | "+' | '.join(cells)+' |')
    lines += ['', '## Architectural and correction controls', '',
        'Full multimodal scores above come from complete-input training. OCT-only and CFP-only are independently trained controls; they are not missing-input versions of the frozen multimodal backbone. For OCT missing, CFP-only is the available-modality comparator; for CFP missing, OCT-only is the comparator.', '',
        '| Unimodal control | Validation Macro-F1, mean ± SD |', '|---|---:|']
    for modality in ('oct_only','cfp_only'):
        values=[r['validation']['complete']['macro_f1'] for stage,r in parent_records.items() if stage.startswith('reference_') and r['selection']['fusion_position']==modality]
        lines.append(f'| {modality} | {display(values)} |')
    lines += ['', 'The existing input-only and fusion-only ablations use normalized_mean and seed 3407 only. They cannot be presented as three-seed evidence.', '',
        '| Fusion | Ablation | OCT missing F1 | CFP missing F1 |', '|---|---|---:|---:|']
    for stage,r in parent_records.items():
        if stage.startswith(('input_only_','fusion_only_')):
            method='input_only' if stage.startswith('input_only_') else 'fusion_only'
            lines.append(f"| {r['selection']['fusion_position']} | {method} | {r['validation']['look_after_fill_oct_missing']['macro_f1']:.4f} | {r['validation']['look_after_fill_cfp_missing']['macro_f1']:.4f} |")
    lines += ['', '## Secondary metrics, random missingness and negative results', '',
        'Macro-F1 remains the sole selection objective. AUROC/AUPRC use raw logit-difference rankings; calibration is reported with ECE and Brier score. A gain in selected F1 does not imply a gain in every secondary metric or random-missingness condition. All scenarios, including adverse results, are retained in the linked tables.', '',
        'Random ratios 20/40/60/80/100% mean that the corresponding fraction of participants lose one entire modality. Both eyes share the same missing direction. At 100%, each participant still retains one modality.', '',
        '## Interpretation and limits', '',
        '- The input-start configuration is included in the search, so best-start validation F1 cannot decrease. This is a search property, not independent evidence of generalization.',
        '- The independent frozen test comparison must decide whether start selection adds useful generalization beyond the original LOOK method. No test result is available in this report.',
        '- Sequential node gains are conditional on upstream accepted corrections. They are not isolated causal contributions of individual nodes.',
        '- Three seeds describe training variability; they do not replace participant-level uncertainty analysis. Do not infer significance from overlapping or non-overlapping seed SD bars.',
        '- This study does not measure model compression or small-device deployment. No such contribution is established by these results.', '',
        '## Evidence index', '',
        '- [Parent controls and provenance](parent_controls.csv): screening, selected replications, filling, existing LOOK/ablations and unimodal controls.',
        '- [Every suffix effect and source hash](start_effects.csv); [node decisions](node_decisions.csv); [complete three-seed groups](three_seed_summary.csv).',
        '- [Original versus selected-start scenarios](original_vs_selected_starts.csv); [all suffix validation metrics](all_validation_metrics.json).',
        '- The PNG/SVG files in this directory show start effects, excluded/OFF/ON sites and sequential conditional gains. Frozen selections and matrix references are in the sibling selected_starts directory.', '']
    atomic_write_text('\n'.join(lines),destination/'advisor_report.md')
