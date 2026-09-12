"""Source-linked study progress and matched aggregate tables, never partial ranking."""
import csv
from pathlib import Path
from look.runtime.state import atomic_write_json,file_sha256,stable_hash


def summarize(tasks,output):
    import json
    def read(path):return json.loads(Path(path).read_text())
    root=Path(output);root.mkdir(parents=True,exist_ok=True);progress=[];rows=[]
    for t in tasks:
        run=Path(t['run_dir']);s=read(t['spec']);r=dict(task=t['id'],run_id=run.name,disease=s['disease'],architecture=s['model']['name'],position=s['position'],seed=s['seed'])
        if not (run/'accepted.json').exists():
            status=read(run/'status.json') if (run/'status.json').exists() else {'state':'pending'}
            progress.append(dict(r,state=status['state'],stage=status.get('stage')));continue
        from look.studies.project_case import verify_case
        accepted=verify_case(run,s)
        progress.append(dict(r,state='accepted',receipt_sha256=file_sha256(run/'accepted.json')))
        for item in read(run/'report'/'source_records.json'):
            rows.append(dict(r,method=item['method'],scenario=item['scenario'],**{k:v for k,v in item['metrics'].items() if isinstance(v,(float,int))}))
    fields=['task','run_id','disease','architecture','position','seed','method','scenario']+sorted(set().union(*(set(r)-{'task','run_id','disease','architecture','position','seed','method','scenario'} for r in rows)))
    with (root/'development_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    complete=len(progress)==81 and all(p['state']=='accepted' for p in progress)
    result=dict(schema='look_expanded_progress_v1',expected_host_cases=81,registered=len(tasks),
                accepted=sum(p['state']=='accepted' for p in progress),complete=complete,rows=len(rows),tasks=progress,
                test_access=False,statistical_scope='development_selected_models; partial rows do not authorize ranking')
    atomic_write_json(result,root/'progress.json')
    lines=['# LOOK 扩展研究进度','',f"宿主及修正完整验收：{result['accepted']}/81；已登记：{len(tasks)}。",'',
        '每项需完成父模型重放、宿主平台、选中权重重放、LOOK拟合与匹配评价，才能计入完成。',
        '原始独立模型重复训练单独计数；评价视图不是独立训练或额外参与者。',
        '完整逐格开发结果见development_results.csv；未齐的组不作最终排名。','',
        '| 任务 | 状态 | 当前阶段 |','|---|---|---|']
    for row in progress:lines.append(f"| {row['task']} | {row['state']} | {row.get('stage') or '—'} |")
    (root/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
    return result
