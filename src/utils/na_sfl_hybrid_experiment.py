"""Approved Hybrid Smoke→100round dispatch and reporting in the existing CLI."""
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from utils.deit_artifacts import csv_file, gpu_snapshot, json_file
from utils.na_sfl_config import DEFAULTS, HYBRID_METHOD, load_config
from utils.na_sfl_network import rate_scenario
from utils.na_sfl_bn_experiments import (_now, last10_mean, controlled_termination,
    owned_stage_process, wait_stage_process, _retention_audit)
from utils.sflv2_metrics import sha256, snapshot_sources, check_sources_unchanged


def _read(path):
    return json.loads(Path(path).read_text())


def validate_matched_config(baseline, hybrid):
    if (baseline['method'] != 'na_sfl_v3_proto' or baseline['rounds'] != 100
            or baseline['bn_policy'] != 'dual_running_stats_shared_affine'
            or baseline['prototype']['fast_sd_weight'] != .8
            or baseline['prototype']['fast_proto_weight'] != .2):
        raise ValueError('baseline must be completed v3 Dual BN0.8:0.2')
    expected = deepcopy(baseline)
    expected.update(method=HYBRID_METHOD, random_sd=hybrid['random_sd'],
        result_path=hybrid['result_path'], device=hybrid['device'])
    if expected != hybrid:
        differences = [k for k in set(expected) | set(hybrid) if expected.get(k) != hybrid.get(k)]
        raise ValueError('Hybrid config differs from baseline beyond sampling/method/output: '+str(differences))


def validate_rate_rows(rows, baseline_rows):
    # apply_wall_s is a measurement, not network identity. The scenario hash
    # covers the requested horizon and differs for a2round Smoke vs100round.
    if len(rows)>len(baseline_rows):raise ValueError('network reference has too few rounds')
    fields=('round','clients','kernel_rates')
    for i,row in enumerate(rows):
        actual,expected=row.get('network_rates'),baseline_rows[i].get('network_rates')
        if (not isinstance(actual,dict) or not isinstance(expected,dict)
                or any(k not in actual or k not in expected or actual[k]!=expected[k] for k in fields)):
            raise ValueError('actual Hybrid per-round rates differ from baseline')


def validate_recovery_contract(root, previous, old_sources, new_sources):
    allowed={'utils/na_sfl_hybrid_experiment.py','tests/test_na_sfl_hybrid_artifacts_cpu.py'}
    changed={k for k in old_sources.keys()|new_sources.keys() if old_sources.get(k)!=new_sources.get(k)}
    if (previous.get('status')!='failed' or previous.get('active_condition')!='smoke_test'
            or previous.get('active_child_pid') is not None
            or previous.get('error')!='actual Hybrid per-round rates differ from baseline'
            or (root/'run_100round').exists() or (root/'inputs/run_100round.yaml').exists()
            or (root/'smoke_passed.json').exists() or set(old_sources)!=set(new_sources)
            or not changed or not changed<=allowed):
        raise ValueError('completed Smoke recovery requires only the reviewed gate/test correction and no main launch')


def accuracy_record(name, run, summary, rows):
    mean = last10_mean(rows)
    evaluations = ([{'round':0, **summary['initial_evaluation']}] if summary.get('initial_evaluation') else [])+rows
    crossing = next((r for r in evaluations if r['test_accuracy'] >= 70), None)
    best = max(evaluations, key=lambda r:r['test_accuracy'])
    return {'name':name, 'run_dir':str(run), 'completed_rounds':len(rows),
        'last10_mean':float(mean), 'last10_mean_exact':str(mean),
        'final_accuracy':rows[-1]['test_accuracy'], 'best_accuracy':best['test_accuracy'], 'best_round':best['round'],
        'first_ge_70':crossing['round'] if crossing else None,
        'tta_70':crossing['elapsed_wall_s'] if crossing else None,
        'training_through_evaluation_s':rows[-1]['elapsed_wall_s'],
        'total_wall_time':summary['launch_wall_s_debug_only'],
        **{k:summary.get(k) for k in ('upload_bytes','download_bytes','upload_payload_bytes','download_payload_bytes',
            'upload_float_bytes','download_float_bytes','server_updates','partition_sha256','data_sha256',
            'network_rate_scenario_sha256','initial_model_hashes')}}


def _communication(run):
    """Read original records only; never rewrite a baseline artifact."""
    totals = {group: {'upload_bytes':0,'download_bytes':0,'upload_float_bytes':0,
        'download_float_bytes':0,'jobs':0,'samples':0,'actual_sd_count':0,'actual_proto_count':0}
        for group in ('fast','slow','all')}
    for line in (run/'per_job.jsonl').open():
        row=json.loads(line); group='slow' if row['client_id'] in (0,5) else 'fast'
        for name in (group,'all'):
            target=totals[name]
            target['jobs']+=1; target['samples']+=row['samples']
            target['actual_sd_count']+=row['samples'] if row['job_kind']=='SD_JOB' else row.get('selected_sd_count',0)
            target['actual_proto_count']+=len(row['proto_class_ids'])
            for k, source in [('upload_bytes','upload_bytes'),('download_bytes','download_bytes'),
                              ('upload_float_bytes','activation_bytes'),('download_float_bytes','gradient_bytes')]:
                target[k]+=row[source]
    return totals


def model_traffic(run):
    result={key:0 for key in ('model_upload_bytes','model_download_bytes',
        'model_upload_payload_bytes','model_download_payload_bytes')}
    for frame in _read(run/'network/network_metadata.json')['model_traffic']:
        direction=frame['direction']
        if direction not in ('upload','download'):raise ValueError('invalid model traffic direction')
        result['model_'+direction+'_bytes']+=frame['wire_bytes']
        result['model_'+direction+'_payload_bytes']+=frame['payload_bytes']
    return result


def render_comparison(root, baseline_run, hybrid_run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records, series, communication = [], [], []
    for name, run in [('v3_dual_bn_08_02',baseline_run),('prototype_random_sd_10pct',hybrid_run)]:
        summary=_read(run/'summary.json'); rows=_read(run/'round_metrics.json')
        record=accuracy_record(name,run,summary,rows)
        traffic=_communication(run)
        record.update({k:traffic['all'][k] for k in ('upload_float_bytes','download_float_bytes')})
        record.update(model_traffic(run))
        record['total_upload_bytes_including_models']=record['upload_bytes']+record['model_upload_bytes']
        record['total_download_bytes_including_models']=record['download_bytes']+record['model_download_bytes']
        record['gpu']=_read(run/'environment.json')['gpu']
        records.append(record); series.append((name,summary,rows))
        communication.extend({'name':name,'client_group':g,**v,
            **({k:record[k] for k in ('model_upload_bytes','model_download_bytes',
                'total_upload_bytes_including_models','total_download_bytes_including_models')} if g=='all' else {})}
            for g,v in traffic.items())
    for record in records:
        record['delta_last10_vs_baseline_pp']=float(Decimal(record['last10_mean_exact'])-Decimal(records[0]['last10_mean_exact']))
    csv_file(root/'comparison.csv',records); json_file(root/'comparison.json',records)
    csv_file(root/'communication_bytes.csv',communication)
    for original, alias in [('per_round.csv','metrics_by_round.csv'),
                            ('per_client_round.csv','metrics_by_client.csv')]:
        (root/alias).write_bytes((hybrid_run/original).read_bytes())
    for filename,key,label,scale in [('accuracy_by_round.png','round','Round',1),
                                    ('accuracy_by_elapsed.png','elapsed_wall_s','Elapsed wall (hours)',3600)]:
        fig,ax=plt.subplots(figsize=(9,4.8),constrained_layout=True)
        for name,summary,rows in series:
            evaluated=[{'round':0,**summary['initial_evaluation']}]+rows
            ax.plot([r[key]/scale for r in evaluated],[r['test_accuracy'] for r in evaluated],label=name,linewidth=1.3)
        ax.axhline(70,color='gray',linestyle=':',linewidth=1)
        ax.set(xlabel=label,ylabel='Original SD Test accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(root/filename,dpi=160);plt.close(fig)
    lines=['# Prototype + Random SD10% versus original Dual BN v3','',
        'Same GPU/initialization/partition/network scenario; single seed42, live FCFS order can differ.','',
        '| Condition | Final | Best | Last10 mean | Delta(pp) | TTA70(s) | Training through evaluation(s) | Job UL(bytes) | Job DL(bytes) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['name']}|{r['final_accuracy']:.2f}|{r['best_accuracy']:.2f}|{r['last10_mean']:.3f}|"
            f"{r['delta_last10_vs_baseline_pp']:.3f}|{r['tta_70']}|{r['training_through_evaluation_s']:.3f}|{r['upload_bytes']}|{r['download_bytes']}|")
    lines+=['','![Accuracy by round](accuracy_by_round.png)','','![Accuracy by elapsed](accuracy_by_elapsed.png)','',
        'Job wire bytes include application framing/metadata; model/FedAvg traffic is recorded separately in each network metadata file. '
        'TCP/IP/link overhead is excluded. SD10% denotes sampled SD count, not total network byte ratio. '
        'TTA70 is the first evaluated crossing; not a maintained-accuracy criterion. Missing crossing is null.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return records


def _validate_run(run, expected_rounds, baseline, gpu_uuid, sources, *, smoke=False):
    summary=_read(run/'summary.json'); rows=_read(run/'round_metrics.json'); env=_read(run/'environment.json')
    if (not summary['completed'] or summary['completed_rounds'] != expected_rounds
            or [r['round'] for r in rows] != list(range(1,expected_rounds+1))
            or summary['samples_seen'] != 30000*expected_rounds or summary['server_updates'] != 240*expected_rounds):
        raise ValueError('Hybrid run incomplete or wrong visits/steps')
    for key in ('initial_model_hashes','partition_sha256','data_sha256'):
        if summary[key] != baseline[key]:raise ValueError('Hybrid/baseline identity mismatch: '+key)
    if env['gpu']['uuid'] != gpu_uuid or env['source_sha256'] != sources:
        raise ValueError('Hybrid GPU/source identity mismatch')
    baseline_rates=_read(Path(baseline['run_dir'])/'round_metrics.json')
    validate_rate_rows(rows,baseline_rates)
    run_config=_read(run/'config.json')
    if summary['network_rate_scenario_sha256']!=rate_scenario(run_config['network'],expected_rounds)['sha256']:
        raise ValueError('Hybrid horizon network scenario differs from configured scenario')
    if not smoke and summary['network_rate_scenario_sha256'] != baseline['network_rate_scenario_sha256']:
        raise ValueError('Hybrid network scenario differs from baseline')
    bn=summary['server_batchnorm']; calls={'sd':240*expected_rounds,'proto':240*expected_rounds}
    if (bn['policy']!='dual_running_stats_shared_affine' or bn['forward_calls']!=calls
            or bn['running_updates']!=calls or len(bn['running_counters'])!=10
            or any(value!=calls for value in bn['running_counters'].values())):
        raise ValueError('Hybrid actual Dual BN counters differ')
    jobs=[json.loads(line) for line in (run/'per_job.jsonl').open()]
    hybrid=[j for j in jobs if j['job_kind']=='HYBRID_JOB']
    if (len(jobs)!=240*expected_rounds or len(hybrid)!=48*expected_rounds
            or sum(j['selected_sd_count'] for j in hybrid)!=610*expected_rounds):
        raise ValueError('Hybrid job or sampled SD counts differ')
    from utils.na_sfl_hybrid import sample_indices
    for j in hybrid:
        indices=sample_indices(j['samples'],.1,42,j['job_id']).tolist()
        if (j['selected_indices']!=indices or sum(j['proto_counts'])!=j['samples']
                or j['server_version_after']!=j['server_version_before']+1
                or j['client_version_after']!=j['client_version_before']+1):
            raise ValueError('Hybrid sample/whole-batch/one-update contract failed')
        expected_float=(j['selected_sd_count']+len(j['proto_class_ids']))*128*16*16*4
        if j['activation_bytes']!=expected_float or j['gradient_bytes']!=expected_float:
            raise ValueError('Hybrid float byte accounting differs from wire shapes')
    net=_read(run/'network/network_metadata.json')
    if not net['closed'] or net['cleanup_errors']:raise ValueError('network did not cleanly close')
    manifest=_read(run/'manifest.json')
    for name,digest in manifest['artifacts'].items():
        if sha256(run/name)!=digest:raise ValueError('Hybrid artifact changed: '+name)
    for name,digest in sources.items():
        if sha256(run/'source_snapshot'/name)!=digest:raise ValueError('Hybrid frozen source changed: '+name)
    if smoke and (summary['checkpoint'].get('retained') is not False or not summary['checkpoint']['finite']):
        raise ValueError('Smoke checkpoint was not verified/discarded')
    return summary


def run_hybrid_experiment(args):
    """Reuse the existing CLI in owned fresh child processes; never a new trainer."""
    from utils.na_sfl_run import validate_training_gpu
    config=args.config; source=Path(__file__).resolve().parents[1]
    if (config['method']!=HYBRID_METHOD or config['rounds']!=100
            or config['random_sd']!={'sample_ratio':.1,'sample_seed':42}):
        raise ValueError('initial Hybrid experiment requires q0.1 / sample_seed42 /100round')
    baseline_run=Path(args.baseline_reference).resolve()
    baseline=_read(baseline_run/'summary.json'); baseline['run_dir']=str(baseline_run)
    baseline_config=load_config(overrides={k:v for k,v in _read(baseline_run/'config.json').items() if k in DEFAULTS})
    validate_matched_config(baseline_config,config)
    last10_mean(_read(baseline_run/'round_metrics.json'))
    if not baseline['completed'] or baseline['completed_rounds']!=100:raise ValueError('incomplete baseline')
    manifest=_read(baseline_run/'manifest.json')
    for name,digest in manifest['artifacts'].items():
        if sha256(baseline_run/name)!=digest:raise ValueError('baseline artifact changed: '+name)
    for name,digest in manifest['source_sha256'].items():
        if sha256(baseline_run/'source_snapshot'/name)!=digest:raise ValueError('baseline frozen source changed')
    gpu=gpu_snapshot(config['device']);validate_training_gpu(gpu,allow_unprofiled=config['allow_unprofiled_gpu'])
    if gpu['uuid']!=_read(baseline_run/'environment.json')['gpu']['uuid']:raise ValueError('baseline GPU mismatch')
    scenario=rate_scenario(config['network'],100)
    if scenario['sha256']!=baseline['network_rate_scenario_sha256']:raise ValueError('baseline network mismatch')
    root=Path(config['result_path'])
    if root==baseline_run or root.is_relative_to(baseline_run) or baseline_run.is_relative_to(root):
        raise ValueError('Hybrid result overlaps baseline')
    root.mkdir(parents=True,exist_ok=True)
    recovery=None
    if (root/'status.json').exists():
        previous=_read(root/'status.json')
        references=list((root/'smoke_test').glob('*/summary.json'))
        recovery_root=root/'recovery_01'
        if len(references)!=1 or recovery_root.exists():
            raise ValueError('Hybrid destination already contains a launch without recoverable Smoke')
        reused_run=references[0].parent
        old_sources=_read(reused_run/'environment.json')['source_sha256']
        sources=snapshot_sources(recovery_root)
        validate_recovery_contract(root,previous,old_sources,sources)
        _validate_run(reused_run,2,baseline,gpu['uuid'],old_sources,smoke=True)
        # Preserve every original run/snapshot/log. The failed supervisor's
        # mutable status and exception get immutable copies before restarting.
        json_file(recovery_root/'previous_status.json',previous)
        (recovery_root/'previous_exception.log').write_bytes((root/'exception.log').read_bytes())
        json_file(recovery_root/'source_sha256.json',sources)
        recovery={'smoke_run':str(reused_run),'reason':'fix network identity projection; reuse unchanged completed training',
            'original_sources':old_sources,'new_sources':sources,'recovered_at':_now(),
            'prior_status':previous,'source_snapshot':str(recovery_root/'source_snapshot')}
        json_file(root/'recovery.json',recovery)
    else:
        if any((root/name).exists() for name in ('smoke_test','run_100round')):
            raise ValueError('Hybrid destination already contains a launch')
        sources=snapshot_sources(root);json_file(root/'source_sha256.json',sources)
    (root/'inputs').mkdir(exist_ok=True)
    json_file(root/'network_rate_scenario.json',scenario)
    reference={'run_dir':str(baseline_run),'summary':baseline,
        'last10_mean_exact':str(last10_mean(_read(baseline_run/'round_metrics.json'))),
        'input_sha256':manifest['artifacts'],'source_sha256':manifest['source_sha256']}
    json_file(root/'baseline_reference.json',reference)
    state={'status':'running','completed':False,'started_at':_now(),'supervisor_pid':os.getpid(),
        'gpu_uuid':gpu['uuid'],'active_condition':None,'active_child_pid':None,'active_rounds_completed':0}
    if recovery:
        state.update(recovery=recovery,started_at=recovery['prior_status']['started_at'])
    def status():
        state['updated_at']=_now();json_file(root/'status.json',state)
    def launch(name,rounds,smoke):
        check_sources_unchanged(sources)
        for f,digest in manifest['artifacts'].items():
            if sha256(baseline_run/f)!=digest:raise ValueError('baseline changed before launch')
        child_config={**{k:config[k] for k in DEFAULTS},'rounds':rounds,
            'smoke_test':smoke,'result_path':str(root/name)}
        path=root/'inputs'/(name+'.yaml');path.write_text(yaml.safe_dump(child_config,sort_keys=True))
        command=[sys.executable,'-u',str(source/'run_exp_main.py'),'--config',str(path)]
        json_file(root/(name+'_launch.json'),{'command':command,'config_sha256':sha256(path),
            'gpu_uuid':gpu['uuid'],'source_sha256':sources})
        state.update(active_condition=name,active_rounds_completed=0,active_run_dir=None);status()
        with (root/(name+'.console.log')).open('ab',buffering=0) as log, owned_stage_process(
                command,cwd=source,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL) as child:
            state['active_child_pid']=child.pid;status()
            def progress():
                paths=list((root/name).glob('*/round_metrics.json'))
                if len(paths)==1:
                    try:state.update(active_rounds_completed=len(_read(paths[0])),active_run_dir=str(paths[0].parent))
                    except (OSError,ValueError):pass
                status()
            result=wait_stage_process(child,progress)
        state['active_child_pid']=None;status()
        if result:raise RuntimeError(name+' failed with exit '+str(result))
        summaries=list((root/name).glob('*/summary.json'))
        if len(summaries)!=1:raise ValueError('expected one Hybrid stage result')
        run=summaries[0].parent
        summary=_validate_run(run,rounds,baseline,gpu['uuid'],sources,smoke=smoke)
        state.update(active_rounds_completed=rounds,active_run_dir=str(run));status()
        json_file(root/(name+'_audit.json'),{'passed':True,'run_dir':str(run),'rounds':rounds,
            'identity_verified':True,'visits':summary['samples_seen'],'server_updates':summary['server_updates'],
            'hybrid_jobs':48*rounds,'selected_sd':610*rounds,'server_batchnorm':summary['server_batchnorm'],
            'memory_peaks':[{k:r[k] for k in ('round','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes')}
                for r in _read(run/'round_metrics.json')],
            'memory_scope':'Per-round actual peaks; a two-round Smoke does not prove long-horizon absence of growth.'})
        return run,summary
    status()
    try:
        with controlled_termination():
            if recovery:
                smoke_run=Path(recovery['smoke_run'])
                smoke_summary=_read(smoke_run/'summary.json')
                json_file(root/'smoke_test_audit.json',{'passed':True,'run_dir':str(smoke_run),
                    'rounds':2,'identity_verified':True,'visits':60000,'server_updates':480,
                    'hybrid_jobs':96,'selected_sd':1220,'server_batchnorm':smoke_summary['server_batchnorm'],
                    'memory_peaks':[{k:r[k] for k in ('round','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes')}
                        for r in _read(smoke_run/'round_metrics.json')],
                    'memory_scope':'Two completed rounds; not proof of unbounded-growth absence.',
                    'recovery_provenance':'recovery.json'})
            else:
                smoke_run,_=launch('smoke_test',2,True)
            # The Smoke is a fresh run; never resume its model for the main run.
            json_file(root/'smoke_passed.json',{'run_dir':str(smoke_run),'passed':True,'completed_at':_now()})
            main_run,summary=launch('run_100round',100,False)
            records=render_comparison(root,baseline_run,main_run)
            _retention_audit(root)
            check_sources_unchanged(sources)
            state.update(status='completed',completed=True,active_condition=None,finished_at=_now(),
                runs=records,main_run_dir=str(main_run),smoke_run_dir=str(smoke_run))
            status();json_file(root/'summary.json',state)
            return state,root
    except BaseException as error:
        import traceback
        (root/'exception.log').write_text(traceback.format_exc())
        state.update(status='failed',completed=False,active_child_pid=None,error_type=type(error).__name__,error=str(error));status()
        raise
