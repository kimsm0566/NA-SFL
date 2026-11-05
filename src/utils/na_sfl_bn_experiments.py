"""Conditional v3 loss/BN experiments dispatched by the existing CLI only."""
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import yaml

from utils.deit_artifacts import csv_file, gpu_snapshot, json_file
from utils.na_sfl_config import DEFAULTS, load_config
from utils.na_sfl_network import rate_scenario
from utils.sflv2_metrics import check_sources_unchanged, sha256, snapshot_sources


def _now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


@contextmanager
def controlled_termination():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('conditional experiment supervisor received SIGTERM')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def wait_stage_process(child, on_progress):
    try:
        while True:
            try:
                return child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                on_progress()
    except BaseException:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        raise


@contextmanager
def owned_stage_process(command, **kwargs):
    child = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        yield child
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()


def last10_mean(rows):
    if (len(rows) != 100 or any(type(row.get('round')) is not int for row in rows)
            or [row['round'] for row in rows] != list(range(1, 101))):
        raise ValueError('branch decision requires complete consecutive rounds1..100')
    values = []
    for row in rows:
        raw = row.get('test_accuracy')
        try:
            value = Decimal(str(raw))
        except (InvalidOperation, TypeError):
            raise ValueError('invalid accuracy for conditional experiment') from None
        if isinstance(raw, bool) or not value.is_finite() or not 0 <= value <= 100:
            raise ValueError('invalid accuracy for conditional experiment')
        values.append(value)
    return sum(values[-10:], Decimal(0)) / 10


def decision_from_rounds(baseline_rows, loss_rows, *, stage_a_completed=True):
    if stage_a_completed is not True:
        raise ValueError('failed/incomplete Stage A cannot select a BN branch')
    baseline, loss = last10_mean(baseline_rows), last10_mean(loss_rows)
    improved = loss > baseline
    sd, proto = (.8, .2) if improved else (.5, .5)
    return {'decision_metric': 'mean(test_accuracy_rounds_91_to_100)',
            'baseline_last10_mean': float(baseline), 'loss_08_02_last10_mean': float(loss),
            'baseline_last10_mean_exact': str(baseline), 'loss_08_02_last10_mean_exact': str(loss),
            'baseline_rounds_91_to_100': [float(row['test_accuracy']) for row in baseline_rows[-10:]],
            'loss_08_02_rounds_91_to_100': [float(row['test_accuracy']) for row in loss_rows[-10:]],
            'improved': improved, 'selected_fast_sd_weight': sd, 'selected_fast_proto_weight': proto,
            'gate_condition': 'loss_08_02_last10_mean > baseline_last10_mean',
            'comparison_arithmetic': 'exact Decimal of recorded accuracy; no rounding before comparison',
            'interpretation': 'single-seed setting selection, not statistical superiority', 'decided_at': _now()}


def execute_stages(launch, baseline_rows, root, on_finished=None):
    root = Path(root)
    first = launch('v3_loss_08_02_shared_bn', .8, .2, 'shared_original')
    decision = decision_from_rounds(baseline_rows, first['rows'],
                                   stage_a_completed=first['summary']['completed'])
    if first.get('run_dir'):
        path = Path(first['run_dir']) / 'round_metrics.json'
        decision['stage_a_round_metrics'] = str(path)
        decision['stage_a_round_metrics_sha256'] = sha256(path)
    reference = root / 'baseline_reference.json'
    if reference.exists():
        decision['baseline_reference_sha256'] = sha256(reference)
    json_file(root / 'decision.json', decision)  # Must exist before either BN launch.
    records = [first]
    if on_finished:
        on_finished(records, decision)
    for name, policy in [('v3_bn_freeze_proto', 'sd_train_proto_eval_bn_only'),
                         ('v3_bn_dual', 'dual_running_stats_shared_affine')]:
        records.append(launch(name, decision['selected_fast_sd_weight'],
                              decision['selected_fast_proto_weight'], policy))
        if on_finished:
            on_finished(records, decision)
    return records, decision


def _read(path):
    return json.loads(Path(path).read_text())


def _record(name, run, summary, rows, config):
    crossing = next((row for row in rows if row['test_accuracy'] >= 70), None)
    bn = summary.get('server_batchnorm', {})
    return {'name': name, 'run_dir': str(run), 'fast_sd_weight': config['prototype']['fast_sd_weight'],
            'fast_proto_weight': config['prototype']['fast_proto_weight'],
            'bn_policy': config.get('bn_policy', 'shared_original'),
            'completed_rounds': summary['completed_rounds'],
            'final_accuracy': rows[-1]['test_accuracy'], 'best_accuracy': max(row['test_accuracy'] for row in rows),
            'last10_mean': float(last10_mean(rows)), 'last10_mean_exact': str(last10_mean(rows)),
            'tta70_s': crossing['elapsed_wall_s'] if crossing else None,
            'tta70_round': crossing['round'] if crossing else None,
            'elapsed_training_through_evaluation_s': rows[-1]['elapsed_wall_s'],
            'launch_wall_s_debug_only': summary['launch_wall_s_debug_only'],
            'server_updates': summary['server_updates'], 'gpu': 'Quadro RTX 5000', 'seed': config['seed'],
            'network_trace_id': summary['network_rate_scenario_sha256'], 'server_batchnorm': bn,
            'evaluation_input_domain': 'sd', 'evaluation_statistics_domain': bn.get('evaluation_statistics_domain', 'shared')}


def _render(root, baseline, records, decision):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = [baseline, *[item['record'] for item in records]]
    same_loss_reference = values[1] if decision['improved'] else baseline
    for value in values:
        reference = baseline if value['name'] in ('v3_original', 'v3_loss_08_02_shared_bn') else same_loss_reference
        value['delta_original_pp'] = float(Decimal(value['last10_mean_exact']) - Decimal(baseline['last10_mean_exact']))
        value['same_loss_shared_reference'] = reference['name']
        value['delta_same_loss_shared_pp'] = float(Decimal(value['last10_mean_exact']) - Decimal(reference['last10_mean_exact']))
    csv_file(root / 'comparison.csv', values)
    json_file(root / 'comparison.json', {'decision': decision, 'records': values,
        'scope': 'each completed condition100round; CPU rendering of actual GPU training results'})
    for filename, xkey, xlabel in [('accuracy_by_round.png', 'round', 'Round'),
                                   ('accuracy_by_elapsed.png', 'elapsed_wall_s', 'Elapsed time (hours)')]:
        fig, axis = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
        for value in values:
            rows = _read(Path(value['run_dir']) / 'round_metrics.json')
            x = [row[xkey] / (3600 if xkey == 'elapsed_wall_s' else 1) for row in rows]
            axis.plot(x, [row['test_accuracy'] for row in rows], linewidth=1.3,
                      label=value['name'] + f" ({value['fast_sd_weight']}:{value['fast_proto_weight']})")
        axis.axhline(70, color='grey', linestyle=':', linewidth=1)
        axis.set(xlabel=xlabel, ylabel='Test accuracy (%)', title='Conditional NA-SFLv3 loss / BN experiments')
        axis.grid(alpha=.2); axis.legend(fontsize=8)
        fig.savefig(root / filename, dpi=180); plt.close(fig)
    lines = ['# Conditional NA-SFLv3 results', '',
        'Gate: last10(Stage A) > last10(original); exact recorded decimals.', '',
        f"Improved: {decision['improved']}; BN weights: {decision['selected_fast_sd_weight']}:{decision['selected_fast_proto_weight']}.", '',
        '| Experiment | SD:Proto | BN | Final | Best | Last10 | Δ original(pp) | Δ same-loss shared(pp) | TTA70(s) | Training wall(s) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for value in values:
        lines.append(f"| {value['name']} | {value['fast_sd_weight']}:{value['fast_proto_weight']} | {value['bn_policy']} | "
            f"{value['final_accuracy']:.2f} | {value['best_accuracy']:.2f} | {value['last10_mean']:.3f} | "
            f"{value['delta_original_pp']:.3f} | {value['delta_same_loss_shared_pp']:.3f} | {value['tta70_s']} | "
            f"{value['elapsed_training_through_evaluation_s']:.3f} |")
    lines += ['', '![Accuracy by round](accuracy_by_round.png)', '',
              '![Accuracy by elapsed](accuracy_by_elapsed.png)', '',
              'TTA is the first evaluated70% crossing, not time to complete100round. Single seed; live FCFS ordering can differ.']
    (root / 'README.md').write_text('\n'.join(lines) + '\n')


def _validate_bn(summary, policy):
    bn = summary['server_batchnorm']
    calls = {'sd': 19200, 'proto': 24000}
    updates = {'sd': 19200, 'proto': 0 if policy == 'sd_train_proto_eval_bn_only' else 24000}
    if bn['policy'] != policy or bn['forward_calls'] != calls or bn['running_updates'] != updates:
        raise ValueError('completed BN policy/forward/update audit failed')
    expected = ({'sd': 43200} if policy == 'shared_original' else {'sd': 19200}
                if policy == 'sd_train_proto_eval_bn_only' else {'sd': 19200, 'proto': 24000})
    if len(bn['running_counters']) != 10 or any(row != expected for row in bn['running_counters'].values()):
        raise ValueError('actual BN tensor counters disagree with100round policy')


def _retention_audit(root):
    source = Path(__file__).resolve().parents[1]
    registry = _read(source / 'results/checkpoint_registry.json')
    for entry in registry['final_checkpoints']:
        path = Path(entry['path'])
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('retained checkpoint identity mismatch')
    json_file(root / 'checkpoint_registry_latest.json', registry)
    doc = source.parent / 'docs/10-24/01-storage-policy/README.md'
    if doc.exists():
        text = doc.read_text(); start = text.index('## 현재 보관 목록'); end = text.index('\n## ', start + 3)
        section = '## 현재 보관 목록\n\n' + _now() + ' 기준 최신 모델이다. 조건부 v3 실험 완료마다 실제 파일/hash를 대조한다.\n\n'
        section += '| 실험 ID | 최종 저장 완료 시각 | 체크포인트 | Bytes | SHA-256 |\n|---|---|---|---:|---|\n'
        for entry in registry['final_checkpoints']:
            relative = Path(entry['path']).relative_to(source.parent).as_posix()
            section += f"| {entry['experiment_id']} | {entry['final_saved_at']} | [final.pt](../../../{relative}) | {entry['bytes']} | {entry['sha256']} |\n"
        section += '\n[전체 보관 대장·정리 이력](../../../src/results/checkpoint_registry.json). 기존 원시 로그/지표/config/snapshot은 보존한다.\n'
        doc.write_text(text[:start] + section + text[end:])


def run_bn_ablation(args):
    """Only child calls perform training, each through run_exp_main.py."""
    from utils.na_sfl_run import validate_training_gpu
    config = args.config
    if (config['method'] != 'na_sfl_v3_proto' or config['rounds'] != 100
            or config['max_batches_per_client'] is not None or config['smoke_test']):
        raise ValueError('conditional experiments require full v3 100round training')
    source = Path(__file__).resolve().parents[1]
    baseline_run = Path(args.baseline_reference).resolve()
    baseline_summary = _read(baseline_run / 'summary.json')
    baseline_config = load_config(overrides={key: value for key, value in _read(baseline_run / 'config.json').items() if key in DEFAULTS})
    baseline_rows = _read(baseline_run / 'round_metrics.json')
    baseline_environment = _read(baseline_run / 'environment.json')
    baseline_manifest = _read(baseline_run / 'manifest.json')
    last10_mean(baseline_rows)
    if (not baseline_summary['completed'] or baseline_summary['completed_rounds'] != 100
            or baseline_config['method'] != 'na_sfl_v3_proto'
            or baseline_config['bn_policy'] != 'shared_original'
            or (baseline_config['prototype']['fast_sd_weight'], baseline_config['prototype']['fast_proto_weight']) != (.5, .5)):
        raise ValueError('reference must be original completed v3 0.5:0.5 shared BN')
    for filename in ('summary.json', 'config.json', 'environment.json', 'round_metrics.json', 'partition.json'):
        if sha256(baseline_run / filename) != baseline_manifest['artifacts'][filename]:
            raise ValueError('baseline artifact changed: ' + filename)
    for filename, digest in baseline_manifest['source_sha256'].items():
        if sha256(baseline_run / 'source_snapshot' / filename) != digest:
            raise ValueError('baseline source snapshot changed')
    expected = deepcopy(baseline_config)
    for key in ('result_path', 'device'):
        expected[key] = config[key]
    if config != expected:
        raise ValueError('suite config must match baseline; only result_path/logical device may differ')
    gpu = gpu_snapshot(config['device']); validate_training_gpu(gpu)
    if gpu['uuid'] != baseline_environment['gpu']['uuid']:
        raise ValueError('conditional experiment GPU differs from original v3; matched baseline required')
    scenario = rate_scenario(config['network'], 100)
    if scenario['sha256'] != baseline_summary['network_rate_scenario_sha256']:
        raise ValueError('conditional experiment network differs from original v3')
    root = Path(config['result_path'])
    if root == baseline_run or root.is_relative_to(baseline_run) or baseline_run.is_relative_to(root):
        raise ValueError('new experiment destination overlaps reference run')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'status.json').exists() or any((root / name).exists() for name in
            ('stage_a_loss_08_02', 'stage_b_bn_freeze_proto', 'stage_b_bn_dual')):
        raise ValueError('conditional destination already contains a launch')
    sources = snapshot_sources(root)
    (root / 'inputs').mkdir(exist_ok=True)
    reference = {'run_dir': str(baseline_run), 'summary': baseline_summary,
                 'last10_mean': float(last10_mean(baseline_rows)), 'last10_mean_exact': str(last10_mean(baseline_rows)),
                 'rounds_91_to_100': baseline_rows[-10:], 'gpu': gpu, 'network_trace_id': scenario['sha256'],
                 'input_sha256': {filename: sha256(baseline_run / filename) for filename in
                     ('summary.json', 'config.json', 'environment.json', 'round_metrics.json', 'partition.json')}}
    json_file(root / 'baseline_reference.json', reference)
    json_file(root / 'network_rate_scenario.json', scenario)
    json_file(root / 'source_sha256.json', sources)
    json_file(root / 'checkpoint_registry_before.json', _read(source / 'results/checkpoint_registry.json'))
    baseline = _record('v3_original', baseline_run, baseline_summary, baseline_rows, baseline_config)
    state = {'status': 'running', 'completed': False, 'started_at': _now(), 'supervisor_pid': os.getpid(),
             'gpu_uuid': gpu['uuid'], 'order': ['loss_08_02_shared_bn', 'bn_freeze_proto', 'bn_dual'],
             'completed_conditions': [], 'active_condition': None, 'records': []}
    def status():
        state['updated_at'] = _now(); json_file(root / 'status.json', state)
    status()
    destinations = {'v3_loss_08_02_shared_bn': 'stage_a_loss_08_02',
                    'v3_bn_freeze_proto': 'stage_b_bn_freeze_proto', 'v3_bn_dual': 'stage_b_bn_dual'}
    def launch(name, sd, proto, policy):
        check_sources_unchanged(sources)
        for filename, digest in reference['input_sha256'].items():
            if sha256(baseline_run / filename) != digest:
                raise ValueError('original baseline changed during suite')
        stage_config = deepcopy(baseline_config)
        stage_config.update(device=config['device'], bn_policy=policy, result_path=str(root / destinations[name]))
        stage_config['prototype'].update(fast_sd_weight=sd, fast_proto_weight=proto)
        stage_config = load_config(overrides={key: stage_config[key] for key in DEFAULTS})
        path = root / 'inputs' / (name + '.yaml')
        path.write_text(yaml.safe_dump({key: stage_config[key] for key in DEFAULTS}, sort_keys=True))
        delta = {'bn_policy': {'baseline': 'shared_original', 'new': policy},
                 'prototype.fast_sd_weight': {'baseline': .5, 'new': sd},
                 'prototype.fast_proto_weight': {'baseline': .5, 'new': proto}}
        command = [sys.executable, '-u', str(source / 'run_exp_main.py'), '--config', str(path)]
        json_file(root / (name + '_launch.json'), {'command': command, 'config_diff': delta,
            'config_sha256': sha256(path), 'gpu_uuid': gpu['uuid'], 'source_sha256': sources})
        state.update(active_condition=name, active_rounds_completed=0, active_run_dir=None)
        status()
        with (root / (name + '.console.log')).open('ab', buffering=0) as log, owned_stage_process(
                command, cwd=source, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL) as child:
            state['active_child_pid'] = child.pid; status()
            def progress():
                paths = list(Path(stage_config['result_path']).glob('*/round_metrics.json'))
                if len(paths) == 1:
                    try:
                        rows = _read(paths[0])
                        state.update(active_rounds_completed=len(rows), active_run_dir=str(paths[0].parent))
                    except (OSError, ValueError):
                        pass
                status()
            result = wait_stage_process(child, progress)
        state['active_child_pid'] = None; status()
        if result:
            raise RuntimeError(name + ' failed with exit ' + str(result))
        matches = list(Path(stage_config['result_path']).glob('*/summary.json'))
        if len(matches) != 1:
            raise ValueError('expected one stage summary')
        run = matches[0].parent; summary = _read(matches[0]); rows = _read(run / 'round_metrics.json')
        last10_mean(rows)
        if (not summary['completed'] or summary['completed_rounds'] != 100
                or summary['samples_seen'] != 3000000 or summary['server_updates'] != 24000):
            raise ValueError('stage did not complete full100round contract')
        for key in ('initial_model_hashes', 'partition_sha256', 'data_sha256', 'network_rate_scenario_sha256'):
            if summary[key] != baseline_summary[key]:
                raise ValueError('stage/baseline identity differs: ' + key)
        environment = _read(run / 'environment.json')
        if environment['gpu']['uuid'] != gpu['uuid'] or environment['source_sha256'] != sources:
            raise ValueError('stage GPU/source mismatch')
        _validate_bn(summary, policy)
        record = _record(name, run, summary, rows, stage_config)
        state['completed_conditions'].append(name); state['records'].append(record)
        state['active_rounds_completed'] = 100; status()
        _retention_audit(root)
        return {'name': name, 'run_dir': str(run), 'rows': rows, 'summary': summary, 'record': record}
    try:
        with controlled_termination():
            records, decision = execute_stages(launch, baseline_rows, root,
                on_finished=lambda records, decision: _render(root, baseline, records, decision))
        check_sources_unchanged(sources)
        state.update(status='completed', completed=True, active_condition=None,
                     finished_at=_now(), decision=decision)
        status(); json_file(root / 'summary.json', state)
        return state, root
    except BaseException as error:
        import traceback
        (root / 'exception.log').write_text(traceback.format_exc())
        state.update(status='failed', completed=False, error_type=type(error).__name__, error=str(error))
        status()
        raise
