"""Single-entrypoint NA launch and sequential, one-GPU count sweep."""
from datetime import datetime, timezone
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import torch
import yaml

from data.data import partition_metadata
from data.na_sfl_data import balanced_partitions, load_local_cifar10
from utils.deit_artifacts import csv_file, gpu_snapshot, json_file
from utils.logger import get_logger
from utils.na_sfl_config import (DEFAULTS, PROTO_METHODS, NO_AUX_METHODS, SD_ONLY_METHOD,
    HYBRID_METHODS, PIPELINE_METHOD, comparison_configs)
from utils.na_sfl_artifacts import (register_checkpoint, render_run, render_suite, target_summary,
                                   discard_failed_checkpoint, sweep_status, render_comparison)
from utils.na_sfl_train import NASFLTrainer, learning_contract
from utils.sflv2_metrics import snapshot_sources, seal_manifest


def validate_training_gpu(gpu, *, allow_unprofiled=False):
    supported = any(name in gpu['name'] for name in ('RTX A5000', 'Quadro RTX 5000'))
    if (not supported and not allow_unprofiled) or gpu['compute_processes'] or gpu['free_bytes'] < .8 * gpu['total_bytes']:
        raise RuntimeError('requested training GPU is unsupported, busy or has insufficient free memory')


def run_na_sfl(args, comm):
    if comm.Get_size() != 1:
        raise ValueError('NA requires one process with live Client/Server threads')
    if getattr(args, 'v3_bn_ablation', False):
        from utils.na_sfl_bn_experiments import run_bn_ablation
        return run_bn_ablation(args)
    if getattr(args, 'hybrid_experiment', False):
        from utils.na_sfl_hybrid_experiment import run_hybrid_experiment
        return run_hybrid_experiment(args)
    if getattr(args, 'sd_only_experiment', False):
        from utils.na_sfl_sd_only_experiment import run_sd_only_experiment
        return run_sd_only_experiment(args)
    if getattr(args, 'pipeline_experiment', False):
        from utils.na_sfl_pipeline_experiment import run_pipeline_experiment
        return run_pipeline_experiment(args)
    if args.comparison:
        return run_comparison(args)
    if args.sweep:
        return run_sweep(args)
    config = args.config
    started = time.perf_counter()
    contract = learning_contract(config)
    experiment_id = 'na_%s_seed%d_k%d_%s' % (config['method'], config['seed'], config['aux_client_count'],
        hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:10])
    run_id = experiment_id + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    root = Path(config['result_path']) / run_id
    root.mkdir(parents=True)
    json_file(root / 'status.json', {'status': 'running', 'completed': False})
    trainer = None
    try:
        gpu = gpu_snapshot(config['device'])
        json_file(root / 'gpu_before.json', gpu)
        validate_training_gpu(gpu, allow_unprofiled=config['allow_unprofiled_gpu'])
        sources = snapshot_sources(root)
        (root / 'source_snapshot_or_git_commit.txt').write_text(
            'Source files: source_snapshot/; SHA256 map: environment.json and manifest.json.\n')
        json_file(root / 'config.json', config)
        (root / 'config.yaml').write_text(yaml.safe_dump({k: config[k] for k in DEFAULTS}, sort_keys=True))
        data = load_local_cifar10(config['data_path'])
        if len(data[0]) != 50000 or len(data[2]) != 10000:
            raise ValueError('canonical CIFAR10 splits required')
        parts = balanced_partitions(data[1], config['n_clients'], config['samples_per_client'],
            config['dirichlet_alpha'], config['seed'], config['partition'])
        partition = {'ids': [p.tolist() for p in parts],
            'labels_by_client': [data[1][p].tolist() for p in parts],
            'metadata': partition_metadata(data[1], parts, config['partition'], config['dirichlet_alpha'], config['seed']),
            'allocation': 'class-wise Dirichlet preferences + IPF capacity balancing + integer residual allocation',
            'probability_floor': 1e-12, 'selected_subset_policy': 'seeded uniform global subset before allocation',
            'samples_per_client': config['samples_per_client'], 'drop_last': False}
        json_file(root / 'partition.json', partition)
        partition_hash = hashlib.sha256(json.dumps(partition['ids'], separators=(',', ':')).encode()).hexdigest()
        histogram = np.stack([np.bincount(data[1][p], minlength=10) for p in parts])
        csv_file(root / 'class_distribution.csv', [{'client_id': cid, **{f'class{cls}': int(row[cls]) for cls in range(10)}}
                                                   for cid, row in enumerate(histogram)])
        json_file(root / 'environment.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'container': os.environ.get('NA_SFL_CONTAINER_NAME'),
            'gpu': gpu, 'execution': 'existing live_async Client threads/streams + single FCFS Server thread/stream',
            'allow_unprofiled_gpu': config['allow_unprofiled_gpu'],
            'network': config['network'] or 'none; memory tensors only', 'precision': 'FP32; TF32 off', 'pretrained': False,
            'client_phase_targets': config['client_compute_seconds'],
            'partial_batch_pacing': 'target times scaled by samples/batch_size; assumed, not measured hardware',
            'server_policy': config['server_policy'], 'server_max_batch_samples': config['server_max_batch_samples'],
            'auxiliary_policy': 'no Aux heads for v3/v4 or bare SplitFedV2; all heads train/FedAvg for NA v1/v2',
            'prototype_policy': config['prototype'] if config['method'] in PROTO_METHODS else None,
            'server_bn_policy': config['bn_policy'],
            'random_sd': config['random_sd'] if config['method'] in (*HYBRID_METHODS, SD_ONLY_METHOD) else None,
            'pipeline': config['pipeline'],
            'sd_loss_weight': 1. if config['method'] == SD_ONLY_METHOD else None,
            'residual_policy': config['v4'] if config['method'] == 'na_sfl_v4_residual' else None,
            'prototype_client_inflight': (2 if config['method'] == PIPELINE_METHOD else 1)
                if config['method'] in PROTO_METHODS else None,
            'queue_policy': 'bounded detached CUDA SD packets; storage backpressure is not cut-gradient waiting',
            'augmentation_rng': 'per-round/client/epoch/batch seed; CPU fork/restore protected by one loader lock',
            'fcfs_order': 'actual readiness; same seed does not guarantee identical order across counts',
            'optimizer_policy': 'Client and Aux reset after each round; Server persists',
            'batchnorm_fedavg': 'floating buffers mean; integer counters max',
            'wall_clock_scope': 'initial model broadcast through each round evaluation; includes TCP FedAvg/model traffic; '
                                'shared GPU with assumed Client phase targets, not independent-device measurement',
            'gpu_regression_suite_executed': False, 'source_sha256': sources})
        logger = get_logger(root / 'training.log')
        trainer = NASFLTrainer(config, root, sources, logger, resume=args.resume)
        if config['network'] is not None:
            from utils.na_sfl_network import MininetNetwork
            network_context = MininetNetwork(config['network'], root / 'network', rounds=config['rounds'])
        else:
            network_context = nullcontext(None)
        with network_context as network:
            trainer.network = network
            trainer.run(*data, parts, stop_after_rounds=args.stop_after_rounds, max_wall_time_s=args.max_wall_time_s)
        if config['smoke_test']:
            saved = torch.load(trainer.last_checkpoint, map_location='cpu', weights_only=False)
            def finite_tree(value):
                if isinstance(value, torch.Tensor):
                    return bool(torch.isfinite(value).all().item())
                if isinstance(value, dict):
                    return all(finite_tree(v) for v in value.values())
                if isinstance(value, (list, tuple)):
                    return all(finite_tree(v) for v in value)
                return True
            if not finite_tree(saved):
                raise ValueError('nonfinite Smoke checkpoint')
            registration = {'retained': False, 'finite': True,
                'sha256': hashlib.sha256(trainer.last_checkpoint.read_bytes()).hexdigest(),
                'bytes': trainer.last_checkpoint.stat().st_size,
                'scope': 'Own temporary Smoke checkpoint verified and discarded; retained models preserved'}
            del saved
            trainer.last_checkpoint.unlink()
        else:
            registration = register_checkpoint(root, config, trainer.last_checkpoint, experiment_id,
                                               final_saved_at=trainer.final_saved_at)
        render_run(root, trainer.rounds, config)
        summary = {'status': 'completed' if trainer.next_round == config['rounds'] else 'stopped_at_round_boundary',
            'completed': trainer.next_round == config['rounds'], 'completed_segment': True,
            'experiment_id': experiment_id, 'run_id': run_id, 'run_dir': str(root),
            'seed': config['seed'], 'method': config['method'], 'aux_client_count': config['aux_client_count'],
            'server_policy': config['server_policy'], 'server_max_batch_samples': config['server_max_batch_samples'],
            'aux_client_ids': config['aux_client_ids'], 'auxiliary_heads': 0 if config['method'] in NO_AUX_METHODS else config['n_clients'],
            'completed_rounds': trainer.next_round, 'target_rounds': config['rounds'],
            'samples_seen': sum(row['samples_seen'] for row in trainer.rounds), 'server_updates': trainer.server.version,
            'server_samples_seen': sum(row['server_samples_seen'] for row in trainer.rounds),
            'server_sd_processed': sum(row['server_sd_processed'] for row in trainer.rounds),
            'initial_evaluation': trainer.initial_evaluation, 'initial_model_hashes': trainer.initial_hashes,
            'partition_sha256': partition_hash, 'data_sha256': trainer.data_sha256,
            'network_rate_scenario_sha256': network.scenario['sha256'] if network is not None else None,
            'final_test_accuracy': trainer.rounds[-1]['test_accuracy'],
            'best_test_accuracy': max(row['test_accuracy'] for row in trainer.rounds),
            'target': target_summary(trainer.rounds, config['target_accuracy'],
                                     initial_evaluation=trainer.initial_evaluation),
            'checkpoint': registration, 'launch_wall_s_debug_only': time.perf_counter() - started,
            'simulated_time_s': None, 'tta_evaluated': config['target_accuracy'] is not None,
            'elapsed_training_through_evaluation_s': trainer.rounds[-1]['elapsed_wall_s'],
            'gpu_regression_suite_executed': False,
            'training_scope': 'full local data' if config['max_batches_per_client'] is None else 'truncated requested training stage'}
        if config['method'] in PROTO_METHODS:
            summary['server_batchnorm'] = trainer.server.batchnorm.summary()
        if config['method'] == SD_ONLY_METHOD:
            summary['server_batchnorm'] = {'policy': 'sd_only',
                'running_counters': trainer.rounds[-1]['server_bn_counters'],
                'proto_forward_calls': 0, 'global_prototype_updates': 0, 'sd_loss_weight': 1.}
        if config['method'] in PROTO_METHODS:
            from utils.na_sfl_proto_artifacts import write_proto_artifacts
            summary.update(write_proto_artifacts(root, trainer.rounds, trainer.job_records,
                trainer.aggregation_records, config=config, initial_evaluation=trainer.initial_evaluation, render=True))
        if config['method'] == PIPELINE_METHOD:
            from utils.na_sfl_pipeline_artifacts import write_pipeline_artifacts
            summary.update(write_pipeline_artifacts(root, trainer.rounds, trainer.job_records,
                config=config, initial_evaluation=trainer.initial_evaluation, render=True))
        json_file(root / 'summary.json', summary)
        json_file(root / 'status.json', {'status': summary['status'], 'completed': summary['completed'], 'completed_segment': True})
        seal_manifest(root, sources, kind='na_sfl_live_async_convergence')
        return summary, root
    except BaseException as error:
        import traceback
        (root / 'exception.log').write_text(traceback.format_exc())
        source_root = Path(__file__).resolve().parents[1]
        if (root / 'checkpoints' / 'final.pt').exists():
            if config['smoke_test']:
                (root / 'checkpoints' / 'final.pt').unlink()
            else:
                discard_failed_checkpoint(root, source_root / 'results' / 'checkpoint_registry.json',
                                          source_root / 'results')
        json_file(root / 'status.json', {'status': 'failed', 'completed': False,
            'error_type': type(error).__name__, 'error': str(error)})
        raise


def run_comparison(args):
    """Four independent subprocess runs; same entrypoint, inputs and policies."""
    config = args.config
    configs = comparison_configs(config)
    root = Path(config['result_path']) / ('comparison_seed%d_%s_%s' % (config['seed'],
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), uuid4().hex[:8]))
    root.mkdir(parents=True)
    records = []
    json_file(root / 'status.json', {'status': 'running', 'completed': False})
    try:
        for condition in configs:
            name = condition['method'] + '_' + condition['server_policy']
            destination = root / name
            destination.mkdir()
            input_path = destination / 'input.yaml'
            input_path.write_text(yaml.safe_dump({key: condition[key] for key in DEFAULTS}, sort_keys=True))
            command = [sys.executable, '-u', 'run_exp_main.py', '--config', str(input_path),
                       '--result-path', str(destination)]
            if args.stop_after_rounds is not None:
                command += ['--stop-after-rounds', str(args.stop_after_rounds)]
            if args.max_wall_time_s is not None:
                command += ['--max-wall-time-s', str(args.max_wall_time_s)]
            with (root / (name + '.log')).open('w') as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if child.returncode:
                raise RuntimeError(f'comparison condition {name} failed; see its log')
            summaries = list(destination.glob('*/summary.json'))
            if len(summaries) != 1:
                raise RuntimeError('expected exactly one result per comparison condition')
            record = json.loads(summaries[0].read_text())
            if records and (record['partition_sha256'] != records[0]['partition_sha256']
                    or record['network_rate_scenario_sha256'] != records[0]['network_rate_scenario_sha256']
                    or any(record['initial_model_hashes'][key] != records[0]['initial_model_hashes'][key]
                           for key in ('client', 'server'))):
                raise RuntimeError('comparison data or initial Client/Server differs')
            records.append(record)
            json_file(root / 'runs.json', records)
            csv_file(root / 'comparison.csv', [{key: item[key] for key in
                ('method', 'server_policy', 'aux_client_count', 'completed_rounds', 'status',
                 'final_test_accuracy', 'best_test_accuracy', 'server_updates', 'server_sd_processed',
                 'elapsed_training_through_evaluation_s', 'run_dir')} for item in records])
            render_comparison(root, records)
        summary = {**sweep_status(records), 'seed': config['seed'], 'runs': records,
                   'tta_evaluated': config['target_accuracy'] is not None, 'gpu_regression_suite_executed': False}
        json_file(root / 'summary.json', summary)
        json_file(root / 'status.json', {**sweep_status(records), 'conditions_finished': len(records)})
        return summary, root
    except BaseException as error:
        json_file(root / 'status.json', {'status': 'failed', 'completed': False,
            'conditions_finished': len(records), 'error': str(error)})
        raise


def run_sweep(args):
    config = args.config
    root = Path(config['result_path']) / ('suite_seed%d_%s_%s' % (config['seed'],
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), uuid4().hex[:8]))
    root.mkdir(parents=True)
    records = []
    json_file(root / 'status.json', {'status': 'running', 'requested_counts': args.client_counts})
    for count in args.client_counts:
        destination = root / f'k{count}'
        command = [sys.executable, '-u', 'run_exp_main.py', 'na-sfl-v1', '--config', str(root / 'input.yaml'),
                   '--aux-client-count', str(count), '--result-path', str(destination)]
        (root / 'input.yaml').write_text(yaml.safe_dump({key: config[key] for key in DEFAULTS}, sort_keys=True))
        if args.stop_after_rounds is not None:
            command += ['--stop-after-rounds', str(args.stop_after_rounds)]
        if args.max_wall_time_s is not None:
            command += ['--max-wall-time-s', str(args.max_wall_time_s)]
        with (root / f'k{count}.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            json_file(root / 'status.json', {'status': 'failed', 'failed_count': count, 'completed_counts': [r['aux_client_count'] for r in records]})
            raise RuntimeError(f'NA count{count} failed; see {root / ("k" + str(count) + ".log")}')
        summaries = list(destination.glob('*/summary.json'))
        if len(summaries) != 1:
            raise RuntimeError('sweep expected exactly one child summary')
        item = json.loads(summaries[0].read_text())
        if records and (item['partition_sha256'] != records[0]['partition_sha256']
                        or item['initial_model_hashes'] != records[0]['initial_model_hashes']):
            raise RuntimeError('sweep partition/model initialization identity mismatch')
        records.append(item)
        json_file(root / 'runs.json', records)
        csv_file(root / 'comparison.csv', [{key: item[key] for key in
            ('aux_client_count', 'completed_rounds', 'status', 'final_test_accuracy', 'best_test_accuracy', 'run_dir')}
                                         for item in records])
        render_suite(root, records)
        print(f"count={count} rounds={item['completed_rounds']} accuracy={item['final_test_accuracy']}", flush=True)
    summary = {**sweep_status(records), 'seed': config['seed'], 'counts': args.client_counts,
               'runs': records, 'tta_evaluated': False, 'gpu_regression_suite_executed': False}
    json_file(root / 'summary.json', summary)
    json_file(root / 'status.json', {**sweep_status(records), 'completed_counts': args.client_counts})
    return summary, root
