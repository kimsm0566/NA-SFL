"""CIFAR10 round coordinator using the existing live_async FCFS workers."""
import hashlib
from collections import Counter
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import random
import threading
import time

import numpy as np
import torch

from data.data import build_loader
from models.auxiliary import AuxiliaryHead
from models.model import FullModel, ClientModel, ServerModel, canonical_state, load_matching_state, model_layout
from utils.deit_async import run_async_round, async_trace_sha256
from utils.deit_artifacts import atomic_checkpoint, capture_rng, cpu_tree, csv_file, json_file, restore_rng
from utils.na_sfl import NASFLClient, average_round_models
from utils.na_sfl_batch import validate_server_batches
from utils.sflv2 import SFLV2Trainer, SFLV2Server, _optimizer, sflv2_fp32
from utils.sflv2_metrics import check_sources_unchanged
from utils.na_sfl_config import PROTO_METHODS


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode() + str(array.dtype).encode() + str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def learning_contract(config):
    return {key: value for key, value in config.items() if key not in ('rounds', 'result_path')}


def validate_feature_memory_boundary(saved):
    """Feature memory and model history must describe the same drained round."""
    if saved['config'].get('method') not in PROTO_METHODS:
        return
    count = saved['next_round']
    extra = saved['server_extra_state']
    memory = extra['memory']
    expected = {tuple(row['job_id']) for row in saved['job_records'] if row['job_id'][0] == count}
    seen = [tuple(job) for job in extra['seen_jobs']]
    if (extra['server_version'] != saved['server_version']
            or memory['version'] != count or memory['round_id'] != count
            or memory['active'] is not False or len(seen) != len(set(seen)) or set(seen) != expected):
        raise ValueError('feature memory does not match completed model checkpoint boundary')
    if saved['config']['method'] == 'na_sfl_v4_residual':
        bank = extra['residual_bank']
        if (bank['version'] != count or bank['round_id'] != count
                or bank['bank_round_id'] != count or bank['active'] is not False):
            raise ValueError('residual bank does not match completed model checkpoint boundary')


def validate_resume_metadata(saved):
    count = saved['next_round']
    jobs = saved['job_records']
    batches = saved.get('server_batch_records')
    if batches is not None:
        steps = validate_server_batches(jobs, batches, saved['config']['server_max_batch_samples'],
                                        saved['config']['server_policy'])
        if batches and batches[0]['server_version_before'] != 0:
            raise ValueError('shared Server history must start at version zero')
    else:
        steps = len(jobs)
    if (type(count) is not int or count < 1
            or [row['round'] for row in saved['rounds']] != list(range(1, count + 1))
            or len(saved['execution_boundaries']) != count or len(saved['aggregation_records']) != count
            or saved['server_version'] != steps
            or len(saved['client_versions']) != saved['config']['n_clients']
            or sum(saved['client_versions']) != len(jobs)
            or len(saved['client_optimizers']) != saved['config']['n_clients']
            or any(optimizer['state'] for optimizer in saved['client_optimizers'])
            or any(optimizer['state'] for optimizer in saved['auxiliary_optimizers'])):
        raise ValueError('inconsistent NA checkpoint counts, versions or reset optimizer state')
    for number, boundary in enumerate(saved['execution_boundaries'], 1):
        own = [row for row in jobs if row['round'] == number]
        if (boundary.get('round') != number or boundary.get('success') is not True
                or boundary.get('queue_empty') is not True or boundary.get('streams_completed') is not True
                or any(boundary.get(key) != 0 for key in ('unresolved_futures', 'workers_alive', 'pending_graphs'))
                or boundary.get('trace_sha256') != async_trace_sha256(own)
                or any(boundary['runtimecounts'].get(key) != len(own)
                       for key in ('admitted_jobs', 'serviced_jobs', 'completed_jobs'))
                or [row['arrival_seq'] for row in own] != list(range(len(own)))):
            raise ValueError('inconsistent NA trace or undrained execution boundary')
        if batches is not None:
            own_batches = [row for row in batches if row['round'] == number]
            if (boundary['runtimecounts'].get('serviced_batches') != len(own_batches)
                    or boundary.get('server_batch_trace_sha256') != async_trace_sha256(own_batches)):
                raise ValueError('inconsistent Server batch boundary')
        transport = boundary.get('transport')
        if transport and any(transport.get(key) != 0 for key in ('workers_alive', 'pending_sd', 'pending_gradients')):
            raise ValueError('TCP transport not drained at checkpoint boundary')
    if batches is None and ([job['server_version_before'] for job in jobs] != list(range(len(jobs)))
            or [job['server_version_after'] for job in jobs] != list(range(1, len(jobs) + 1))):
        raise ValueError('noncontiguous shared server update versions')
    validate_feature_memory_boundary(saved)


class NASFLTrainer(SFLV2Trainer):
    """Reuse native split nodes, live actor engine, data-loader and evaluation.

    The base supplies seeding/evaluation. Optional Client pacing and actual
    Mininet TCP preserve the same model ownership and optimizer policies.
    """
    isolated_augmentation = True

    def __init__(self, config, run_dir, sources, logger, resume=None):
        from utils.client import require_cuda_device
        self.args = dict(config)
        self.run_dir, self.sources, self.logger = Path(run_dir), sources, logger
        self.device = require_cuda_device(config['device'])
        self.scheduler_rng = random.Random(self._seed('scheduler'))
        self.augmentation_lock = threading.Lock()
        self.job_records, self.rounds, self.aggregation_records, self.execution_boundaries = [], [], [], []
        self.server_batch_records, self.network = [], None
        self.next_round, self.last_async_boundary = 0, None
        self.partitions, self.data_sha256, self.initial_evaluation = None, None, None
        self.client_optimizer_config = {**config, **config['client_optimizer']}
        self.server_optimizer_config = {**config, **config['server_optimizer']}
        self.aux_optimizer_config = {**config, **config['aux_optimizer']}
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[self.device.index]), sflv2_fp32():
            torch.random.default_generator.manual_seed(config['seed'])
            torch.cuda.manual_seed(config['seed'])
            full = FullModel(config['model']).to(self.device)
            state = canonical_state(full)
            del full
            server = ServerModel(config['model'], config['cut']).to(self.device)
            load_matching_state(server, state)
            if config['method'] in PROTO_METHODS:
                from utils.na_sfl_prototype import PrototypeServer
                boundary = model_layout(config['model'], config['cut'])['boundary_shape_per_sample']
                self.server = PrototypeServer(server, {**self.server_optimizer_config,
                    'boundary_shape_per_sample': boundary})
            else:
                self.server = SFLV2Server(server, self.server_optimizer_config)
            self.clients = []
            for cid in range(config['n_clients']):
                model = ClientModel(config['model'], config['cut']).to(self.device)
                load_matching_state(model, state)
                head = None
                if config['method'] not in ('splitfedv2', *PROTO_METHODS):
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(self._seed('auxiliary_initialization'))
                        channels = model_layout(config['model'], config['cut'])['boundary_shape_per_sample'][0]
                        head = AuxiliaryHead(channels, 10).to(self.device)
                if config['method'] in PROTO_METHODS:
                    from utils.na_sfl_proto_client import PrototypeClient
                    node = PrototypeClient(cid, model, self.client_optimizer_config,
                        prototype_mode=cid in config['slow_client_ids'])
                else:
                    node = NASFLClient(cid, model, self.client_optimizer_config, head=head,
                        head_config=self.aux_optimizer_config,
                        mode='aux' if cid in config['aux_client_ids'] else 'server')
                self.clients.append(node)
            del state
        self.initial_hashes = {'client': model_digest(self.clients[0].model),
            'server': model_digest(self.server.model),
            'auxiliary': model_digest(self.clients[0].auxiliary_head)
                if self.clients[0].auxiliary_head is not None else None}
        if resume:
            self._restore(resume)

    def _restore(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved.get('schema') != 'na_sfl_live_round_v2'
                or saved['config'] != learning_contract(self.args)
                or saved['source_sha256'] != self.sources
                or saved['next_round'] > self.args['rounds']):
            raise ValueError('NA resume config/source/round contract mismatch')
        validate_resume_metadata(saved)
        self.server.model.load_state_dict(saved['server_model'])
        self.server.optimizer.load_state_dict(saved['server_optimizer'])
        self.server.version = saved['server_version']
        if self.args['method'] in PROTO_METHODS:
            self.server.load_extra_state_dict(saved['server_extra_state'])
        for cid, node in enumerate(self.clients):
            node.model.load_state_dict(saved['client_model'])
            node.version = saved['client_versions'][cid]
            node.optimizer.load_state_dict(saved['client_optimizers'][cid])
            if node.auxiliary_head is not None:
                head_state = saved['auxiliary_models'][0 if self.args['auxiliary_fedavg'] else cid]
                node.auxiliary_head.load_state_dict(head_state)
                node.auxiliary_optimizer.load_state_dict(saved['auxiliary_optimizers'][cid])
        self.next_round = saved['next_round']
        for key in ('partitions', 'data_sha256', 'initial_evaluation', 'initial_hashes', 'job_records',
                    'rounds', 'aggregation_records', 'execution_boundaries', 'server_batch_records'):
            setattr(self, key, saved[key])
        self.last_async_boundary = self.execution_boundaries[-1]
        restore_rng(saved['rng'], self.scheduler_rng)

    def _log(self, message):
        self.logger.info(message)

    def _client_batches(self, images, labels, sample_ids, round_id, client_id):
        index = 0
        for epoch in range(self.args['local_epochs']):
            ids = np.asarray(sample_ids)
            generator = torch.Generator().manual_seed(self._seed('shuffle', round_id, client_id, epoch))
            if not self.args['augmentation']:
                ids = ids[torch.randperm(len(ids), generator=generator).numpy()]
            loader = build_loader(images[ids], labels[ids], ids, self.args['batch_size'],
                                  self.args['augmentation'], 0, generator)
            iterator = iter(loader)
            while True:
                cap = self.args['max_batches_per_client']
                if cap is not None and index >= cap:
                    return
                # torchvision transforms use CPU-global Torch RNG. Protect the
                # complete next(loader) fork/restore, never a model/CUDA phase.
                with self.augmentation_lock, torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(
                        self._seed('augmentation', round_id, client_id, epoch, index))
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                yield index, batch
                index += 1

    def _aggregate(self, weights, round_id):
        if any(client.pending is not None for client in self.clients):
            raise RuntimeError('cannot aggregate pending local graphs')
        server_version_before = self.server.version
        aggregate_started = time.perf_counter()
        upload_wall = broadcast_wall = 0.
        heads = [node.auxiliary_head for node in self.clients if node.auxiliary_head is not None]
        if getattr(self, 'network', None) is None:
            compute_started = time.perf_counter()
            average_round_models([node.model for node in self.clients], heads, weights,
                                 auxiliary_fedavg=self.args['auxiliary_fedavg'])
            compute_wall = time.perf_counter() - compute_started
        else:
            from models.model import average_states
            upload_started = time.perf_counter()
            uploaded = self.network.transfer_models(self._model_packets(), 'upload', f'round{round_id}_models')
            upload_wall = time.perf_counter() - upload_started
            compute_started = time.perf_counter()
            client_state = average_states([{k[7:]: v for k, v in item.items() if k.startswith('client:')}
                                           for item in uploaded], weights)
            aux_states = [{k[4:]: v for k, v in item.items() if k.startswith('aux:')} for item in uploaded]
            if heads and self.args['auxiliary_fedavg']:
                aux_states = [average_states(aux_states, weights)] * len(self.clients)
            broadcast = [{**{'client:' + k: v for k, v in client_state.items()},
                          **{'aux:' + k: v for k, v in state.items()}} for state in aux_states]
            compute_wall = time.perf_counter() - compute_started
            broadcast_started = time.perf_counter()
            self._load_model_packets(self.network.transfer_models(broadcast, 'download', f'round{round_id}_fedavg'))
            broadcast_wall = time.perf_counter() - broadcast_started
        reset_started = time.perf_counter()
        for node in self.clients:
            node.optimizer = _optimizer(node.model, self.client_optimizer_config)
            if node.auxiliary_head is not None:
                node.auxiliary_optimizer = _optimizer(node.auxiliary_head, self.aux_optimizer_config)
        self.aggregation_records.append({'round': round_id, 'weights': weights,
            'model_upload_wall_s': upload_wall, 'fedavg_compute_wall_s': compute_wall,
            'model_broadcast_wall_s': broadcast_wall,
            'optimizer_reset_wall_s': time.perf_counter() - reset_started,
            'aggregation_wall_s': time.perf_counter() - aggregate_started,
            'fedavg_completed_perf_counter_s': time.perf_counter(),
            'auxiliary_fedavg': self.args['auxiliary_fedavg'],
            'server_version_before': server_version_before, 'server_version_after': self.server.version,
            'client_optimizer_states_after_reset': [len(node.optimizer.state) for node in self.clients],
            'aux_optimizer_states_after_reset': [len(node.auxiliary_optimizer.state) for node in self.clients
                                                if node.auxiliary_head is not None]})

    def _model_packets(self):
        result = []
        for node in self.clients:
            item = {'client:' + key: value for key, value in canonical_state(node.model).items()}
            if node.auxiliary_head is not None:
                item.update({'aux:' + key: value for key, value in canonical_state(node.auxiliary_head).items()})
            result.append(item)
        return result

    def _load_model_packets(self, packets):
        for node, state in zip(self.clients, packets):
            node.model.load_state_dict({k[7:]: v for k, v in state.items() if k.startswith('client:')})
            if node.auxiliary_head is not None:
                node.auxiliary_head.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith('aux:')})

    def _validate_jobs(self, jobs, parts, round_id):
        expected_per_client = math.ceil(self.args['samples_per_client'] / self.args['batch_size']) * self.args['local_epochs']
        if self.args['max_batches_per_client'] is not None:
            expected_per_client = min(expected_per_client, self.args['max_batches_per_client'])
        expected = len(self.clients) * expected_per_client
        if (len(jobs) != expected or len({tuple(job['job_id']) for job in jobs}) != expected
                or [job['arrival_seq'] for job in jobs] != list(range(expected))):
            raise RuntimeError('missing/duplicate SD or FCFS service ordering violation')
        for cid, part in enumerate(parts):
            own = [job for job in jobs if job['client_id'] == cid]
            if len(own) != expected_per_client:
                raise RuntimeError('wrong local/SD batch count')
            expected_mode = 'aux' if cid in self.args['aux_client_ids'] else 'server'
            if any(job['gradient_mode'] != expected_mode or
                   job['auxiliary_updated'] != (self.args['method'] not in ('splitfedv2', *PROTO_METHODS)) or
                   (job['gradient_bytes'] > 0) != (expected_mode == 'server') or
                   job['client_version_after'] != job['client_version_before'] + 1 for job in own):
                raise RuntimeError('wrong gradient mode, auxiliary update or local version')
            if self.args['method'] in PROTO_METHODS:
                kind = 'PROTO_JOB' if cid in self.args['slow_client_ids'] else 'SD_JOB'
                if any(job.get('job_kind') != kind for job in own):
                    raise RuntimeError('wrong SD/Prototype upload for Client mode')
            visited = [sample for job in own for sample in job['sample_ids']]
            if self.args['max_batches_per_client'] is None:
                frequencies = Counter(visited)
                if len(visited) != len(part) * self.args['local_epochs'] or any(
                        frequencies[int(sample)] != self.args['local_epochs'] for sample in part):
                    raise RuntimeError('round sample coverage mismatch')
            elif len(set(visited)) != len(visited) or not set(visited) <= set(part.tolist()):
                raise RuntimeError('invalid truncated-stage sample ownership')
        return expected

    def checkpoint_payload(self):
        if (not self.last_async_boundary or not self.last_async_boundary['success']
                or any(node.pending is not None or node.optimizer.state for node in self.clients)):
            raise RuntimeError('checkpoint requires drained round and reset local optimizers')
        heads = [node.auxiliary_head for node in self.clients if node.auxiliary_head is not None]
        if self.args['auxiliary_fedavg']:
            heads = heads[:1]
        return {'schema': 'na_sfl_live_round_v2', 'config': learning_contract(self.args),
            'server_extra_state': self.server.extra_state_dict() if self.args['method'] in PROTO_METHODS else None,
            'source_sha256': self.sources, 'next_round': self.next_round,
            'client_model': canonical_state(self.clients[0].model),
            'server_model': canonical_state(self.server.model),
            'auxiliary_models': [canonical_state(head) for head in heads],
            'server_optimizer': cpu_tree(self.server.optimizer.state_dict()),
            'client_optimizers': [cpu_tree(node.optimizer.state_dict()) for node in self.clients],
            'auxiliary_optimizers': [cpu_tree(node.auxiliary_optimizer.state_dict()) for node in self.clients
                                     if node.auxiliary_head is not None],
            'client_versions': [node.version for node in self.clients], 'server_version': self.server.version,
            'rng': capture_rng(self.scheduler_rng), 'partitions': self.partitions,
            'data_sha256': self.data_sha256, 'initial_evaluation': self.initial_evaluation,
            'initial_hashes': self.initial_hashes, 'job_records': self.job_records,
            'rounds': self.rounds, 'aggregation_records': self.aggregation_records,
            'execution_boundaries': self.execution_boundaries, 'server_batch_records': self.server_batch_records}

    def _progress_files(self):
        csv_file(self.run_dir / 'round_metrics.csv', self.rounds)
        csv_file(self.run_dir / 'job_timings.csv', self.job_records)
        json_file(self.run_dir / 'round_metrics.json', self.rounds)
        json_file(self.run_dir / 'aggregation.json', self.aggregation_records)
        json_file(self.run_dir / 'execution_boundaries.json', self.execution_boundaries)
        json_file(self.run_dir / 'server_batches.json', self.server_batch_records)
        csv_file(self.run_dir / 'server_batches.csv', self.server_batch_records)
        if self.args['method'] in PROTO_METHODS:
            from utils.na_sfl_proto_artifacts import write_proto_artifacts
            write_proto_artifacts(self.run_dir, self.rounds, self.job_records, self.aggregation_records,
                                  config=self.args, initial_evaluation=self.initial_evaluation)

    def run(self, train_images, train_labels, test_images, test_labels, partitions,
            stop_after_rounds=None, max_wall_time_s=None):
        parts = [np.asarray(part) for part in partitions]
        ids = [part.tolist() for part in parts]
        digest = hashlib.sha256()
        for value in (train_images, train_labels, test_images, test_labels):
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(str(ids).encode())
        signature = digest.hexdigest()
        if self.data_sha256 is not None and (signature != self.data_sha256 or ids != self.partitions):
            raise ValueError('resume data/partition identity differs')
        self.partitions, self.data_sha256 = ids, signature
        segment_origin = time.perf_counter()
        previous_elapsed = self.rounds[-1].get('elapsed_wall_s', 0.) if self.rounds else 0.
        if self.network is not None:
            self.network.apply_round(min(self.next_round + 1, self.args['rounds']))
            self._load_model_packets(self.network.transfer_models(self._model_packets(), 'download',
                                                                  f'segment{self.next_round}_initial'))
        if self.initial_evaluation is None:
            accuracy, loss, _ = self._evaluate(test_images, test_labels)
            self.initial_evaluation = {'test_accuracy': accuracy, 'test_loss': loss,
                'elapsed_wall_s': previous_elapsed + time.perf_counter() - segment_origin}
        stop = self.args['rounds'] if stop_after_rounds is None else min(
            self.args['rounds'], self.next_round + stop_after_rounds)
        segment_start = time.perf_counter()
        for round_id in range(self.next_round + 1, stop + 1):
            check_sources_unchanged(self.sources)
            network_rates = self.network.apply_round(round_id) if self.network is not None else None
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(self.device)
            for node in self.clients:
                node.model.train()
                if node.auxiliary_head is not None:
                    node.auxiliary_head.train()
            self.server.model.train()
            if self.args['method'] in PROTO_METHODS:
                self.server.start_round(round_id)
            server_before = self.server.version
            with sflv2_fp32():
                jobs = run_async_round(self, train_images, train_labels, parts, round_id)
            expected = self._validate_jobs(jobs, parts, round_id)
            batch_count = validate_server_batches(jobs, self.last_async_batches,
                self.args['server_max_batch_samples'], self.args['server_policy'])
            if self.server.version != server_before + batch_count:
                raise RuntimeError('Server optimizer step count differs from processed batches')
            self.server_batch_records.extend(self.last_async_batches)
            self.execution_boundaries.append(dict(self.last_async_boundary))
            training_wall = time.perf_counter() - started
            worker_drain_elapsed = time.perf_counter() - self.last_async_boundary.get('wall_origin_perf_counter_s', started)
            samples = sum(job['samples'] for job in jobs)
            weights = [sum(job['samples'] for job in jobs if job['client_id'] == cid)
                       for cid in range(len(parts))]
            self._aggregate(weights, round_id)
            prototype_stats = None
            if self.args['method'] in PROTO_METHODS:
                memory_started = time.perf_counter()
                prototype_stats = self.server.finish_round()
                prototype_stats['feature_memory_update_wall_s'] = time.perf_counter() - memory_started
                prototype_stats['memory_swap_completed_perf_counter_s'] = time.perf_counter()
            accuracy, loss, _ = self._evaluate(test_images, test_labels)
            def aux_loss(mode):
                own = [job for job in jobs if job['gradient_mode'] == mode and job['aux_loss'] is not None]
                count = sum(job['samples'] for job in own)
                return sum(job['aux_loss'] * job['samples'] for job in own) / count if count else None
            row = {'round': round_id, 'seed': self.args['seed'], 'method': self.args['method'],
                'aux_client_count': self.args['aux_client_count'], 'aux_client_ids': self.args['aux_client_ids'],
                'test_accuracy': accuracy, 'test_ce_loss': loss,
                'train_server_ce_loss': sum(job['loss'] * job['samples'] for job in jobs) / samples,
                'train_aux_ce_loss_server_mode': aux_loss('server'),
                'train_aux_ce_loss_aux_mode': aux_loss('aux'),
                'samples_seen': samples, 'client_updates': len(jobs),
                'aux_updates': sum(job['auxiliary_updated'] for job in jobs),
                'server_updates': self.server.version - server_before,
                'server_sd_processed': len(jobs), 'total_expected_sd': expected,
                'server_updates_cumulative': self.server.version,
                'server_policy': self.args['server_policy'],
                'network_rates': network_rates,
                'server_batch_samples': [batch['samples'] for batch in self.last_async_batches],
                'elapsed_wall_s': previous_elapsed + time.perf_counter() - segment_origin,
                'training_wall_s_debug_only': training_wall,
                'worker_drain_elapsed_s': worker_drain_elapsed,
                'round_wall_s_debug_only': time.perf_counter() - started,
                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device),
                'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device),
                'simulated_time_s': None}
            if prototype_stats is not None:
                row['global_prototype_stats'] = prototype_stats
            self.rounds.append(row)
            self.next_round = round_id
            self._progress_files()
            self._log('round_complete ' + str(row))
            if max_wall_time_s is not None and time.perf_counter() - segment_start >= max_wall_time_s:
                break
        if not self.rounds:
            raise ValueError('run has no completed rounds')
        checkpoint = self.run_dir / 'checkpoints' / 'final.pt'
        atomic_checkpoint(checkpoint, self.checkpoint_payload())
        self.final_saved_at = datetime.now(timezone(timedelta(hours=9))).isoformat()
        self.last_checkpoint = checkpoint
        return self.rounds
