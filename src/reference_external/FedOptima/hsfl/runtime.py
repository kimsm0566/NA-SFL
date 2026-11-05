"""Opt-in HSFL environment around the author's FedOptima learning engine.

There is one GPU process with original AsyncClient workers, an original
AsyncServerModel, counter scheduler and TCP Message frames. Adaptations concern
model/data, physical namespace links, visit-based evaluation and cancellation.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime
from functools import partial
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import queue
import socket
import subprocess
import sys
import threading
import time
import traceback
from zoneinfo import ZoneInfo

import torch

from config import Config
from communication.bandwidth_limiter import BandwidthLimiter
from communication.message import Message
from node.async_client import AsyncClient
from node.async_server_model import AsyncServerModel
from hsfl.control import ExperimentControl
from hsfl.model_data import make_loaders, make_models
from hsfl.network import StreamingNetwork
from hsfl.protocol import (CancelledTraining, FrozenClientComm, FrozenServerComm,
                           StoppableScheduler, WireAudit, freeze_payload, recv_exact, local_pipe)
from utils.deit_artifacts import atomic_checkpoint, cpu_tree, csv_file, json_file
from utils.na_sfl_artifacts import retain_final_checkpoint
from utils.na_sfl_network import NETWORK_DEFAULTS


GPU_UUID = 'GPU-029a0a4c-92bf-adf8-8c9f-2b082a20dd50'
SRC = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger('fedoptima.hsfl')


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(key.encode() + str(value.dtype).encode() + str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def source_map():
    files = [p for p in REPO.rglob('*.py') if '.hsfl_deps' not in p.parts]
    for directory in ('models', 'data', 'utils'):
        files += list((SRC / directory).glob('*.py'))
    return {str(p.relative_to(SRC)): digest(p) for p in sorted(set(files))}


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return True


class MatchedServerModel(AsyncServerModel):
    def attach(self, experiment):
        self.experiment = experiment
        self.pending_samples = None
        original_step = self.optimiser.step
        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.experiment.server_stream.synchronize()
            self.experiment.server_steps += 1
            for samples, origin in self.pending_samples:
                self.experiment.server_serviced[origin] += int(samples)
            return result
        self.optimiser.step = step

    def receive(self, batch_size):
        control = self.experiment.control
        if control.poll_stop():
            raise CancelledTraining()
        if control.evaluation_due():
            self.experiment.evaluate(self)
            if control.poll_stop():
                raise CancelledTraining()
        self.connect.send(batch_size)
        while not self.connect.poll(.1):
            if control.poll_stop():
                raise CancelledTraining()
        kind, data = pickle.loads(self.connect.recv())
        if kind == 'hsfl_stop' or control.poll_stop():
            raise CancelledTraining()
        if kind == 'train':
            self.pending_samples = data[2]
        return kind, data


class MatchedExperiment:
    def __init__(self, directory, *, smoke=False):
        self.directory = Path(directory).resolve()
        if not self.directory.is_relative_to(SRC/'results'):
            raise ValueError('experiment output must remain inside src/results')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.smoke = smoke
        self.control = ExperimentControl(max_wall_s=240 if smoke else 18000)
        self.done = threading.Event()
        self.errors = queue.Queue()
        self.status_lock = threading.RLock()
        self.threads = []
        self.clients = []
        self.client_comms = []
        self.client_steps = [0] * 10
        self.client_partial = [0] * 10
        self.client_phase = [Counter() for _ in range(10)]
        self.server_steps = 0
        self.server_serviced = [0] * 10
        self.network = None
        self.scheduler = None
        self.audit = None
        self.model = None
        self.sources = source_map()
        self.config_record = dict(method='fedoptima', seed=42, device=1, gpu_uuid=GPU_UUID,
            model='resnet18_cifar', cut=2, n_clients=10, samples_per_client=3000,
            partition='capacity_constrained_dirichlet', dirichlet_alpha=.1,
            batch_size=128, partial_batch_size=56, optimizer='adam', lr=.001,
            weight_decay=.0005, betas=[.9,.999], eps=1e-8,
            label_on_device=False, auxiliary='original_ResNet_aux_adjusted_for_cut2',
            async_agg=True, agg_all_models=False, sort_algo='counter',
            act_buffer=1, act_parallel_num=1, aggregation_frequency=24,
            activation_upload_credit=1,
            activation_credit_policy='skip redundant ACT_CHECK while an SD upload is queued/inprogress; continue local Aux',
            local_optimizer_policy='author_persistent_across_async_aggregation',
            server_optimizer_policy='persistent',
            client_compute_seconds={'forward':.3, 'backward':.6, 'auxiliary':.1},
            compute_pacing='forward0.3 + combined_Aux_forward/backward/Client_backward/steps0.7; partial proportional',
            stop={'rolling_window':10, 'strict_mean_above':70,
                  'max_training_wall_s':self.control.max_wall_s, 'fixed_round_cap':None},
            round_definition='30000 completed local sample visits; no global client barrier',
            evaluation='current global Client+current Server on all10000 official Test samples',
            early_stop_plateau=False, smoke_test=smoke, precision='FP32', tf32=False,
            shared_gpu_processes=1, client_threads=10,
            network={**NETWORK_DEFAULTS, 'round_rate_variation':.1, 'rate_seed':42,
                     'send_buffer_sd':1})
        json_file(self.directory / 'config.json', self.config_record)
        json_file(self.directory / 'source_sha256.json', self.sources)
        for relative in self.sources:
            destination = self.directory / 'source_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((SRC / relative).read_bytes())

    def spawn(self, name, function):
        def guarded():
            try:
                with torch.cuda.device(0):
                    function()
            except CancelledTraining:
                pass
            except BaseException:
                detail = dict(worker=name, traceback=traceback.format_exc())
                self.errors.put(detail)
                LOGGER.error('%s\n%s', name, detail['traceback'])
                self.control.request_stop('worker_failure')
        thread = threading.Thread(name='fedoptima-' + name, target=guarded, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread

    def status(self, status='running', **extra):
        with self.status_lock:
            return self._status(status, **extra)

    def _status(self, status='running', *, persist=True, **extra):
        snapshot = self.control.snapshot()
        evaluations = snapshot.pop('evaluations')
        result = dict(status=status, completed=status=='completed', updated_at=now(),
            pid=os.getpid(), gpu_uuid=GPU_UUID, smoke_test=self.smoke,
            rounds_completed=len(evaluations), latest_evaluation=evaluations[-1] if evaluations else None,
            **snapshot, client_steps=list(self.client_steps), client_partial_batches=list(self.client_partial),
            server_steps=self.server_steps, server_serviced_samples=list(self.server_serviced),
            gpu_allocated_bytes=torch.cuda.memory_allocated(0),
            gpu_reserved_bytes=torch.cuda.memory_reserved(0),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0), **extra)
        if persist:
            json_file(self.directory / 'status.json', result)
            json_file(self.directory / 'round_metrics.json', evaluations)
            if evaluations:
                csv_file(self.directory / 'round_metrics.csv', evaluations)
        return result

    def evaluate(self, model):
        started = time.monotonic()
        before = self.control.snapshot()
        model.client_net.load_state_dict(model.global_client_model)
        model.client_net.to('cuda:0').eval()
        model.net.eval()
        correct, total, loss_sum = 0, 0, 0.
        with torch.no_grad():
            for images, targets in self.test_loader:
                if self.control.poll_stop():
                    return
                images, targets = images.to('cuda:0'), targets.to('cuda:0')
                logits = model.net(model.client_net(images))
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError('nonfinite test logits')
                loss = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
                loss_sum += float(loss.item())
                correct += int((logits.argmax(1) == targets).sum().item())
                total += len(targets)
        self.server_stream.synchronize()
        if total != 10000:
            raise RuntimeError('Test evaluation did not cover10000 samples')
        row = self.control.record_evaluation(correct, total, test_ce_loss=loss_sum/total,
            evaluation_wall_s=time.monotonic()-started, visits_at_evaluation_start=before['total_visits'],
            server_steps=self.server_steps, server_serviced_samples=list(self.server_serviced),
            global_client_version=model.global_client_version)
        LOGGER.info('EVAL round=%s accuracy=%s visits=%s server_steps=%s mean10=%s elapsed=%.3f',
            row['round'], row['accuracy'], row['actual_visits'], self.server_steps,
            self.control.snapshot()['latest10mean'], row['elapsed_s'])
        self.status()
        if self.smoke and row['round'] >= 2 and min(self.client_partial) >= 1:
            self.control.request_stop('smoke_complete')

    def pace_client(self, client, loader, stream):
        phase = {}
        original_forward = client.net.forward
        def forward(inputs):
            if self.control.poll_stop():
                raise CancelledTraining()
            started = time.monotonic()
            result = original_forward(inputs)
            stream.synchronize()
            measured = time.monotonic()-started
            self.control.stop_event.wait(max(0., .3 * len(inputs)/128 - measured))
            self.client_phase[client.config.CLIENT_INDEX]['forward_actual_s'] += measured
            return result
        client.net.forward = forward
        original_aux = client.auxiliary_net.forward
        def auxiliary(inputs):
            phase['started'] = time.monotonic()
            return original_aux(inputs)
        client.auxiliary_net.forward = auxiliary
        original_step = client.optimiser.step
        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            stream.synchronize()
            size = loader.last_batch_size
            measured = time.monotonic()-phase['started']
            self.control.stop_event.wait(max(0., .7 * size/128 - measured))
            cid = client.config.CLIENT_INDEX
            self.client_phase[cid]['aux_backward_steps_actual_s'] += measured
            self.client_steps[cid] += 1
            self.client_partial[cid] += size==56
            self.control.record_visits(cid, size)
            return result
        client.optimiser.step = step

    def client_worker(self, cid, cfg, sock):
        stream = torch.cuda.Stream(device=0)
        with torch.cuda.stream(stream):
            client = AsyncClient(cfg)
            client.socket.close()  # Node's unused default-namespace socket.
            client.socket = sock
            client.receive_net()
            if any(p.device.type != 'cpu' for p in client.net.parameters()):
                raise RuntimeError('initial model frame was not restored on CPU')
            client.set_data_loader = lambda: setattr(client, 'data_loader', self.loaders[cid])
            comm = FrozenClientComm(sock, config=cfg, timing=client.client_timing,
                                    stop_event=self.control.stop_event)
            self.client_comms.append(comm)
            client.set_comm(comm)
            self.clients.append((cid, client))
            self.pace_client(client, self.loaders[cid], stream)
            sender = self.spawn('client%d-sender' % cid, comm.run_sender)
            self.spawn('client%d-receiver' % cid, comm.run_receiver)
            try:
                client.train()
            finally:
                stream.synchronize()
                comm.send(Message(title='CLOSE_COMMUNICATION_CHANNEL', close_comm=True))
                sender.join(45)
                if sender.is_alive():
                    raise RuntimeError('Client sender failed to drain')

    def watchdog(self):
        while not self.done.wait(.2):
            stopped = self.control.poll_stop()
            for comm in self.client_comms:
                if stopped:
                    comm.stop_signal = True
            if not stopped:
                desired = self.control.snapshot()['total_visits']//30000+1
                active = self.network.stats()['active_round']
                while active < desired:
                    self.network.apply_live_round(active+1)
                    active += 1

    def install_wire_audit(self):
        original_send, original_recv = Message.send_to, BandwidthLimiter.recv
        def send(message, sock):
            amount = original_send(message, sock)
            self.audit.sent(sock, amount+4, message.msg_type)
            return amount
        def receive(limiter, sock, size):
            return recv_exact(sock, size, on_chunk=lambda count:self.audit.received(sock,count))
        Message.send_to, BandwidthLimiter.recv = send, receive
        return original_send, original_recv

    def train_network(self):
        self.network.apply_live_round(1)
        pairs = self.network.create_connections()
        for pair in pairs:
            for sock in pair:
                sock.settimeout(90)
        self.audit = WireAudit(pairs)
        original_send, original_recv = self.install_wire_audit()
        compute, scheduling = local_pipe()
        comm = FrozenServerComm([server for _, server in pairs])
        def receiver_error(cid, error):
            self.errors.put(dict(worker='server-input-%d'%cid, traceback=repr(error)))
            self.control.request_stop('worker_failure')
        comm.on_receiver_error = receiver_error
        cfg = Config(experiment_name='hsfl_fedoptima_matched', client_num=10,
            dataset_name='CIFAR-10', model_name='ResNet18', data_size=3000,
            test_data_size=10000, batch_size=128, layer_num_on_client=2,
            aggregation_frequency=24, learning_rate=.001, weight_decay=.0005, optim='adam',
            label_on_device=False, simulate_bandwidth=False, server_cpu_num=2, client_cpu_num=2,
            max_val_step=sys.maxsize, non_improve_step=sys.maxsize)
        cfg.server_init()
        self.model = MatchedServerModel('cuda:0', self.models['server'], self.models['client'],
            self.models['auxiliary_net'], torch.nn.CrossEntropyLoss(), compute, cfg,
            None, None, comm)
        self.model.global_client_model = freeze_payload(self.model.global_client_model)
        self.model.global_auxiliary_model = freeze_payload(self.model.global_auxiliary_model)
        self.server_stream = torch.cuda.Stream(device=0)
        self.model.net.to('cuda:0')
        self.server_stream.wait_stream(torch.cuda.current_stream())
        self.model.attach(self)
        self.scheduler = StoppableScheduler(10, scheduling, comm, 3000, 'counter',
            act_check=True, act_buffer=1, timing=self.model.server_timing,
            async_agg=True, agg_all_models=False, act_parallel_num=1,
            stop_event=self.control.stop_event)
        drained = False
        try:
            self.spawn('server-sender', comm.run_sender)
            self.spawn('server-receiver', comm.run_receiver)
            self.spawn('scheduler-put', self.scheduler.put)
            self.spawn('scheduler-get', self.scheduler.get)
            self.control.start()  # Include initial model transfer in the5h clock.
            self.training_started_at = now()
            self.spawn('watchdog', self.watchdog)
            client_workers = []
            for cid, (sock, _) in enumerate(pairs):
                local = deepcopy(cfg)
                local.CLIENT_INDEX = cid
                local.client_init()
                client_workers.append(self.spawn('client%d' % cid,
                    partial(self.client_worker, cid, local, sock)))
            for _, server in pairs:
                Message(title='MSG_INITIAL_MODEL',
                    content=[self.models['client'], self.models['auxiliary_net']]).send_to(server)
            def train_server():
                with torch.cuda.stream(self.server_stream):
                    self.model.server_timing.start()
                    try:
                        self.model.train()
                    finally:
                        self.server_stream.synchronize()
                        self.model.server_timing.stop()
            self.spawn('server-compute', train_server)
            last_status = 0.
            while not self.control.poll_stop():
                if time.monotonic()-last_status >= 10:
                    value = self.status()
                    LOGGER.info('PROGRESS visits=%s rounds=%s server_steps=%s allocated_GiB=%.3f',
                        value['total_visits'], value['rounds_completed'], self.server_steps,
                        value['gpu_allocated_bytes']/2**30)
                    last_status = time.monotonic()
                time.sleep(.2)
            self.stop_requested_at = now()
            self.cleanup_started = time.monotonic()
            client_deadline = time.monotonic()+45
            for worker in client_workers:
                worker.join(max(0., client_deadline-time.monotonic()))
            for cid in range(10):
                comm.send(Message(title='CLOSE_COMMUNICATION_CHANNEL', close_comm=True), cid)
            self.done.set()
            deadline = time.monotonic()+45
            for worker in self.threads:
                worker.join(max(0., deadline-time.monotonic()))
            alive = [t.name for t in self.threads if t.is_alive()]
            if alive:
                raise RuntimeError('worker shutdown deadline exceeded: '+str(alive))
            if not self.errors.empty():
                raise RuntimeError('worker error: '+json.dumps(list(self.errors.queue)))
            drained = True
        finally:
            self.done.set()
            self.control.request_stop('runtime_cleanup')
            for pair in pairs:
                for sock in pair:
                    try: sock.shutdown(socket.SHUT_RDWR)
                    except OSError: pass
                    sock.close()
            if not drained:
                for cid in range(10):
                    comm.send(Message(title='ABORT_CHANNEL', close_comm=True), cid)
                for client_comm in self.client_comms:
                    client_comm.stop_signal = True
                    client_comm.send(Message(title='ABORT_CHANNEL', close_comm=True))
                deadline = time.monotonic()+5
                for worker in self.threads:
                    worker.join(max(0.,deadline-time.monotonic()))
            compute.close(); scheduling.close()
            Message.send_to, BandwidthLimiter.recv = original_send, original_recv

    def save_final(self):
        torch.cuda.synchronize(0)
        final = dict(config=self.config_record, control=self.control.snapshot(),
            global_client=cpu_tree(self.model.global_client_model),
            server=cpu_tree(self.model.net.state_dict()),
            global_auxiliary=cpu_tree(self.model.global_auxiliary_model),
            server_optimizer=cpu_tree(self.model.optimiser.state_dict()),
            clients={cid:dict(model=cpu_tree(c.net.state_dict()), auxiliary=cpu_tree(c.auxiliary_net.state_dict()),
                optimizer=cpu_tree(c.optimiser.state_dict()),
                auxiliary_optimizer=cpu_tree(c.auxiliary_optimiser.state_dict())) for cid,c in self.clients},
            server_version=self.server_steps, source_sha256=self.sources,
            limitation='timeout can end inside a global30000-visit round; checkpoint is final, not exact-resume validated')
        if not finite_tree(final):
            raise RuntimeError('nonfinite final checkpoint')
        checkpoint = self.directory/'checkpoints/final.pt'
        checkpoint.parent.mkdir(exist_ok=True)
        atomic_checkpoint(checkpoint, final)
        restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not finite_tree(restored):
            checkpoint.unlink()
            raise RuntimeError('nonfinite saved checkpoint')
        entry = dict(experiment_id='fedoptima_'+self.directory.relative_to(SRC/'results').as_posix().replace('/','_'),
            run_dir=str(self.directory), path=str(checkpoint),
            bytes=checkpoint.stat().st_size, sha256=digest(checkpoint), final_saved_at=now(),
            scope='FedOptima asynchronous matched environment final; all roles bundled',
            method='fedoptima', seed=42, aux_client_count=10)
        if self.smoke:
            checkpoint.unlink()
            json_file(self.directory/'checkpoint_discard.json', dict(**entry, discarded_after_finite_validation=True))
        else:
            registry = retain_final_checkpoint(SRC/'results/checkpoint_registry.json', entry, SRC/'results')
            json_file(self.directory/'checkpoint_retention.json', dict(registered=entry, registry=registry))
            for item in registry['final_checkpoints']:
                path = Path(item['path'])
                if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
                    raise RuntimeError('retained checkpoint identity mismatch')
            document = SRC.parent/'docs/10-24/01-storage-policy/README.md'
            text = document.read_text()
            start = text.index('## 현재 보관 목록')
            end = text.index('\n## ',start+3)
            section = '## 현재 보관 목록\n\n'+now()+' 기준 최신 모델이다. FedOptima 완료 후 실제 파일/hash를 대조했다.\n\n'
            section += '| 실험 ID | 최종 저장 완료 시각 | 체크포인트 | Bytes | SHA-256 |\n|---|---|---|---:|---|\n'
            for item in registry['final_checkpoints']:
                relative = Path(item['path']).relative_to(SRC.parent).as_posix()
                section += f"| {item['experiment_id']} | {item['final_saved_at']} | [final.pt](../../../{relative}) | {item['bytes']} | {item['sha256']} |\n"
            section += '\n[전체 보관 대장·정리 이력](../../../src/results/checkpoint_registry.json). 기존 원시 로그/지표/config/snapshot은 보존한다.\n'
            document.write_text(text[:start]+section+text[end:])
        return entry

    def run(self):
        started = time.monotonic()
        if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
            raise RuntimeError('matched experiment must expose only physical GPU1 UUID')
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('exactly one CUDA GPU is required')
        info = subprocess.check_output(['nvidia-smi','-i',GPU_UUID,
            '--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        fields = [x.strip() for x in info.split(',')]
        if float(fields[2])/float(fields[3])<.8 or float(fields[4])>0:
            raise RuntimeError('GPU1 is busy or has insufficient free memory: '+info)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(42)
        self.models = make_models()
        self.initial_hashes = {name:model_digest(model) for name,model in self.models.items()}
        self.loaders, self.test_loader, data = make_loaders()
        json_file(self.directory/'data_identity.json', data)
        json_file(self.directory/'initial_models.json', self.initial_hashes)
        json_file(self.directory/'environment.json', dict(gpu=info, python=sys.version,
            torch=torch.__version__, cuda=torch.version.cuda, pid=os.getpid(), shared_gpu=True))
        # tqdm is output-only. Disable carriage-return spam from eleven actors.
        import node.async_client as author_client
        import node.async_server_model as author_server
        from tqdm import tqdm
        author_client.tqdm.tqdm = partial(tqdm, disable=True)
        author_server.tqdm = partial(tqdm, disable=True)
        try:
            with StreamingNetwork(self.config_record['network'], self.directory/'network') as net:
                self.network = net
                self.train_network()
            if any(digest(SRC/p)!=sha for p,sha in self.sources.items()):
                raise RuntimeError('training sources changed during experiment')
            if self.network.stats()['cleanup_errors']:
                raise RuntimeError('network cleanup error')
            counters = self.scheduler.counters()
            counters['local_samples_not_uploaded'] = [self.control.snapshot()['client_visits'].get(cid,0)-count
                for cid,count in enumerate(counters['received_samples'])]
            counters['local_jobs_not_uploaded'] = [steps-count for steps,count in zip(self.client_steps,counters['received_jobs'])]
            if min(counters['local_samples_not_uploaded'])<0 or min(counters['local_jobs_not_uploaded'])<0:
                raise RuntimeError('Server received SD without a completed local batch')
            if counters['selected_samples'] != self.server_serviced:
                # A selected but unserviced final batch is an explicit stop tail.
                counters['selected_but_unserviced_samples'] = [a-b for a,b in zip(counters['selected_samples'],self.server_serviced)]
            wire = self.audit.snapshot()
            if any(item['sent']!=item['received'] for directions in wire.values() for item in directions.values()):
                raise RuntimeError('TCP sent/received framed byte mismatch after drain')
            state = self.control.snapshot()
            if self.smoke and (state['reason']!='smoke_complete' or len(state['evaluations'])<2
                               or min(self.client_partial)<1 or min(self.server_serviced)<1):
                raise RuntimeError('short actual training did not meet all completion gates')
            if not self.smoke and state['reason'] not in ('wall_time_limit','rolling_accuracy'):
                raise RuntimeError('main training stopped outside authorized termination conditions')
            checkpoint = self.save_final()
            cleanup_s = time.monotonic()-self.cleanup_started
            self.status('finalizing')
            status = self._status('completed', persist=False, training_started_at=self.training_started_at,
                stop_requested_at=self.stop_requested_at, cleanup_wall_s=cleanup_s,
                launch_wall_s=time.monotonic()-started, sources_unchanged=True,
                network_closed=self.network.stats()['closed'], checkpoint=checkpoint,
                checkpoint_discarded=self.smoke)
            json_file(self.directory/'communication.json', dict(wire=wire, scheduler=counters,
                client_compute=[dict(x) for x in self.client_phase],
                sender_credits=[dict(client_id=c.config.CLIENT_INDEX, pending=c.sd_pending,
                    peak_pending=c.sd_peak_pending, rejected_send_attempt_jobs=c.sd_skipped_jobs,
                    rejected_send_attempt_samples=c.sd_skipped_samples) for c in self.client_comms],
                server_serviced_samples=self.server_serviced, server_steps=self.server_steps))
            json_file(self.directory/'summary.json', status)
            artifacts = {str(p.relative_to(self.directory)):digest(p)
                         for p in self.directory.rglob('*') if p.is_file()
                         and p.name not in ('artifact_manifest.json','status.json')}
            json_file(self.directory/'artifact_manifest.json', artifacts)
            json_file(self.directory/'status.json', status)
            LOGGER.info('COMPLETED %s', json.dumps(status))
            return status
        except BaseException:
            detail = traceback.format_exc()
            (self.directory/'exception.log').write_text(detail)
            self.control.request_stop('runtime_failure')
            self.status('failed', traceback=detail)
            raise


def run(*, result_path=None, smoke=False):
    if result_path is None:
        result_path = SRC/'results/20261105_fedoptima_matched'/('smoke' if smoke else 'main')
    return MatchedExperiment(result_path, smoke=smoke).run()
