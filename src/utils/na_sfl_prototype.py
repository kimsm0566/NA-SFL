"""NA-SFLv3/v4 class means, frozen round memory and one-job Shared Server.

The same server classifier handles SD and prototypes. Global feature memory is
not a model parameter and changes only at finish_round after all jobs drain.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from utils.sflv2 import (SFLV2Server, _finite, _gradients, _model_device,
                        _tensor, _timed, sflv2_fp32)


def _feature_shape(shape):
    shape = tuple(shape)
    if not shape or any(type(d) is not int or d < 1 for d in shape):
        raise ValueError('feature_shape must contain positive integer dimensions')
    return shape


def make_local_prototypes(z, labels):
    """Return differentiable class means and sorted present int64 class/counts.

    Backward through a returned mean distributes its gradient by 1/count to
    every member SD. No missing class is materialized.
    """
    if (not isinstance(z, torch.Tensor) or z.dtype != torch.float32 or z.ndim < 2
            or z.shape[0] < 1):
        raise ValueError('SD must be a nonempty FP32 feature tensor')
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
            or labels.device != z.device or labels.shape != (z.shape[0],)
            or (labels < 0).any().item()):
        raise ValueError('labels must be nonnegative int64, one per SD, on its device')
    _finite(z, 'SD')
    classes, counts = torch.unique(labels, sorted=True, return_counts=True)
    protos = torch.stack([z[labels == c].mean(dim=0) for c in classes])
    return protos, classes, counts


def validate_prototypes(protos, classes, counts, feature_shape, num_classes, actual_batch_size):
    """Reject malformed typed wire tensors, including partial-batch count errors."""
    feature_shape = _feature_shape(feature_shape)
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError('num_classes must be positive')
    if type(actual_batch_size) is not int or actual_batch_size < 1:
        raise ValueError('actual_batch_size must be positive')
    if (not isinstance(protos, torch.Tensor) or protos.dtype != torch.float32
            or tuple(protos.shape[1:]) != feature_shape or protos.ndim != len(feature_shape) + 1
            or not 1 <= protos.shape[0] <= num_classes):
        raise ValueError('prototypes must be nonempty FP32 and match boundary shape')
    size = protos.shape[0]
    for value, name in ((classes, 'classes'), (counts, 'counts')):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64
                or value.device != protos.device or value.shape != (size,)):
            raise ValueError(f'{name} must be int64 with one entry per prototype on its device')
    if ((classes < 0).any().item() or (classes >= num_classes).any().item()
            or (classes[1:] <= classes[:-1]).any().item()):
        raise ValueError('prototype classes must be unique, sorted and in range')
    if (counts <= 0).any().item() or int(counts.sum().item()) != actual_batch_size:
        raise ValueError('prototype counts must be positive and sum to actual batch size')
    _finite(protos, 'prototypes')


class GlobalPrototypeMemory:
    """Count-weighted all-job feature memory with a fixed snapshot per round."""

    def __init__(self, num_classes, feature_shape, *, device='cpu'):
        if type(num_classes) is not int or num_classes < 1:
            raise ValueError('num_classes must be positive')
        self.num_classes = num_classes
        self.feature_shape = _feature_shape(feature_shape)
        self.values = torch.zeros((num_classes, *self.feature_shape), device=device)
        self.valid = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.sample_counts = torch.zeros(num_classes, dtype=torch.int64, device=device)
        self.sums = torch.zeros_like(self.values)
        self.round_counts = torch.zeros_like(self.sample_counts)
        self.snapshot_values = self.values.clone()
        self.snapshot_valid = self.valid.clone()
        self.version = 0
        self.round_id = None
        self.active = False

    def start_round(self, round_id):
        if self.active:
            raise RuntimeError('previous prototype round is still active')
        if (type(round_id) is not int or round_id < 0
                or (self.round_id is not None and round_id <= self.round_id)):
            raise ValueError('round_id must increase monotonically')
        self.round_id, self.active = round_id, True
        self.snapshot_values = self.values.detach().clone()
        self.snapshot_valid = self.valid.clone()
        self.sums.zero_(); self.round_counts.zero_()

    def _require_active(self):
        if not self.active:
            raise RuntimeError('prototype operation requires an active round')

    @torch.no_grad()
    def accumulate(self, protos, classes, counts):
        self._require_active()
        validate_prototypes(protos, classes, counts, self.feature_shape, self.num_classes,
                            int(counts.sum().item()))
        classes = classes.to(self.values.device)
        counts = counts.to(self.values.device)
        weights = counts.to(self.values.dtype).reshape((-1,) + (1,) * len(self.feature_shape))
        self.sums.index_add_(0, classes, protos.detach().to(self.values.device) * weights)
        self.round_counts.index_add_(0, classes, counts)
        _finite(self.sums, 'global prototype accumulator')

    def alignment_loss(self, protos, classes, counts, reduction='mse'):
        self._require_active()
        if reduction not in ('mse', 'l2_squared'):
            raise ValueError('alignment reduction must be mse or l2_squared')
        valid = self.snapshot_valid.to(protos.device)[classes]
        if not valid.any().item():
            return protos.sum() * 0.
        target = self.snapshot_values.to(protos.device)[classes[valid]].detach()
        distances = (protos[valid] - target).square().flatten(1)
        distances = distances.mean(1) if reduction == 'mse' else distances.sum(1)
        weights = counts[valid].to(protos.dtype)
        return (distances * weights).sum() / weights.sum()

    @torch.no_grad()
    def finish_round(self):
        self._require_active()
        observed = self.round_counts > 0
        denominator = self.round_counts[observed].to(self.values.dtype).reshape(
            (-1,) + (1,) * len(self.feature_shape))
        self.values[observed] = self.sums[observed] / denominator
        self.valid |= observed
        self.sample_counts.copy_(self.round_counts)
        self.version += 1
        self.active = False
        _finite(self.values, 'global prototypes')
        return {'round': self.round_id, 'version': self.version,
            'valid_class_mask': self.valid.cpu().tolist(),
            'sample_counts_by_class': self.sample_counts.cpu().tolist(),
            'prototype_norm': self.values.flatten(1).norm(dim=1).cpu().tolist()}

    def state_dict(self):
        return {'num_classes': self.num_classes, 'feature_shape': self.feature_shape,
            'version': self.version, 'round_id': self.round_id, 'active': self.active,
            **{key: getattr(self, key).detach().cpu().clone() for key in
                ('values', 'valid', 'sample_counts', 'sums', 'round_counts',
                 'snapshot_values', 'snapshot_valid')}}

    def load_state_dict(self, state):
        if state['num_classes'] != self.num_classes or tuple(state['feature_shape']) != self.feature_shape:
            raise ValueError('prototype checkpoint layout differs')
        loaded = {}
        for key in ('values', 'valid', 'sample_counts', 'sums', 'round_counts',
                    'snapshot_values', 'snapshot_valid'):
            value, target = state[key], getattr(self, key)
            if not isinstance(value, torch.Tensor) or value.dtype != target.dtype or value.shape != target.shape:
                raise ValueError(f'invalid prototype checkpoint tensor: {key}')
            _finite(value, key)
            if key in ('sample_counts', 'round_counts') and (value < 0).any().item():
                raise ValueError('negative prototype checkpoint counts')
            loaded[key] = value.detach().to(target.device).clone()
        if (type(state['version']) is not int or state['version'] < 0
                or type(state['active']) is not bool
                or (state['round_id'] is not None and
                    (type(state['round_id']) is not int or state['round_id'] < 0))
                or (state['active'] and state['round_id'] is None)):
            raise ValueError('invalid prototype checkpoint lifecycle')
        for key, value in loaded.items():
            setattr(self, key, value)
        self.version, self.round_id, self.active = state['version'], state['round_id'], state['active']


class PrototypeServer(SFLV2Server):
    """Individual FCFS v3/v4 jobs with one combined backward/optimizer step."""

    def __init__(self, model, config):
        if config.get('method') not in ('na_sfl_v3_proto', 'na_sfl_v4_residual'):
            raise ValueError('PrototypeServer requires a prototype method')
        super().__init__(model, config)
        self.feature_shape = _feature_shape(config['boundary_shape_per_sample'])
        self.prototype_config = dict(config['prototype'])
        for key in ('fast_sd_weight', 'fast_proto_weight', 'slow_proto_weight', 'alignment_weight_beta'):
            value = self.prototype_config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.prototype_config['alignment_reduction'] not in ('mse', 'l2_squared'):
            raise ValueError('alignment reduction must be mse or l2_squared')
        self.memory = GlobalPrototypeMemory(self.num_classes, self.feature_shape, device=self.device)
        self.seen_jobs = set()
        self.v4_config = None
        self.residual_bank = None
        if config['method'] == 'na_sfl_v4_residual':
            from utils.na_sfl_residual import ResidualBank
            self.v4_config = dict(config['v4'])
            v4 = self.v4_config
            gamma = v4['pseudo_loss_weight_gamma']
            if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma < 0:
                raise ValueError('pseudo gamma must be finite and nonnegative')
            if (v4['enabled'] is not True or v4['residual_source'] != 'previous_completed_round'
                    or v4['residual_store_dtype'] != 'float32' or v4['fallback_when_missing'] != 'proto_only'):
                raise ValueError('unsupported v4 residual policy')
            self.residual_bank = ResidualBank(self.num_classes, self.feature_shape,
                seed=config.get('seed', 42), max_residuals_per_class=v4['max_residuals_per_class'],
                min_residuals_per_class=v4['min_residuals_per_class'])

    def start_round(self, round_id):
        self.memory.start_round(round_id)
        if self.residual_bank is not None:
            self.residual_bank.start_round(round_id)
        self.seen_jobs.clear()

    def finish_round(self):
        stats = self.memory.finish_round()
        if self.residual_bank is not None:
            stats['residual_bank_stats'] = self.residual_bank.finish_round()
        return stats

    def extra_state_dict(self):
        state = {'memory': self.memory.state_dict(), 'server_version': self.version,
                'seen_jobs': sorted(self.seen_jobs)}
        if self.residual_bank is not None:
            state['residual_bank'] = self.residual_bank.state_dict()
        return state

    def load_extra_state_dict(self, state):
        version = state['server_version']
        if type(version) is not int or version < 0:
            raise ValueError('invalid prototype server version')
        seen = [tuple(job) for job in state['seen_jobs']]
        if len(set(seen)) != len(seen):
            raise ValueError('duplicate saved prototype job IDs')
        memory = GlobalPrototypeMemory(self.num_classes, self.feature_shape, device=self.device)
        memory.load_state_dict(state['memory'])
        bank = None
        if self.residual_bank is not None:
            from utils.na_sfl_residual import ResidualBank
            bank = ResidualBank(self.num_classes, self.feature_shape,
                seed=self.residual_bank.seed, max_residuals_per_class=self.residual_bank.capacity,
                min_residuals_per_class=self.residual_bank.minimum)
            bank.load_state_dict(state['residual_bank'])
            if (bank.round_id != memory.round_id or bank.version != memory.version
                    or bank.active != memory.active or not bank.seen_jobs.issubset(set(seen))):
                raise ValueError('residual/global checkpoint lifecycle or job set differs')
        elif 'residual_bank' in state:
            raise ValueError('v3 checkpoint cannot include residual state')
        self.memory, self.residual_bank = memory, bank
        self.version, self.seen_jobs = version, set(seen)

    def process(self, job_id, activation, labels, *, return_cut_gradient=True):
        # Force all prototype jobs through the typed batch entrypoint.
        raise RuntimeError('prototype server requires typed process_batch([packet])')

    def process_batch(self, packets):
        if len(packets) != 1:
            raise ValueError('prototype server accepts exactly one individual job')
        self.memory._require_active()
        packet = packets[0]
        job = packet['job_id']
        if (not isinstance(job, tuple) or len(job) != 3
                or any(type(i) is not int or i < 0 for i in job)
                or job[0] != self.memory.round_id):
            raise ValueError('job ID must identify this active round/client/batch')
        if job in self.seen_jobs:
            raise ValueError('duplicate prototype server job')
        if packet.get('return_cut_gradient') is not True:
            raise ValueError('prototype jobs must return their input gradient')
        kind = packet['job_kind']
        if kind not in ('SD_JOB', 'PROTO_JOB'):
            raise ValueError('unknown prototype job kind')
        x, labels = packet['activation'], packet['labels']
        _tensor(x, self.device, 'activation'); _tensor(labels, self.device, 'labels')
        if kind == 'SD_JOB':
            if 'counts' in packet:
                raise ValueError('SD job must not upload separate prototype counts')
            if (x.dtype != torch.float32 or tuple(x.shape[1:]) != self.feature_shape
                    or x.ndim != len(self.feature_shape) + 1 or len(x) < 1
                    or labels.dtype != torch.int64 or labels.shape != (len(x),)
                    or (labels < 0).any().item() or (labels >= self.num_classes).any().item()):
                raise ValueError('SD shape, dtype or labels do not match boundary')
            samples = len(x)
            if packet.get('actual_batch_size', samples) != samples:
                raise ValueError('SD actual batch size differs')
        else:
            counts = packet['counts']
            _tensor(counts, self.device, 'counts')
            samples = packet.get('actual_batch_size', int(counts.sum().item()))
            validate_prototypes(x, labels, counts, self.feature_shape, self.num_classes, samples)
        leaf = x.detach().clone().requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        cfg = self.prototype_config
        pseudo = {'activation': None, 'labels': None, 'weights': None,
            'pseudo_samples_generated': 0, 'pseudo_fallback_classes': [],
            'pseudo_fallback_reasons': {}, 'pseudo_donors': [],
            'bank_version': self.residual_bank.version if self.residual_bank is not None else 0,
            'pseudo_gradient_to_client': self.v4_config['pseudo_gradient_to_client'] if self.v4_config else False}
        with sflv2_fp32():
            if kind == 'SD_JOB':
                (protos, classes, counts), prototype_time = _timed(
                    self.device, lambda: make_local_prototypes(leaf, labels))
            else:
                protos, classes, prototype_time = leaf, labels, 0.
            def forward():
                nonlocal pseudo
                sd_logits = self.model(leaf) if kind == 'SD_JOB' else None
                logits = self.model(protos)
                _finite(logits, 'prototype logits')
                if sd_logits is not None:
                    _finite(sd_logits, 'SD logits')
                    sd_loss = F.cross_entropy(sd_logits, labels)
                else:
                    sd_loss = leaf.sum() * 0.
                weights = counts.to(protos.dtype)
                proto_loss = (F.cross_entropy(logits, classes, reduction='none') * weights).sum() / weights.sum()
                align = self.memory.alignment_loss(protos, classes, counts, cfg['alignment_reduction'])
                combined = proto_loss + cfg['alignment_weight_beta'] * align
                total = (cfg['fast_sd_weight'] * sd_loss + cfg['fast_proto_weight'] * combined
                    if kind == 'SD_JOB' else cfg['slow_proto_weight'] * combined)
                pseudo_loss = leaf.sum() * 0.
                # Gamma-zero skips generation AND the extra classifier/BN forward.
                if (kind == 'PROTO_JOB' and self.v4_config is not None
                        and self.v4_config['pseudo_loss_weight_gamma'] > 0):
                    pseudo = self.residual_bank.make_pseudo(protos, classes, counts, job, self.v4_config)
                    if pseudo['activation'] is not None:
                        pseudo_logits = self.model(pseudo['activation'])
                        _finite(pseudo_logits, 'pseudo logits')
                        pseudo_loss = (F.cross_entropy(pseudo_logits, pseudo['labels'], reduction='none')
                            * pseudo['weights']).sum()
                        total = total + self.v4_config['pseudo_loss_weight_gamma'] * pseudo_loss
                return sd_logits, logits, sd_loss, proto_loss, align, pseudo_loss, total
            (sd_logits, logits, sd_loss, proto_loss, align, pseudo_loss, loss), forward_time = _timed(self.device, forward)
            for value, name in ((sd_loss, 'SD loss'), (proto_loss, 'prototype CE'), (align, 'alignment loss'), (pseudo_loss, 'pseudo loss'), (loss, 'server loss')):
                _finite(value, name)
            _, backward_time = _timed(self.device, loss.backward)
            _gradients(self.model)
            _finite(leaf.grad, 'prototype/cut gradient')
            gradient = leaf.grad.detach().clone()  # Capture before the sole optimizer step.
            _, optimizer_time = _timed(self.device, self.optimizer.step)
        self.memory.accumulate(protos, classes, counts)
        if self.residual_bank is not None and kind == 'SD_JOB':
            self.residual_bank.accumulate(leaf, labels, protos, classes, counts, job)
        before = self.version
        self.version += 1
        self.seen_jobs.add(job)
        correct = (int((sd_logits.argmax(1) == labels).sum().item()) if kind == 'SD_JOB'
            else int(((logits.argmax(1) == classes) * counts).sum().item()))
        item = {'job_id': job, 'job_kind': kind, 'cut_gradient': gradient,
            'samples': samples, 'loss': float(loss.item()), 'correct': correct,
            'version_before': before, 'version_after': self.version,
            'forward_cuda_s': forward_time, 'backward_cuda_s': backward_time,
            'optimizer_cuda_s': optimizer_time, 'gradient_scale_to_local_mean': 1.,
            'L_SD': float(sd_loss.item()), 'L_ProtoCE': float(proto_loss.item()),
            'L_Align': float(align.item()), 'L_Pseudo': float(pseudo_loss.item()), 'L_total': float(loss.item()),
            'proto_class_ids': classes.detach().cpu().tolist(),
            'proto_counts': counts.detach().cpu().tolist(),
            'prototype_compute_s': prototype_time,
            'server_forward_calls': (2 if kind == 'SD_JOB' else 1) + int(pseudo['activation'] is not None),
            'server_sd_forward_calls': int(kind == 'SD_JOB'), 'server_proto_forward_calls': 1,
            'server_optimizer_step_id': self.version}
        if self.residual_bank is not None:
            item.update({key: value for key, value in pseudo.items() if key not in ('activation', 'labels', 'weights')})
            item['server_pseudo_forward_calls'] = int(pseudo['activation'] is not None)
            item['pseudo_generation_disabled'] = self.v4_config['pseudo_loss_weight_gamma'] == 0
        return {'jobs': [item], 'samples': samples, **{key: item[key] for key in
            ('version_before', 'version_after', 'forward_cuda_s', 'backward_cuda_s', 'optimizer_cuda_s')}}
