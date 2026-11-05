"""Random SD sampling and one retained Client graph for both Hybrid branches."""
import hashlib
import json
import math
from dataclasses import replace

import torch

from utils.na_sfl_proto_client import PrototypeClient
from utils.sflv2 import SFLV2Client, _timed, _tensor, _gradients, sflv2_fp32


def sample_count(batch_size, ratio):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('sampling batch size must be a positive integer')
    if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio) or not 0 <= ratio <= 1):
        raise ValueError('sampling ratio must be finite in [0, 1]')
    return 0 if ratio == 0 else max(1, math.floor(ratio * batch_size + .5))


def sample_indices(batch_size, ratio, seed, job_id):
    """Sorted int32 indices from a stateless dedicated CPU RNG, never global RNG."""
    count = sample_count(batch_size, ratio)
    if type(seed) is not int or seed < 0:
        raise ValueError('sampling seed must be a nonnegative integer')
    if (not isinstance(job_id, (tuple, list)) or len(job_id) != 3
            or any(type(value) is not int or value < (1 if i == 0 else 0)
                   for i, value in enumerate(job_id))):
        raise ValueError('sampling identity must be (round, client, batch)')
    encoded = json.dumps([seed, *job_id], separators=(',', ':')).encode('ascii')
    derived = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big')
    generator = torch.Generator(device='cpu').manual_seed(derived)
    return torch.randperm(batch_size, generator=generator)[:count].sort().values.to(torch.int32)


class SDOnlyClient(SFLV2Client):
    """Full Client forward, optionally selected SD, and one ordinary Server reply.

    No Prototype operation is called. Indexing retains the original full-batch
    graph (including Client BN); the Server mean-CE gradient is applied once.
    """
    def __init__(self, client_id, model, config, *, sampled_mode=False):
        super().__init__(client_id, model, config)
        self.sampled_mode = sampled_mode
        self.sample_ratio = config['random_sd']['sample_ratio']
        self.sample_seed = config['random_sd']['sample_seed']
        if sample_count(1, self.sample_ratio) == 0:
            raise ValueError('SD-only requires positive sampling')
        sample_indices(1, self.sample_ratio, self.sample_seed, (1, client_id, 0))
        self.gradient_mode = 'server'
        self.auxiliary_head = self.auxiliary_optimizer = None
        self._sd_prepared = False

    def begin(self, job_id, images):
        result = super().begin(job_id, images)
        self._sd_prepared = False
        return result

    def prepare_sd_upload(self, job_id, begun, labels):
        if (self.pending is None or self.pending.job_id != job_id
                or self.pending.version != self.version or job_id[1] != self.client_id):
            raise RuntimeError('SD-only upload requires the matching retained graph')
        if self._sd_prepared:
            raise RuntimeError('SD-only upload already prepared')
        z = self.pending.activation
        _tensor(labels, self.device, 'SD-only labels')
        if labels.dtype != torch.int64 or labels.shape != (z.shape[0],):
            raise ValueError('SD-only labels must match the full Client batch')
        def select():
            indices = (sample_indices(z.shape[0], self.sample_ratio, self.sample_seed, job_id).to(self.device)
                       if self.sampled_mode else torch.arange(z.shape[0], device=self.device, dtype=torch.int32))
            return indices, z[indices.long()] if self.sampled_mode else z, labels[indices.long()]
        (indices, selected, chosen_labels), elapsed = _timed(self.device, select)
        self.pending = replace(self.pending, activation=selected)
        self._sd_prepared = True
        return dict(activation=selected.detach().clone(), labels=chosen_labels.detach().clone(),
            selected_indices=indices.detach().clone(), sampling_compute_s=elapsed)


class RandomSDClient(PrototypeClient):
    def __init__(self, client_id, model, config, *, prototype_mode=False):
        super().__init__(client_id, model, config, prototype_mode=prototype_mode)
        sampling = config.get('random_sd', {})
        self.sample_ratio = sampling.get('sample_ratio', .1)
        self.sample_seed = sampling.get('sample_seed', 42)
        sample_indices(1, self.sample_ratio, self.sample_seed, (1, client_id, 0))
        self._hybrid_pending = None

    def prepare_upload(self, job_id, begun, labels):
        if not self.prototype_mode or self.sample_ratio == 0:
            return super().prepare_upload(job_id, begun, labels)
        if self.pending is None or self.pending.job_id != job_id or self.pending.version != self.version:
            raise RuntimeError('Hybrid upload requires the matching retained graph')
        if self._hybrid_pending is not None:
            raise RuntimeError('Hybrid upload already prepared for pending job')
        from utils.na_sfl_prototype import make_local_prototypes
        z = self.pending.activation
        (prototypes, classes, counts), proto_time = _timed(self.device,
            lambda: make_local_prototypes(z, labels))
        def select():
            indices = sample_indices(z.shape[0], self.sample_ratio, self.sample_seed, job_id).to(self.device)
            return indices, z[indices.long()], labels[indices.long()]
        (indices, selected, selected_labels), sampling_time = _timed(self.device, select)
        self._hybrid_pending = (selected, prototypes, indices.detach().clone(), classes.detach().clone())
        return dict(job_kind='HYBRID_JOB', activation=prototypes.detach().clone(), labels=classes,
            counts=counts, selected_sd=selected.detach().clone(), selected_labels=selected_labels.detach().clone(),
            selected_indices=indices.detach().clone(), sample_ratio=self.sample_ratio, sample_seed=self.sample_seed,
            actual_batch_size=z.shape[0], prototype_compute_s=proto_time, sampling_compute_s=sampling_time)

    def complete(self, job_id, gradient):
        if self._hybrid_pending is None:
            return super().complete(job_id, gradient)
        pending = self.pending
        if pending is None:
            raise RuntimeError('client has no pending job')
        if pending.job_id != job_id:
            raise ValueError('completion job_id does not match pending job')
        if pending.version != self.version:
            raise RuntimeError('client version changed while graph was pending')
        keys = {'grad_selected_sd', 'grad_prototypes', 'selected_indices', 'classes'}
        if not isinstance(gradient, dict) or set(gradient) != keys:
            raise ValueError('Hybrid completion requires both gradients and identity tensors')
        selected, prototypes, indices, classes = self._hybrid_pending
        for name, expected in (('grad_selected_sd', selected), ('grad_prototypes', prototypes),
                               ('selected_indices', indices), ('classes', classes)):
            value = gradient[name]
            _tensor(value, self.device, name)
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError('Hybrid gradient/identity shape or dtype mismatch: ' + name)
            if name in ('selected_indices', 'classes') and not torch.equal(value, expected):
                raise ValueError('Hybrid gradient identity mismatch: ' + name)
        with sflv2_fp32():
            _, backward = _timed(self.device, lambda: torch.autograd.backward(
                [selected, prototypes], [gradient['grad_selected_sd'].detach(), gradient['grad_prototypes'].detach()]))
            _gradients(self.model)
            _, optimizer = _timed(self.device, self.optimizer.step)
        before = self.version
        self.version += 1
        self.clear_pending()
        return dict(job_id=job_id, version_before=before, version_after=self.version,
            backward_cuda_s=backward, optimizer_cuda_s=optimizer)

    def clear_pending(self):
        self.pending = None
        self._hybrid_pending = None
