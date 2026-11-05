"""Trainer-owned two-graph, one-Client-step v5 pipeline primitive.

The transport receives detached clones only. This module retains the original
whole-batch autograd graph and never executes an optimizer step from a reply.
"""
from dataclasses import dataclass

import torch

from utils.na_sfl_hybrid import sample_indices
from utils.na_sfl_prototype import make_local_prototypes
from utils.sflv2 import (_finite, _gradients, _model_device, _optimizer,
                        _tensor, _timed, sflv2_fp32)


def _timed_model_forward(device, operation):
    """Expose exact model Events, excluding finite checks and boundary clones.

    The surrounding timer owns synchronization. CPU synthetic adapters retain
    their existing timer substitute and explicitly have no CUDA Events.
    """
    events = None
    def forward():
        nonlocal events
        if device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            value = operation()
            end.record()
            events = start, end
            return value
        return operation()
    value, elapsed = _timed(device, forward)
    return value, elapsed, events


@dataclass
class _Graph:
    job_id: tuple
    version: int
    activation: torch.Tensor
    prepared: bool = False
    selected: object = None
    prototypes: object = None
    indices: object = None
    classes: object = None


class PipelineClient:
    """Two consecutive forwards at one parameter version, then one Adam step.

    Pending graph bytes count retained boundary tensors, a lower bound rather
    than total autograd/CUDA memory. Rejected response validation leaves graphs
    and accumulated parameter gradients available for a correct response.
    """

    def __init__(self, client_id, model, config, *, prototype_mode=False):
        if type(client_id) is not int or client_id < 0:
            raise ValueError('client_id must be a nonnegative integer')
        self.client_id, self.model = client_id, model
        self.device = _model_device(model)
        self.optimizer = _optimizer(model, config)
        self.version = 0
        self.prototype_mode = prototype_mode
        self.gradient_mode = 'server'
        self.auxiliary_head = self.auxiliary_optimizer = None
        sampling = config.get('random_sd', {})
        self.sample_ratio = sampling.get('sample_ratio', .1)
        self.sample_seed = sampling.get('sample_seed', 42)
        # C preserves the v5 10% payload; no additional experiment modes here.
        if self.sample_ratio != .1 or self.sample_seed != 42:
            raise ValueError('pipeline C requires v5 sample_ratio=.1 and sample_seed=42')
        sample_indices(1, self.sample_ratio, self.sample_seed, (1, client_id, 0))
        self.num_classes = config.get('num_classes', 10)
        self._graphs = {}
        self._group_ids = None
        self._group_version = None
        self._parameter_versions = None
        self._started = set()
        self._backward_done = set()
        self._seen_jobs = set()
        self.max_pending_graphs = 0
        self.max_pending_graph_bytes = 0

    @property
    def pending(self):
        return self._graphs if self._graphs else None

    @property
    def pending_graph_count(self):
        return len(self._graphs)

    @property
    def pending_graph_bytes(self):
        return sum(t.numel() * t.element_size() for g in self._graphs.values()
                   for t in (g.activation, g.selected, g.prototypes)
                   if t is not None)

    def _record_peaks(self):
        self.max_pending_graphs = max(self.max_pending_graphs, self.pending_graph_count)
        self.max_pending_graph_bytes = max(self.max_pending_graph_bytes, self.pending_graph_bytes)

    def _parameter_state(self):
        return tuple((name, id(value), value._version, value.data_ptr())
                     for name, value in self.model.named_parameters())

    def _check_group(self):
        if self._group_ids is None:
            raise RuntimeError('client has no active pipeline group')
        if self.version != self._group_version:
            raise RuntimeError('client version changed while group was pending')
        if self._parameter_state() != self._parameter_versions:
            raise RuntimeError('client parameter tensor changed while group was pending')

    def begin_group(self, job_ids):
        if self._group_ids is not None or self._graphs:
            raise RuntimeError('previous pipeline group is still active')
        if not isinstance(job_ids, (list, tuple)) or not 1 <= len(job_ids) <= 2:
            raise ValueError('pipeline group requires one or two job IDs')
        for job in job_ids:
            if (not isinstance(job, tuple) or len(job) != 3
                    or any(type(v) is not int or v < (1 if k == 0 else 0)
                           for k, v in enumerate(job)) or job[1] != self.client_id):
                raise ValueError('job ID must identify this round/client/batch')
        if (len(set(job_ids)) != len(job_ids)
                or any(job in self._seen_jobs for job in job_ids)
                or any(job[:2] != job_ids[0][:2] or job[2] != job_ids[0][2] + k
                       for k, job in enumerate(job_ids))):
            raise ValueError('group job IDs must be unseen consecutive batches in one round')
        self.optimizer.zero_grad(set_to_none=True)
        self._group_ids = tuple(job_ids)
        self._group_version = self.version
        self._parameter_versions = self._parameter_state()
        self._started.clear()
        self._backward_done.clear()

    def begin(self, job_id, images):
        self._check_group()
        if (job_id not in self._group_ids or job_id in self._started
                or job_id != self._group_ids[len(self._started)]):
            raise ValueError('forward must match the next reserved group job')
        if self.pending_graph_count >= 2:
            raise RuntimeError('pipeline graph window is full')
        _tensor(images, self.device, 'images')
        if images.ndim != 4 or images.shape[0] < 1 or not images.is_floating_point():
            raise ValueError('images must be nonempty floating NCHW')
        with sflv2_fp32():
            activation, elapsed, events = _timed_model_forward(self.device, lambda: self.model(images))
        _finite(activation, 'activation')
        if not activation.requires_grad:
            raise RuntimeError('client forward must retain an autograd graph')
        self._check_group()
        self._graphs[job_id] = _Graph(job_id, self.version, activation)
        self._started.add(job_id)
        self._seen_jobs.add(job_id)
        self._record_peaks()
        return dict(job_id=job_id, client_version=self.version,
                    activation=activation.detach().clone(), forward_cuda_s=elapsed,
                    forward_cuda_events=events)

    def prepare_upload(self, job_id, begun, labels):
        self._check_group()
        graph = self._graphs.get(job_id)
        if graph is None or graph.prepared:
            raise RuntimeError('upload requires an unprepared matching retained graph')
        if (not isinstance(begun, dict) or begun.get('job_id') != job_id
                or begun.get('client_version') != self._group_version):
            raise ValueError('upload begin identity/version mismatch')
        z = graph.activation
        _tensor(labels, self.device, 'labels')
        if (labels.dtype != torch.int64 or labels.shape != (z.shape[0],)
                or (labels < 0).any().item() or (labels >= self.num_classes).any().item()):
            raise ValueError('labels must be in-range int64 for the whole Client batch')
        if not self.prototype_mode:
            result = dict(job_kind='SD_JOB', activation=z.detach().clone(),
                          labels=labels.detach().clone(), prototype_compute_s=0.)
        else:
            (protos, classes, counts), proto_time = _timed(self.device,
                lambda: make_local_prototypes(z, labels))
            def select():
                indices = sample_indices(z.shape[0], self.sample_ratio, self.sample_seed,
                                         job_id).to(self.device)
                return indices, z[indices.long()], labels[indices.long()]
            (indices, selected, selected_labels), sampling_time = _timed(self.device, select)
            graph.selected, graph.prototypes = selected, protos
            graph.indices, graph.classes = indices.detach().clone(), classes.detach().clone()
            result = dict(job_kind='HYBRID_JOB', activation=protos.detach().clone(),
                labels=classes.detach().clone(), counts=counts.detach().clone(),
                selected_sd=selected.detach().clone(), selected_labels=selected_labels.detach().clone(),
                selected_indices=indices.detach().clone(), sample_ratio=self.sample_ratio,
                sample_seed=self.sample_seed, actual_batch_size=z.shape[0],
                prototype_compute_s=proto_time, sampling_compute_s=sampling_time)
        graph.prepared = True
        self._record_peaks()
        return result

    def backward_job(self, job_id, gradient):
        self._check_group()
        graph = self._graphs.get(job_id)
        if graph is None:
            raise ValueError('completion job ID is unknown or already consumed')
        if (len(self._started) != len(self._group_ids)
                or any(not item.prepared for item in self._graphs.values())):
            raise RuntimeError('both group forwards/uploads must finish before backward')
        scale = 1. / len(self._group_ids)
        if self.prototype_mode:
            keys = {'grad_selected_sd', 'grad_prototypes', 'selected_indices', 'classes'}
            if not isinstance(gradient, dict) or set(gradient) != keys:
                raise ValueError('Hybrid reply requires both gradients and exact identities')
            for name, expected in (('grad_selected_sd', graph.selected),
                                   ('grad_prototypes', graph.prototypes),
                                   ('selected_indices', graph.indices), ('classes', graph.classes)):
                value = gradient[name]
                _tensor(value, self.device, name)
                if value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError('Hybrid gradient/identity shape or dtype mismatch: ' + name)
                if name in ('selected_indices', 'classes') and not torch.equal(value, expected):
                    raise ValueError('Hybrid gradient identity mismatch: ' + name)
            tensors = [graph.selected, graph.prototypes]
            gradients = [gradient[key].detach() * scale
                         for key in ('grad_selected_sd', 'grad_prototypes')]
        else:
            _tensor(gradient, self.device, 'cut gradient')
            if gradient.shape != graph.activation.shape or gradient.dtype != graph.activation.dtype:
                raise ValueError('cut gradient shape/dtype must match retained activation')
            tensors, gradients = [graph.activation], [gradient.detach() * scale]
        with sflv2_fp32():
            _, elapsed = _timed(self.device, lambda: torch.autograd.backward(tensors, gradients))
            _gradients(self.model)
        self._backward_done.add(job_id)
        del self._graphs[job_id]
        return dict(job_id=job_id, version_before=self.version, backward_cuda_s=elapsed,
                    group_size=len(self._group_ids), gradient_scale=scale)

    def step_group(self):
        self._check_group()
        if (self._graphs or self._backward_done != set(self._group_ids)
                or self._started != set(self._group_ids)):
            raise RuntimeError('optimizer step requires every group forward and backward')
        before, size = self.version, len(self._group_ids)
        with sflv2_fp32():
            _gradients(self.model)
            _, elapsed = _timed(self.device, self.optimizer.step)
        self.version += 1
        self._reset_group()
        return dict(version_before=before, version_after=self.version,
                    optimizer_cuda_s=elapsed, group_size=size)

    def _reset_group(self):
        self._graphs.clear()
        self._group_ids = self._group_version = self._parameter_versions = None
        self._started.clear()
        self._backward_done.clear()

    def clear_pending(self):
        """Abort a group without a partial step; the run owner handles failure."""
        self._reset_group()
        self.optimizer.zero_grad(set_to_none=True)
