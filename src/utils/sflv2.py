"""Existing split Client/Server primitives and evaluation used by NA-SFL.

Each Client retains one forward graph. The persistent shared Server performs
one mean-CE update per SD. CUDA event times describe actual shared-GPU work.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from utils.na_sfl_config import _positive


def _optimizer(model, config):
    name = config.get('optimizer', 'adam')
    lr = _positive(config.get('lr', .001), 'lr')
    decay = _positive(config.get('weight_decay', 0.), 'weight_decay', True)
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr,
            betas=tuple(config.get('betas', (.9, .999))),
            eps=_positive(config.get('eps', 1e-8), 'eps'), weight_decay=decay)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr,
            momentum=_positive(config.get('momentum', 0.), 'momentum', True), weight_decay=decay)
    raise ValueError('NA-SFL optimizer must be adam or sgd')


@contextmanager
def sflv2_fp32():
    """Scope reproducible FP32 settings to new SFLV2 work, restoring on exit."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _model_device(model):
    tensors = list(model.parameters()) + list(model.buffers())
    if not torch.cuda.is_available() or not tensors or tensors[0].device.type != "cuda":
        raise RuntimeError("SFLV2 models require CUDA; CPU fallback is forbidden")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise RuntimeError("SFLV2 model tensors must share one CUDA device")
    return device


def _finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite")


def _tensor(tensor, device, name):
    if not isinstance(tensor, torch.Tensor) or tensor.device != device:
        raise ValueError(f"{name} must be a tensor on {device}")
    _finite(tensor, name)


def _gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise ValueError(f"missing parameter gradient: {name}")
            _finite(parameter.grad, f"gradient {name}")


def _timed(device, operation):
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = operation()
        end.record()
        end.synchronize()
        return value, float(start.elapsed_time(end) / 1000.)


@dataclass(frozen=True)
class _Pending:
    job_id: object
    version: int
    activation: torch.Tensor


class SFLV2Client:
    """One client bottom with a single pending graph and monotonically rising version.

    begin returns {job_id, client_version, activation, forward_cuda_s}; activation
    is a detached CUDA clone for the server. complete validates the retained job
    and version, consumes an already mean-normalized cut gradient, then returns
    {job_id, version_before, version_after, backward_cuda_s, optimizer_cuda_s}.
    Successful completion clears pending; rejected identity/gradient input leaves
    the graph and model intact so the correct completion can still be submitted.
    """

    def __init__(self, client_id, model, config):
        if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0:
            raise ValueError("client_id must be a nonnegative integer")
        self.client_id = client_id
        self.model = model
        self.device = _model_device(model)
        self.optimizer = _optimizer(model, config)
        self.version = 0
        self.pending = None

    def begin(self, job_id, images):
        if self.pending is not None:
            raise RuntimeError("client already has a pending job")
        _tensor(images, self.device, "images")
        if images.ndim != 4 or images.shape[0] <= 0 or not images.is_floating_point():
            raise ValueError("images must be a nonempty floating NCHW tensor")
        self.optimizer.zero_grad(set_to_none=True)
        with sflv2_fp32():
            activation, elapsed = _timed(self.device, lambda: self.model(images))
        _finite(activation, "activation")
        if not activation.requires_grad:
            raise RuntimeError("client forward must retain an autograd graph")
        self.pending = _Pending(job_id, self.version, activation)
        return {
            "job_id": job_id, "client_version": self.version,
            "activation": activation.detach().clone(), "forward_cuda_s": elapsed,
        }

    def complete(self, job_id, gradient):
        pending = self.pending
        if pending is None:
            raise RuntimeError("client has no pending job")
        if job_id != pending.job_id:
            raise ValueError("completion job_id does not match pending job")
        if self.version != pending.version:
            raise RuntimeError("client version changed while graph was pending")
        _tensor(gradient, self.device, "cut gradient")
        if gradient.shape != pending.activation.shape or gradient.dtype != pending.activation.dtype:
            raise ValueError("cut gradient shape/dtype must match pending activation")
        with sflv2_fp32():
            _, backward = _timed(self.device, lambda: pending.activation.backward(gradient.detach()))
            _gradients(self.model)
            _, optimizer = _timed(self.device, self.optimizer.step)
        before = self.version
        self.version += 1
        self.pending = None
        return {
            "job_id": job_id, "version_before": before, "version_after": self.version,
            "backward_cuda_s": backward, "optimizer_cuda_s": optimizer,
        }


class SFLV2Server:
    """Persistent shared top; process performs one mean-CE SGD update per job.

    Returns {job_id, cut_gradient, loss, correct, samples, version_before,
    version_after, forward_cuda_s, backward_cuda_s, optimizer_cuda_s}. The cut
    gradient is a detached CUDA clone captured before optimizer.step; do not
    divide it by batch size again. Loss is a Python mean, accuracy counts ints.
    """

    def __init__(self, model, config):
        self.model = model
        self.device = _model_device(model)
        self.num_classes = config.get('num_classes',10)
        if type(self.num_classes) is not int or self.num_classes<1:
            raise ValueError('num_classes must be a positive integer')
        self.optimizer = _optimizer(model, config)
        self.version = 0

    def process(self, job_id, activation, labels, *, return_cut_gradient=True):
        _tensor(activation, self.device, "activation")
        _tensor(labels, self.device, "labels")
        expected_ndim = 4
        if activation.ndim != expected_ndim or activation.shape[0] <= 0 or not activation.is_floating_point():
            raise ValueError("activation must match the model's nonempty floating boundary tensor")
        if labels.dtype != torch.long or labels.shape != (activation.shape[0],):
            raise ValueError("labels must be int64 with one label per activation")
        if (labels < 0).any().item() or (labels >= self.num_classes).any().item():
            raise ValueError(f"labels must be between 0 and {self.num_classes-1}")
        if type(return_cut_gradient) is not bool:
            raise ValueError('return_cut_gradient must be boolean')
        leaf = activation.detach().clone().requires_grad_(return_cut_gradient)
        self.optimizer.zero_grad(set_to_none=True)
        with sflv2_fp32():
            def forward():
                logits = self.model(leaf)
                return logits, F.cross_entropy(logits, labels, reduction="mean")
            (logits, loss), forward_time = _timed(self.device, forward)
            _finite(logits, "server logits")
            _finite(loss, "server loss")
            _, backward = _timed(self.device, loss.backward)
            _gradients(self.model)
            gradient = None
            if return_cut_gradient:
                _finite(leaf.grad, "cut gradient")
                gradient = leaf.grad.detach().clone()
            _, optimizer = _timed(self.device, self.optimizer.step)
        before = self.version
        self.version += 1
        return {
            "job_id": job_id, "cut_gradient": gradient, "loss": float(loss.item()),
            "correct": int((logits.argmax(1) == labels).sum().item()),
            "samples": int(labels.numel()), "version_before": before,
            "version_after": self.version, "forward_cuda_s": forward_time,
            "backward_cuda_s": backward, "optimizer_cuda_s": optimizer,
        }


    def process_batch(self, packets):
        """One sample-mean Server update, local-mean cut gradients for S jobs.

        A one-packet call reuses process verbatim. Concat intentionally changes
        Server optimizer frequency and training BatchNorm statistics.
        """
        if not packets:
            raise ValueError('server batch must contain at least one SD')
        if len(packets) == 1:
            packet = packets[0]
            item = self.process(packet['job_id'], packet['activation'], packet['labels'],
                                return_cut_gradient=packet['return_cut_gradient'])
            item['gradient_scale_to_local_mean'] = 1. if packet['return_cut_gradient'] else None
            return {'jobs': [item], 'samples': item['samples'],
                **{key: item[key] for key in ('version_before', 'version_after',
                    'forward_cuda_s', 'backward_cuda_s', 'optimizer_cuda_s')}}
        for packet in packets:
            x, y, wanted = packet['activation'], packet['labels'], packet['return_cut_gradient']
            _tensor(x, self.device, 'activation')
            _tensor(y, self.device, 'labels')
            if (x.ndim != 4 or x.shape[0] < 1 or not x.is_floating_point()
                    or y.dtype != torch.long or y.shape != (x.shape[0],)
                    or (y < 0).any().item() or (y >= self.num_classes).any().item()
                    or type(wanted) is not bool):
                raise ValueError('invalid SD/label shape, dtype, classes or gradient mode')
        sizes = [packet['activation'].shape[0] for packet in packets]
        total = sum(sizes)
        leaf = torch.cat([p['activation'].detach() for p in packets]).requires_grad_(
            any(p['return_cut_gradient'] for p in packets))
        labels = torch.cat([p['labels'] for p in packets])
        self.optimizer.zero_grad(set_to_none=True)
        with sflv2_fp32():
            def forward():
                logits = self.model(leaf)
                losses = F.cross_entropy(logits, labels, reduction='none')
                return logits, losses, losses.mean()
            (logits, losses, loss), forward_time = _timed(self.device, forward)
            _finite(logits, 'server logits')
            _finite(loss, 'server loss')
            _, backward_time = _timed(self.device, loss.backward)
            _gradients(self.model)
            gradients, offset = [], 0
            for packet, size in zip(packets, sizes):
                gradient = None
                if packet['return_cut_gradient']:
                    gradient = leaf.grad[offset:offset+size].detach().clone() * (total / size)
                    _finite(gradient, 'cut gradient')
                gradients.append(gradient)
                offset += size
            _, optimizer_time = _timed(self.device, self.optimizer.step)
        before = self.version
        self.version += 1
        jobs, offset = [], 0
        for packet, size, gradient in zip(packets, sizes, gradients):
            jobs.append({'job_id': packet['job_id'], 'samples': size, 'cut_gradient': gradient,
                'loss': float(losses[offset:offset+size].mean().item()),
                'correct': int((logits[offset:offset+size].argmax(1) == labels[offset:offset+size]).sum().item()),
                'version_before': before, 'version_after': self.version,
                'gradient_scale_to_local_mean': total / size if gradient is not None else None})
            offset += size
        return {'jobs': jobs, 'samples': total, 'version_before': before, 'version_after': self.version,
                'forward_cuda_s': forward_time, 'backward_cuda_s': backward_time,
                'optimizer_cuda_s': optimizer_time}


class SFLV2Trainer:
    """Shared deterministic seeding and global test evaluation for NASFLTrainer."""

    def _seed(self, *parts):
        import hashlib
        text = ":".join(map(str, (self.args["seed"], *parts)))
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)

    @torch.inference_mode()
    def _evaluate(self, images, labels):
        import numpy as np
        from data.data import build_loader
        client, server = self.clients[0].model, self.server.model
        previous = client.training, server.training
        loader = build_loader(images, labels, np.arange(len(labels)), self.args["eval_batch_size"],
                              False, 0, torch.Generator().manual_seed(self._seed("evaluation")))
        total, correct, loss_sum = 0, 0, 0.
        client.eval()
        server.eval()
        try:
            with sflv2_fp32():
                for batch_images, batch_labels, _ids in loader:
                    batch_images = batch_images.to(self.device, non_blocking=True)
                    batch_labels = batch_labels.to(self.device, non_blocking=True)
                    _tensor(batch_images, self.device, "evaluation images")
                    _tensor(batch_labels, self.device, "evaluation labels")
                    logits = server(client(batch_images))
                    _finite(logits, "evaluation logits")
                    loss = F.cross_entropy(logits, batch_labels, reduction="sum")
                    _finite(loss, "evaluation loss")
                    loss_sum += float(loss.item())
                    correct += int((logits.argmax(1) == batch_labels).sum().item())
                    total += len(batch_labels)
        finally:
            client.train(previous[0])
            server.train(previous[1])
        return 100. * correct / total, loss_sum / total, total
