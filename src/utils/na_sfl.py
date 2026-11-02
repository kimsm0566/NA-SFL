"""Selective-gradient primitives, reusing existing SFLV2 ownership contracts."""
import torch
from torch.nn import functional as F

from models.model import average_states
from utils.sflv2 import SFLV2Client, _timed, sflv2_fp32


def auxiliary_update(activation, labels, head, head_optimizer, client_optimizer=None):
    """No client optimizer means stop-gradient S; otherwise local-gradient A."""
    head_optimizer.zero_grad(set_to_none=True)
    inputs = activation.detach() if client_optimizer is None else activation
    logits = head(inputs)
    loss = F.cross_entropy(logits, labels, reduction='mean')
    if not torch.isfinite(loss).item():
        raise ValueError('nonfinite auxiliary loss')
    loss.backward()
    for model in (head,):
        if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise ValueError('missing or nonfinite auxiliary gradient')
    if client_optimizer is not None:
        for group in client_optimizer.param_groups:
            if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in group['params']):
                raise ValueError('missing or nonfinite local client gradient')
        client_optimizer.step()
    head_optimizer.step()
    return {'loss': float(loss.detach().item()), 'samples': int(labels.numel()),
            'correct': int((logits.detach().argmax(1) == labels).sum().item())}


def average_round_models(clients, heads, weights, *, auxiliary_fedavg):
    state = average_states([model.state_dict() for model in clients], weights)
    for model in clients:
        model.load_state_dict(state, strict=True)
    if heads and auxiliary_fedavg:
        state = average_states([head.state_dict() for head in heads], weights)
        for head in heads:
            head.load_state_dict(state, strict=True)


class NASFLClient(SFLV2Client):
    def __init__(self, client_id, model, config, *, head=None, head_config=None, mode='server'):
        super().__init__(client_id, model, config)
        if mode not in ('server', 'aux') or (mode == 'aux' and head is None):
            raise ValueError('invalid auxiliary mode/head')
        from utils.sflv2 import _optimizer
        self.gradient_mode, self.auxiliary_head = mode, head
        self.auxiliary_optimizer = _optimizer(head, head_config) if head is not None else None

    def auxiliary_step(self, job_id, labels):
        pending = self.pending
        if pending is None or pending.job_id != job_id or pending.version != self.version:
            raise RuntimeError('auxiliary update lacks the matching client graph/version')
        if self.auxiliary_head is None:
            return None
        with sflv2_fp32():
            metrics, elapsed = _timed(self.device, lambda: auxiliary_update(
                pending.activation, labels, self.auxiliary_head, self.auxiliary_optimizer,
                self.optimizer if self.gradient_mode == 'aux' else None))
        metrics['cuda_s'] = elapsed
        if self.gradient_mode == 'aux':
            metrics['version_before'] = self.version
            self.version += 1
            self.pending = None
            metrics['version_after'] = self.version
        return metrics
