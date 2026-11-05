"""Server-only BN policies; affine tensors and the SD model state stay shared."""
from contextlib import contextmanager

import torch
from torch import nn


BN_POLICIES = ('shared_original', 'sd_train_proto_eval_bn_only',
               'dual_running_stats_shared_affine')
_BUFFERS = ('running_mean', 'running_var', 'num_batches_tracked')


class ServerBatchNorm:
    """A sidecar owns Proto statistics; ordinary model buffers always mean SD.

    Swapping references, rather than copying values in place between forwards,
    preserves tensors saved by autograd for the combined SD/Proto backward.
    The sidecar is checkpointed through PrototypeServer.extra_state_dict().
    """

    def __init__(self, model, policy='shared_original'):
        if policy not in BN_POLICIES:
            raise ValueError('unsupported Server BN policy')
        self.model, self.policy = model, policy
        self.layers = {name: layer for name, layer in model.named_modules()
                       if isinstance(layer, nn.modules.batchnorm._BatchNorm)}
        if any(not layer.track_running_stats for layer in self.layers.values()):
            raise ValueError('Server BN policies require tracked running statistics')
        self.proto = ({name: {key: getattr(layer, key).detach().clone() for key in _BUFFERS}
                       for name, layer in self.layers.items()}
                      if policy == 'dual_running_stats_shared_affine' else {})
        self.forward_calls = {'sd': 0, 'proto': 0}
        self.running_updates = {'sd': 0, 'proto': 0}
        self.proto_forwards_before_first_sd = 0
        self.active_domain = None

    def require_sd_domain(self):
        if self.active_domain is not None:
            raise RuntimeError('SD evaluation/checkpoint requires restored BN buffers')

    @contextmanager
    def domain(self, domain):
        if domain not in ('sd', 'proto') or self.active_domain is not None:
            raise ValueError('invalid or nested Server BN domain')
        originals = []
        self.active_domain = domain
        train_forward = self.model.training
        try:
            for name, layer in self.layers.items():
                saved = (layer, layer.training, {key: getattr(layer, key) for key in _BUFFERS})
                originals.append(saved)
                if domain == 'proto' and self.policy == 'sd_train_proto_eval_bn_only':
                    layer.training = False
                elif domain == 'proto' and self.policy == 'dual_running_stats_shared_affine':
                    for key in _BUFFERS:
                        setattr(layer, key, self.proto[name][key])
            yield
            if train_forward:
                if domain == 'proto' and self.forward_calls['sd'] == 0:
                    self.proto_forwards_before_first_sd += 1
                self.forward_calls[domain] += 1
                if not (domain == 'proto' and self.policy == 'sd_train_proto_eval_bn_only'):
                    self.running_updates[domain] += 1
        finally:
            for layer, training, buffers in reversed(originals):
                for key, value in buffers.items():
                    setattr(layer, key, value)
                layer.training = training
            self.active_domain = None

    def state_dict(self):
        self.require_sd_domain()
        layers = {}
        for name, layer in self.layers.items():
            item = {'sd': {key: getattr(layer, key).detach().cpu().clone() for key in _BUFFERS}}
            if name in self.proto:
                item['proto'] = {key: value.detach().cpu().clone() for key, value in self.proto[name].items()}
            layers[name] = item
        return {'policy': self.policy, 'layers': layers,
                'forward_calls': dict(self.forward_calls), 'running_updates': dict(self.running_updates),
                'proto_forwards_before_first_sd': self.proto_forwards_before_first_sd}

    def load_state_dict(self, state):
        self.require_sd_domain()
        if (not isinstance(state, dict) or state.get('policy') != self.policy
                or set(state.get('layers', {})) != set(self.layers)):
            raise ValueError('Server BN checkpoint policy/layers mismatch')
        domains = {'sd', 'proto'} if self.proto else {'sd'}
        loaded = {}
        for name, layer in self.layers.items():
            entry = state['layers'][name]
            if set(entry) != domains:
                raise ValueError('Server BN checkpoint domain mismatch')
            loaded[name] = {}
            for domain in domains:
                values = entry[domain]
                if set(values) != set(_BUFFERS):
                    raise ValueError('Server BN checkpoint buffer mismatch')
                loaded[name][domain] = {}
                for key in _BUFFERS:
                    value, target = values[key], getattr(layer, key)
                    if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                            or value.dtype != target.dtype or not torch.isfinite(value).all().item()
                            or (key in ('running_var', 'num_batches_tracked') and (value < 0).any().item())):
                        raise ValueError('invalid Server BN checkpoint buffer')
                    loaded[name][domain][key] = value.detach().to(target.device).clone()
        counts = {}
        for key in ('forward_calls', 'running_updates'):
            values = state.get(key)
            if (not isinstance(values, dict) or set(values) != {'sd', 'proto'}
                    or any(type(value) is not int or value < 0 for value in values.values())):
                raise ValueError('invalid Server BN checkpoint counters')
            counts[key] = dict(values)
        early = state.get('proto_forwards_before_first_sd')
        if type(early) is not int or not 0 <= early <= counts['forward_calls']['proto']:
            raise ValueError('invalid first-SD BN checkpoint counter')
        expected_updates = dict(counts['forward_calls'])
        if self.policy == 'sd_train_proto_eval_bn_only':
            expected_updates['proto'] = 0
        if counts['running_updates'] != expected_updates:
            raise ValueError('Server BN update/forward counters mismatch')
        expected_sd = (sum(counts['forward_calls'].values()) if self.policy == 'shared_original'
                       else counts['forward_calls']['sd'])
        for name, layer in self.layers.items():
            if loaded[name]['sd']['num_batches_tracked'].item() != expected_sd:
                raise ValueError('SD BN tensor counter disagrees with forward history')
            if self.proto and loaded[name]['proto']['num_batches_tracked'].item() != counts['forward_calls']['proto']:
                raise ValueError('Proto BN tensor counter disagrees with forward history')
            if any(not torch.equal(getattr(layer, key), value) for key, value in loaded[name]['sd'].items()):
                raise ValueError('duplicated SD BN state disagrees with restored model buffers')
        for name in self.proto:
            self.proto[name] = loaded[name]['proto']
        self.forward_calls = counts['forward_calls']
        self.running_updates = counts['running_updates']
        self.proto_forwards_before_first_sd = early

    def summary(self):
        self.require_sd_domain()
        return {'policy': self.policy, 'forward_calls': dict(self.forward_calls),
                'running_updates': dict(self.running_updates),
                'proto_forwards_before_first_sd': self.proto_forwards_before_first_sd,
                'evaluation_input_domain': 'sd',
                'evaluation_statistics_domain': 'shared' if self.policy == 'shared_original' else 'sd',
                'running_counters': {name: {'sd': int(layer.num_batches_tracked.item()),
                    **({'proto': int(self.proto[name]['num_batches_tracked'].item())} if self.proto else {})}
                    for name, layer in self.layers.items()}}


def validate_bn_checkpoint_boundary(saved):
    state = saved['server_extra_state'].get('batchnorm')
    policy = saved['config'].get('bn_policy', 'shared_original')
    if state is None:
        if policy != 'shared_original':
            raise ValueError('BN experiment checkpoint is missing domain state')
        return
    jobs = saved['job_records']
    sd = sum(job.get('job_kind') in ('SD_JOB', 'HYBRID_JOB') for job in jobs)
    proto = len(jobs) + sum(job.get('server_pseudo_forward_calls', 0) for job in jobs)
    calls = {'sd': sd, 'proto': proto}
    updates = {'sd': sd, 'proto': 0 if policy == 'sd_train_proto_eval_bn_only' else proto}
    if state['policy'] != policy or state['forward_calls'] != calls or state['running_updates'] != updates:
        raise ValueError('BN checkpoint counters disagree with committed Job history')
    expected_sd = sd + proto if policy == 'shared_original' else sd
    for name, domains in state['layers'].items():
        if int(domains['sd']['num_batches_tracked']) != expected_sd:
            raise ValueError('BN SD tensor counter disagrees with committed Job history')
        if policy == 'dual_running_stats_shared_affine' and int(domains['proto']['num_batches_tracked']) != proto:
            raise ValueError('BN Proto tensor counter disagrees with committed Job history')
        for key in _BUFFERS:
            model_key = name + '.' + key if name else key
            if model_key not in saved['server_model'] or not torch.equal(
                    domains['sd'][key].cpu(), saved['server_model'][model_key].cpu()):
                raise ValueError('BN checkpoint SD domain disagrees with saved Server model')
