"""Opt-in CIFAR10 NA-SFL settings, independent of historical DeiT defaults."""
import hashlib
import math
from pathlib import Path

import numpy as np
import yaml
from utils.na_sfl_network import NETWORK_DEFAULTS, validate_network_config


DEFAULTS = {
    'mode': 'live_async', 'model': 'resnet18_cifar', 'dataset': 'cifar10',
    'n_clients': 10, 'samples_per_client': 3000, 'partition': 'dirichlet',
    'dirichlet_alpha': .1, 'batch_size': 128, 'local_epochs': 1, 'rounds': 50,
    'cut': 2, 'optimizer': 'adam', 'lr': .001, 'weight_decay': .0005,
    'betas': [.9, .999], 'eps': 1e-8, 'momentum': .9,
    'client_optimizer': {}, 'server_optimizer': {}, 'aux_optimizer': {},
    'method': 'na_sfl_v1', 'aux_client_count': 0, 'seed': 42,
    'augmentation': True, 'device': 0, 'allow_unprofiled_gpu': False, 'num_workers': 0, 'eval_batch_size': 250,
    'worker_timeout_s': 300., 'queue_capacity': 20,
    'max_batches_per_client': None, 'target_accuracy': None,
    'data_path': 'datasets', 'result_path': 'results/na_sfl_v1',
    'server_policy': 'individual', 'server_max_batch_samples': 512,
    'client_compute_seconds': {}, 'network': None,
    'slow_client_ids': [0, 5], 'prototype': {}, 'v4': {}, 'metrics': {}, 'smoke_test': False,
}

PROTO_METHODS = ('na_sfl_v3_proto', 'na_sfl_v4_residual')
PROTOTYPE_DEFAULTS = dict(fast_sd_weight=.5, fast_proto_weight=.5, slow_proto_weight=1.,
    alignment_weight_beta=.1, alignment_reduction='mse', global_update='once_per_round',
    weighted_by_class_sample_count=True)
V4_DEFAULTS = dict(pseudo_loss_weight_gamma=.25, pseudo_gradient_to_client=True,
    pseudo_samples_per_class=4, min_residuals_per_class=2, max_residuals_per_class=64,
    residual_source='previous_completed_round', residual_recenter=True, residual_scale=1.,
    residual_store_dtype='float32', fallback_when_missing='proto_only')


def _prototype_options(config):
    for name, defaults in (('prototype', PROTOTYPE_DEFAULTS), ('v4', V4_DEFAULTS)):
        raw = config[name]
        allowed = set(defaults) | ({'enabled'} if name == 'v4' else set())
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ValueError('invalid ' + name + ' configuration')
        config[name] = {**defaults, **raw}
    p, v = config['prototype'], config['v4']
    for key in ('fast_sd_weight', 'fast_proto_weight', 'slow_proto_weight', 'alignment_weight_beta'):
        p[key] = _positive(p[key], key, True)
    if (p['alignment_reduction'] not in ('mse', 'l2_squared') or p['global_update'] != 'once_per_round'
            or p['weighted_by_class_sample_count'] is not True):
        raise ValueError('unsupported prototype reduction or global update')
    for key in ('pseudo_loss_weight_gamma', 'residual_scale'):
        v[key] = _positive(v[key], key, True)
    for key in ('pseudo_gradient_to_client', 'residual_recenter'):
        if type(v[key]) is not bool:
            raise ValueError('invalid boolean: ' + key)
    for key in ('pseudo_samples_per_class', 'min_residuals_per_class', 'max_residuals_per_class'):
        if type(v[key]) is not int or v[key] < 1:
            raise ValueError('invalid positive integer: ' + key)
    if (v['min_residuals_per_class'] < 2 or v['max_residuals_per_class'] < v['min_residuals_per_class']
            or v['pseudo_samples_per_class'] < v['min_residuals_per_class']
            or v['residual_source'] != 'previous_completed_round' or v['residual_store_dtype'] != 'float32'
            or v['fallback_when_missing'] != 'proto_only'):
        raise ValueError('unsupported residual bank policy')
    enabled = config['method'] == 'na_sfl_v4_residual'
    if 'enabled' in v and (type(v['enabled']) is not bool or v['enabled'] != enabled):
        raise ValueError('v4.enabled must agree with selected method')
    v['enabled'] = enabled
    ids = config['slow_client_ids']
    if config['method'] in PROTO_METHODS:
        if (not isinstance(ids, list) or len(set(ids)) != len(ids)
                or any(type(i) is not int or not 0 <= i < config['n_clients'] for i in ids)):
            raise ValueError('invalid prototype Slow Client IDs')
        if (config['server_policy'] != 'individual' or config['aux_client_count']
                or config['client_compute_seconds']['auxiliary']):
            raise ValueError('v3/v4 requires individual jobs, zero Aux heads and no Aux pacing')
        if config['network'] and set(ids) != set(config['network']['slow_client_ids']):
            raise ValueError('prototype Slow IDs must match slow network links')
    if type(config['smoke_test']) is not bool:
        raise ValueError('smoke_test must be boolean')
    if type(config['allow_unprofiled_gpu']) is not bool:
        raise ValueError('allow_unprofiled_gpu must be boolean')
    raw = config['metrics']
    if not isinstance(raw, dict) or set(raw) - {'target_accuracies', 'save_per_job', 'save_per_round'}:
        raise ValueError('invalid metrics configuration')
    config['metrics'] = {'target_accuracies': [70., 65., 72., 74.],
                         'save_per_job': True, 'save_per_round': True, **raw}
    targets = config['metrics']['target_accuracies']
    if not isinstance(targets, list) or any(not 0 < _positive(t, 'accuracy') <= 100 for t in targets):
        raise ValueError('invalid target accuracy list')

def select_aux_clients(seed, n_clients, count):
    """One independent permutation: prefixes make count comparisons nested."""
    if type(count) is not int or not 0 <= count <= n_clients:
        raise ValueError('aux_client_count must be an integer in [0,n_clients]')
    digest = hashlib.sha256(f'{seed}:na_sfl:mode_selection'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little')).permutation(n_clients)[:count].tolist()


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('invalid finite numeric setting: ' + name)
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError('invalid positive setting: ' + name)
    return float(value)


def optimizer_config(config, role):
    allowed = {'optimizer', 'lr', 'weight_decay', 'betas', 'eps', 'momentum'}
    overrides = config[role + '_optimizer']
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError('invalid per-role optimizer mapping: ' + role)
    result = {key: config[key] for key in allowed}
    result.update(overrides)
    if result['optimizer'] not in ('adam', 'sgd'):
        raise ValueError('optimizer must be adam or sgd')
    for name in ('lr', 'eps', 'weight_decay', 'momentum'):
        result[name] = _positive(result[name], name, name in ('weight_decay', 'momentum'))
    if result['momentum'] >= 1:
        raise ValueError('momentum must be below1')
    betas = result['betas']
    if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1
            for v in betas):
        raise ValueError('invalid Adam betas')
    result['betas'] = list(betas)
    return result


def load_config(path=None, overrides=None):
    raw = {} if path is None else yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or (overrides is not None and not isinstance(overrides, dict)):
        raise ValueError('NA config must be a mapping')
    raw = {**raw, **(overrides or {})}
    if set(raw) - set(DEFAULTS):
        raise ValueError('unknown NA options: ' + str(sorted(set(raw) - set(DEFAULTS))))
    config = {**DEFAULTS, **raw}
    if (config['mode'], config['model'], config['dataset']) != ('live_async', 'resnet18_cifar', 'cifar10'):
        raise ValueError('NA initially requires live_async / CIFAR ResNet18 / CIFAR10')
    for name in ('n_clients', 'samples_per_client', 'batch_size', 'local_epochs', 'rounds',
                 'eval_batch_size', 'queue_capacity'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError('invalid positive integer: ' + name)
    for name in ('device', 'seed'):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError('invalid nonnegative integer: ' + name)
    if config['seed'] >= 2**32 or config['n_clients'] * config['samples_per_client'] > 50000:
        raise ValueError('seed or total sample budget out of range')
    if type(config['cut']) is not int or config['cut'] not in (1, 2, 3):
        raise ValueError('cut must be1,2 or3')
    if config['partition'] not in ('iid', 'dirichlet') or type(config['augmentation']) is not bool:
        raise ValueError('invalid partition or augmentation')
    if type(config['num_workers']) is not int or config['num_workers'] != 0:
        raise ValueError('NA data RNG requires num_workers=0')
    for name in ('dirichlet_alpha', 'worker_timeout_s'):
        config[name] = _positive(config[name], name)
    cap = config['max_batches_per_client']
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError('max_batches_per_client must be positive or null')
    target = config['target_accuracy']
    if target is not None and not 0 < _positive(target, 'target_accuracy') <= 100:
        raise ValueError('target accuracy must be in (0,100]')
    if config['method'] not in ('splitfedv2', 'na_sfl_v1', 'na_sfl_v2', 'hybrid_no_aux_avg', *PROTO_METHODS):
        raise ValueError('unsupported NA comparison method')
    if (config['server_policy'] not in ('individual', 'concat')
            or (config['method'] == 'na_sfl_v2') != (config['method'].startswith('na_sfl_')
                                                   and config['server_policy'] == 'concat')):
        raise ValueError('NA-SFLv1 is individual and NA-SFLv2 requires concat')
    cap = config['server_max_batch_samples']
    if type(cap) is not int or cap < config['batch_size']:
        raise ValueError('server sample cap must fit one complete Client SD')
    targets = config['client_compute_seconds']
    if not isinstance(targets, dict) or set(targets) - {'forward', 'backward', 'auxiliary'}:
        raise ValueError('invalid Client phase targets')
    config['client_compute_seconds'] = {key: _positive(targets.get(key, 0.), key, True)
                                       for key in ('forward', 'backward', 'auxiliary')}
    if config['network'] is not None:
        if config['n_clients'] != 10:
            raise ValueError('current Mininet topology requires exactly10 Clients')
        config['network'] = validate_network_config(config['network'])
    _prototype_options(config)
    config['aux_client_ids'] = select_aux_clients(config['seed'], config['n_clients'], config['aux_client_count'])
    if config['method'] == 'splitfedv2' and config['aux_client_count']:
        raise ValueError('splitfedv2 requires zero auxiliary-gradient clients')
    for role in ('client', 'server', 'aux'):
        config[role + '_optimizer'] = optimizer_config(config, role)
    for name in ('data_path', 'result_path'):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError('invalid path: ' + name)
        config[name] = str(Path(config[name]).expanduser().resolve())
    config['training_subset'] = config['n_clients'] * config['samples_per_client']
    config['num_classes'] = 10
    config['auxiliary_fedavg'] = config['method'] in ('na_sfl_v1', 'na_sfl_v2')
    return config


def comparison_configs(config):
    base = {key: config[key] for key in DEFAULTS}
    conditions = [('splitfedv2', 'individual', 0), ('splitfedv2', 'concat', 0),
                  ('na_sfl_v1', 'individual', 2), ('na_sfl_v2', 'concat', 2)]
    result = [load_config(overrides={**base, 'method': method, 'server_policy': policy,
                                    'aux_client_count': count}) for method, policy, count in conditions]
    if config['network'] and any(set(row['aux_client_ids']) != set(config['network']['slow_client_ids'])
                                 for row in result[2:]):
        raise ValueError('four-way comparison requires the two slow IDs to match selected Aux IDs')
    return result
