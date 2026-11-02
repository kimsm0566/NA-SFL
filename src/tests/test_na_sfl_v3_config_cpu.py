import pytest

from utils.na_sfl_config import load_config


def test_v3_and_v4_have_no_auxiliary_and_share_protocol_defaults():
    for method in ('na_sfl_v3_proto', 'na_sfl_v4_residual'):
        cfg = load_config(overrides={'method': method})
        assert cfg['aux_client_ids'] == [] and not cfg['auxiliary_fedavg']
        assert cfg['slow_client_ids'] == [0, 5]
        assert cfg['server_policy'] == 'individual'
        assert cfg['prototype']['fast_sd_weight'] == cfg['prototype']['fast_proto_weight'] == .5
        assert cfg['prototype']['slow_proto_weight'] == 1.
        assert cfg['prototype']['alignment_reduction'] == 'mse'
        assert cfg['v4']['enabled'] == method.endswith('residual')
        assert cfg['v4']['max_residuals_per_class'] == 64


@pytest.mark.parametrize('patch', [
    {'server_policy': 'concat'}, {'aux_client_count': 2},
    {'prototype': {'alignment_reduction': 'other'}},
    {'prototype': {'alignment_weight_beta': -1}},
    {'v4': {'pseudo_loss_weight_gamma': float('nan')}},
    {'v4': {'pseudo_gradient_to_client': 1}},
    {'slow_client_ids': [0, 0]},
    {'client_compute_seconds': {'auxiliary': .1}}])
def test_invalid_proto_mode_is_rejected_before_cuda_or_network(patch):
    with pytest.raises(ValueError):
        load_config(overrides={'method': 'na_sfl_v3_proto', **patch})


@pytest.mark.parametrize('method,policy', [('splitfedv2', 'individual'),
    ('na_sfl_v1', 'individual'), ('na_sfl_v2', 'concat')])
def test_legacy_small_client_configs_ignore_unused_proto_slow_ids(method, policy):
    cfg = load_config(overrides={'method': method, 'server_policy': policy, 'n_clients': 2})
    assert cfg['n_clients'] == 2


@pytest.mark.parametrize('corruption', ['server_version', 'memory_version', 'memory_round',
    'active', 'bank_version', 'bank_round', 'source_round', 'seen_jobs'])
def test_resume_rejects_feature_memory_outside_completed_boundary(corruption):
    import copy
    from utils.na_sfl_train import validate_feature_memory_boundary
    saved = {'config': {'method': 'na_sfl_v4_residual'}, 'next_round': 2,
        'server_version': 4, 'job_records': [{'job_id': (2, 0, 0)}],
        'server_extra_state': {'server_version': 4, 'seen_jobs': [(2, 0, 0)],
            'memory': {'version': 2, 'round_id': 2, 'active': False},
            'residual_bank': {'version': 2, 'round_id': 2, 'bank_round_id': 2, 'active': False}}}
    validate_feature_memory_boundary(saved)
    broken = copy.deepcopy(saved)
    state = broken['server_extra_state']
    if corruption == 'server_version': state['server_version'] = 3
    elif corruption == 'memory_version': state['memory']['version'] = 1
    elif corruption == 'memory_round': state['memory']['round_id'] = 1
    elif corruption == 'active': state['memory']['active'] = True
    elif corruption == 'bank_version': state['residual_bank']['version'] = 1
    elif corruption == 'bank_round': state['residual_bank']['round_id'] = 1
    elif corruption == 'source_round': state['residual_bank']['bank_round_id'] = 1
    else: state['seen_jobs'] = []
    with pytest.raises(ValueError):
        validate_feature_memory_boundary(broken)
