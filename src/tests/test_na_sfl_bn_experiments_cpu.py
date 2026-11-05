"""Decision precision, fail-stop ordering and command/config contracts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def _rows(accuracy):
    return [{'round': i, 'test_accuracy': accuracy} for i in range(1, 101)]


def _api():
    from utils import na_sfl_bn_experiments
    return na_sfl_bn_experiments


@pytest.mark.parametrize('value,improved,weights', [
    ('57.54', False, (.5, .5)), ('57.539999999999', False, (.5, .5)),
    ('57.540000000001', True, (.8, .2)), ('60', True, (.8, .2))])
def test_decision_uses_strict_unrounded_decimal_last10_mean(value, improved, weights):
    baseline = _rows('57.54'); loss = _rows(value)
    # Earlier accuracy and the final single round cannot override last10 mean.
    loss[0]['test_accuracy'] = '100'
    decision = _api().decision_from_rounds(baseline, loss)
    assert decision['improved'] is improved
    assert (decision['selected_fast_sd_weight'], decision['selected_fast_proto_weight']) == weights
    assert decision['baseline_last10_mean_exact'] == '57.54'


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'nan', 'bool', 'range', 'uncompleted'])
def test_invalid_stage_a_cannot_be_treated_as_no_improvement(bad):
    rows = _rows(60)
    if bad == 'missing': rows.pop(95)
    if bad == 'duplicate': rows[-1]['round'] = 99
    if bad == 'nan': rows[-1]['test_accuracy'] = float('nan')
    if bad == 'bool': rows[-1]['test_accuracy'] = True
    if bad == 'range': rows[-1]['test_accuracy'] = 101
    with pytest.raises(ValueError):
        _api().decision_from_rounds(_rows(57), rows, stage_a_completed=bad != 'uncompleted')


@pytest.mark.parametrize('accuracy,weights', [(58, (.8, .2)), (57, (.5, .5))])
def test_stage_order_decision_before_bn_and_same_weights_for_both_bn(tmp_path, accuracy, weights):
    calls = []
    def launch(name, sd, proto, policy):
        if calls:
            decision = json.loads((tmp_path/'decision.json').read_text())
            assert (decision['selected_fast_sd_weight'], decision['selected_fast_proto_weight']) == weights
            assert (sd, proto) == weights
        calls.append((name, sd, proto, policy))
        return {'rows': _rows(accuracy), 'summary': {'completed': True}, 'name': name}
    _api().execute_stages(launch, _rows('57.54'), tmp_path)
    assert [call[0] for call in calls] == ['v3_loss_08_02_shared_bn', 'v3_bn_freeze_proto', 'v3_bn_dual']
    assert calls[0][1:] == (.8, .2, 'shared_original')
    assert calls[1][-1] == 'sd_train_proto_eval_bn_only'
    assert calls[2][-1] == 'dual_running_stats_shared_affine'


def test_failed_stage_a_never_starts_bn(tmp_path):
    calls = []
    def launch(name, sd, proto, policy):
        calls.append(name)
        return {'rows': _rows(60)[:99], 'summary': {'completed': False}}
    with pytest.raises(ValueError): _api().execute_stages(launch, _rows(57), tmp_path)
    assert calls == ['v3_loss_08_02_shared_bn']
    assert not (tmp_path/'decision.json').exists()


def test_bn_config_and_cli_reject_v4_and_preserve_default():
    from utils.na_sfl_config import load_config
    from utils.option import parse_args
    default = load_config('configs/na_sfl_v3_variable.yaml')
    assert default['bn_policy'] == 'shared_original'
    args = parse_args(['--config', 'configs/na_sfl_v3_variable.yaml', '--fast-sd-weight', '.8',
                      '--fast-proto-weight', '.2', '--bn-policy', 'dual_running_stats_shared_affine'])
    assert args.config['prototype']['fast_sd_weight'] == .8
    assert args.config['prototype']['fast_proto_weight'] == .2
    with pytest.raises(ValueError):
        load_config('configs/na_sfl_v4_variable.yaml', {'bn_policy': 'dual_running_stats_shared_affine'})
    with pytest.raises(SystemExit):
        parse_args(['--config', 'configs/na_sfl_v3_variable.yaml', '--v3-bn-ablation'])


@pytest.mark.parametrize('where', ['waiting', 'before_wait'])
def test_supervisor_sigterm_stops_and_reaps_owned_child(tmp_path, where):
    marker = tmp_path/'child.json'
    result = tmp_path/'cancelled.json'
    script = f'''
import json, subprocess, sys
from pathlib import Path
from utils.na_sfl_bn_experiments import controlled_termination, wait_stage_process, owned_stage_process
try:
    with controlled_termination():
        with owned_stage_process([sys.executable, '-c', 'import time; time.sleep(120)']) as child:
            Path({str(marker)!r}).write_text(json.dumps({{'pid': child.pid}}))
            if {where!r} == 'before_wait':
                import time
                time.sleep(120)
            wait_stage_process(child, lambda: None)
except KeyboardInterrupt:
    Path({str(result)!r}).write_text(json.dumps({{'child_returncode': child.poll()}}))
'''
    supervisor = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(300):
            if marker.exists() or supervisor.poll() is not None: break
            time.sleep(.01)
        assert marker.exists(), supervisor.communicate(timeout=2)[1].decode()
        child_pid = json.loads(marker.read_text())['pid']
        supervisor.send_signal(signal.SIGTERM)
        supervisor.wait(timeout=8)
        assert json.loads(result.read_text())['child_returncode'] is not None
        with pytest.raises(ProcessLookupError): os.kill(child_pid, 0)
    finally:
        if supervisor.poll() is None:
            supervisor.kill(); supervisor.wait()
