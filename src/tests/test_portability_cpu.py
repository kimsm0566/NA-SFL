"""Portable GPU selection checks; no CUDA context or model execution."""
import subprocess

import pytest


def test_unprofiled_gpu_requires_explicit_opt_in_without_bypassing_busy_guard():
    from utils.na_sfl_run import validate_training_gpu
    gpu = {'name': 'NVIDIA RTX 4090', 'compute_processes': [], 'free_bytes': 90, 'total_bytes': 100}
    with pytest.raises(RuntimeError):
        validate_training_gpu(gpu)
    validate_training_gpu(gpu, allow_unprofiled=True)
    for patch in ({'compute_processes': ['another job']}, {'free_bytes': 70}):
        with pytest.raises(RuntimeError):
            validate_training_gpu({**gpu, **patch}, allow_unprofiled=True)


def test_cli_unprofiled_gpu_opt_in_is_explicit_and_boolean():
    from utils.option import parse_args
    from utils.na_sfl_config import load_config
    assert parse_args([]).config['allow_unprofiled_gpu'] is False
    assert parse_args(['--allow-unprofiled-gpu']).config['allow_unprofiled_gpu'] is True
    with pytest.raises(ValueError):
        load_config(overrides={'allow_unprofiled_gpu': 'yes'})


@pytest.mark.parametrize('visible,expected', [('1', '1'), ('GPU-example, GPU-other', 'GPU-example')])
def test_gpu_snapshot_inspects_the_actual_visible_physical_gpu(monkeypatch, visible, expected):
    from utils import deit_artifacts as api
    calls = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    def run(command, **kwargs):
        calls.append(command)
        if '--query-compute-apps=gpu_uuid,pid,used_gpu_memory' in command:
            return subprocess.CompletedProcess(command, 0, 'GPU-example, 999, 200\nGPU-other, 888, 100\n', '')
        return subprocess.CompletedProcess(command, 0, 'GPU-example, NVIDIA RTX A5000, 24000, 100, 23900, 35, 0\n', '')
    monkeypatch.setattr(api.subprocess, 'run', run)
    snapshot = api.gpu_snapshot(0)
    assert '--id=' + expected in calls[0]
    assert snapshot['compute_processes'] == ['GPU-example, 999, 200']


def test_gpu_snapshot_rejects_an_unavailable_logical_index_before_nvml_query(monkeypatch):
    from utils.deit_artifacts import gpu_snapshot
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-example')
    with pytest.raises(RuntimeError, match='visible'):
        gpu_snapshot(1)


@pytest.mark.parametrize('cuda_mask', [None, '0'])
def test_single_uuid_exposed_by_nvidia_runtime_selects_that_gpu_without_cuda_mask(monkeypatch, cuda_mask):
    from utils import deit_artifacts as api
    if cuda_mask is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', cuda_mask)
    monkeypatch.setenv('NVIDIA_VISIBLE_DEVICES', 'GPU-selected')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        out = '' if '--query-compute-apps=gpu_uuid,pid,used_gpu_memory' in command else 'GPU-selected, NVIDIA RTX A5000, 24000, 100, 23900, 35, 0\n'
        return subprocess.CompletedProcess(command, 0, out, '')
    monkeypatch.setattr(api.subprocess, 'run', run)
    assert api.gpu_snapshot(0)['uuid'] == 'GPU-selected'
    assert '--id=GPU-selected' in calls[0]
