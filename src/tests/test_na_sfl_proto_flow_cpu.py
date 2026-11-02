"""Real worker/graph orchestration with only the CUDA API boundary replaced."""
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn


def test_proto_worker_without_aux_records_gradient_mode_and_drains(tmp_path, monkeypatch):
    import torch._dynamo  # Load optimizer/device interfaces before replacing CUDA objects.
    from utils import sflv2, na_sfl_prototype, na_sfl_proto_client
    from utils.deit_async import run_async_round
    from utils.na_sfl_config import load_config
    class Event:
        def __init__(self, **kwargs): pass
        def record(self, *args): pass
        def synchronize(self): pass
        def query(self): return True
        def elapsed_time(self, other): return 0.
    class Stream:
        cuda_stream = 1
        def __init__(self, **kwargs): pass
        def wait_event(self, event): pass
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'Stream', Stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: Stream())
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]).startswith('cuda'):
            return tensor
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda *args: None)
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    for module in (sflv2, na_sfl_prototype, na_sfl_proto_client):
        monkeypatch.setattr(module, '_timed', lambda device, fn: (fn(), 0.))
    cfg = load_config(overrides={'method': 'na_sfl_v3_proto', 'n_clients': 1,
        'slow_client_ids': [0], 'batch_size': 2, 'samples_per_client': 2})
    server = na_sfl_prototype.PrototypeServer(nn.Sequential(nn.Conv2d(1, 10, 1),
        nn.AdaptiveAvgPool2d(1), nn.Flatten()), {**cfg, 'boundary_shape_per_sample': (1, 2, 2)})
    node = na_sfl_proto_client.PrototypeClient(0, nn.Conv2d(1, 1, 1), cfg, prototype_mode=True)
    owner = SimpleNamespace(args=cfg, device='cuda:0', clients=[node], server=server,
        isolated_augmentation=True, run_dir=tmp_path, job_records=[], network=None,
        _log=lambda text: None)
    owner._client_batches = lambda *args: iter([(0, (torch.ones(2, 1, 2, 2),
        torch.tensor([1, 3]), torch.tensor([0, 1])))])
    server.start_round(1)
    rows = run_async_round(owner, None, None, [[0, 1]], 1)
    assert len(rows) == 1 and rows[0]['gradient_mode'] == 'server'
    assert rows[0]['job_kind'] == 'PROTO_JOB' and not rows[0]['auxiliary_updated']
    assert node.version == server.version == 1 and node.pending is None
    assert owner.last_async_boundary['success'] and owner.last_async_boundary['unresolved_futures'] == 0
    assert not torch.cuda.is_initialized()
