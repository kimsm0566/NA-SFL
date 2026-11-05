"""Synthetic CPU graph contracts. Production still requires CUDA models."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F


def make_client(monkeypatch, *, slow=False, bn=False, optimizer='sgd'):
    from utils import na_sfl_pipeline as api
    monkeypatch.setattr(api, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(api, '_timed', lambda device, fn: (fn(), 0.))
    torch.manual_seed(19)
    model = nn.Sequential(nn.Conv2d(1, 3, 1, bias=False),
        nn.BatchNorm2d(3) if bn else nn.Identity())
    node = api.PipelineClient(0, model, {'optimizer': optimizer, 'lr': .01,
        'num_classes': 3, 'random_sd': {'sample_ratio': .1, 'sample_seed': 42}},
        prototype_mode=slow)
    return node


def batches(sizes=(128, 56)):
    generator = torch.Generator().manual_seed(7)
    return [(torch.randn(size, 1, 2, 2, generator=generator),
             torch.arange(size, dtype=torch.long) % 3) for size in sizes]


def uploads(node, data, start=0):
    ids = [(1, 0, start + k) for k in range(len(data))]
    node.begin_group(ids)
    result = []
    for job, (images, labels) in zip(ids, data):
        begun = node.begin(job, images)
        result.append(node.prepare_upload(job, begun, labels))
    return ids, result


def objective(z, labels, selected=None):
    classes, counts = labels.unique(sorted=True, return_counts=True)
    p = torch.stack([z[labels == c].mean(0) for c in classes])
    sd = z if selected is None else z[selected.long()]
    sd_labels = labels if selected is None else labels[selected.long()]
    logits = sd.mean((2, 3))
    proto_logits = p.mean((2, 3))
    proto_ce = (F.cross_entropy(proto_logits, classes, reduction='none') * counts).sum() / counts.sum()
    # Nonzero fixed alignment target tests the full-batch class-mean chain rule.
    alignment = ((p - .17).square().flatten(1).mean(1) * counts).sum() / counts.sum()
    return .8 * F.cross_entropy(logits, sd_labels) + .2 * (proto_ce + .1 * alignment)


def response(upload):
    z = upload.get('selected_sd', upload['activation']).detach().clone().requires_grad_()
    if upload['job_kind'] == 'SD_JOB':
        objective(z, upload['labels']).backward()
        return z.grad
    p = upload['activation'].detach().clone().requires_grad_()
    labels, counts = upload['labels'], upload['counts']
    proto_ce = (F.cross_entropy(p.mean((2, 3)), labels, reduction='none') * counts).sum() / counts.sum()
    align = ((p - .17).square().flatten(1).mean(1) * counts).sum() / counts.sum()
    loss = .8 * F.cross_entropy(z.mean((2, 3)), upload['selected_labels']) + .2 * (proto_ce + .1 * align)
    loss.backward()
    return dict(grad_selected_sd=z.grad, grad_prototypes=p.grad,
        selected_indices=upload['selected_indices'].clone(), classes=labels.clone())


@pytest.mark.parametrize('slow', [False, True])
@pytest.mark.parametrize('bn', [False, True])
@pytest.mark.parametrize('sizes', [(128, 56), (7,)])
def test_group_gradients_match_equal_batch_local_objective(monkeypatch, slow, bn, sizes):
    node = make_client(monkeypatch, slow=slow, bn=bn)
    reference = copy.deepcopy(node.model)
    data = batches(sizes)
    ids, packets = uploads(node, data)
    for (images, labels), packet in zip(data, packets):
        selected = packet.get('selected_indices')
        (objective(reference(images), labels, selected) / len(ids)).backward()
    before = [p.detach().clone() for p in node.model.parameters()]
    expected_bytes = node.pending_graph_bytes
    assert node.pending_graph_count == len(ids) and expected_bytes > 0
    for job, packet in reversed(list(zip(ids, packets))):
        result = node.backward_job(job, response(packet))
        assert result['gradient_scale'] == 1 / len(ids)
        assert result['group_size'] == len(ids) and result['version_before'] == 0
        assert node.version == 0
        assert all(torch.equal(p, saved) for p, saved in zip(node.model.parameters(), before))
    for p, ref in zip(node.model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, ref.grad, rtol=2e-5, atol=2e-6)
    result = node.step_group()
    assert result['version_after'] == 1 and result['version_before'] == 0
    assert node.pending is None and node.pending_graph_count == node.pending_graph_bytes == 0
    assert node.max_pending_graphs == len(ids)
    assert all(torch.equal(p, old - .01 * p.grad) for p, old in zip(node.model.parameters(), before))
    if bn:
        torch.testing.assert_close(node.model[1].running_mean, reference[1].running_mean)
        assert node.model[1].num_batches_tracked.item() == len(ids)


def test_24_jobs_have_12_adam_steps_and_bound_two(monkeypatch):
    node = make_client(monkeypatch, optimizer='adam', slow=True)
    sizes = [128] * 23 + [56]
    count = 0
    for start in range(0, 24, 2):
        ids, packets = uploads(node, batches(sizes[start:start + 2]), start)
        assert node.pending_graph_count == 2
        for job, packet in zip(ids, packets):
            node.backward_job(job, response(packet))
            count += 1
        node.step_group()
    assert count == 24 and node.version == 12 and node.max_pending_graphs == 2
    assert all(int(state['step']) == 12 for state in node.optimizer.state.values())


@pytest.mark.parametrize('bad', ['nan', 'shape', 'dtype', 'indices', 'classes', 'missing', 'extra', 'wrong_job'])
def test_rejected_hybrid_reply_preserves_graph_and_gradients_for_retry(monkeypatch, bad):
    node = make_client(monkeypatch, slow=True)
    ids, packets = uploads(node, batches())
    good = response(packets[0])
    invalid = {key: value.clone() for key, value in good.items()}
    job = ids[0]
    if bad == 'nan': invalid['grad_prototypes'][0] = float('nan')
    if bad == 'shape': invalid['grad_selected_sd'] = invalid['grad_selected_sd'][:1]
    if bad == 'dtype': invalid['grad_prototypes'] = invalid['grad_prototypes'].double()
    if bad == 'indices': invalid['selected_indices'] = invalid['selected_indices'].flip(0)
    if bad == 'classes': invalid['classes'] = invalid['classes'].flip(0)
    if bad == 'missing': invalid.pop('classes')
    if bad == 'extra': invalid['job_id'] = torch.tensor([0])
    if bad == 'wrong_job': job = (1, 0, 2)
    before = [p.detach().clone() for p in node.model.parameters()]
    with pytest.raises((ValueError, RuntimeError)):
        node.backward_job(job, invalid)
    assert node.pending_graph_count == 2 and node.version == 0
    assert all(p.grad is None and torch.equal(p, old) for p, old in zip(node.model.parameters(), before))
    node.backward_job(ids[0], good)
    grads = [p.grad.clone() for p in node.model.parameters()]
    with pytest.raises((ValueError, RuntimeError)):
        node.backward_job(ids[0], good)
    for p, grad in zip(node.model.parameters(), grads): torch.testing.assert_close(p.grad, grad)
    node.backward_job(ids[1], response(packets[1]))
    node.step_group()


@pytest.mark.parametrize('ids', [[], [(1, 0, 0)] * 2, [(1, 0, 0), (1, 0, 2)],
    [(1, 0, 0), (2, 0, 1)], [(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 0, 1), (1, 0, 2)]])
def test_group_reservation_rejects_invalid_identity_and_bounds(monkeypatch, ids):
    node = make_client(monkeypatch)
    with pytest.raises(ValueError): node.begin_group(ids)
    assert node.pending is None and node.version == 0


def test_group_barrier_rejects_early_step_backward_third_and_duplicate_forward(monkeypatch):
    node = make_client(monkeypatch)
    data = batches()
    ids = [(1, 0, 0), (1, 0, 1)]
    node.begin_group(ids)
    first = node.begin(ids[0], data[0][0])
    packet = node.prepare_upload(ids[0], first, data[0][1])
    with pytest.raises(RuntimeError): node.backward_job(ids[0], response(packet))
    with pytest.raises(RuntimeError): node.step_group()
    with pytest.raises(RuntimeError): node.begin_group(ids)
    with pytest.raises((ValueError, RuntimeError)): node.begin(ids[0], data[0][0])
    second = node.begin(ids[1], data[1][0])
    p2 = node.prepare_upload(ids[1], second, data[1][1])
    with pytest.raises((ValueError, RuntimeError)): node.begin((1, 0, 2), data[0][0])
    node.backward_job(ids[0], response(packet))
    with pytest.raises(RuntimeError): node.step_group()
    node.backward_job(ids[1], response(p2))
    node.step_group()
    with pytest.raises(RuntimeError): node.step_group()


@pytest.mark.parametrize('mutation', ['version', 'inplace', 'replacement'])
def test_external_parameter_or_version_change_rejected_before_backward(monkeypatch, mutation):
    node = make_client(monkeypatch)
    ids, packets = uploads(node, batches())
    if mutation == 'version': node.version += 1
    elif mutation == 'inplace':
        with torch.no_grad(): node.model[0].weight.add_(1)
    else: node.model[0].weight = nn.Parameter(node.model[0].weight.detach().clone())
    with pytest.raises(RuntimeError): node.backward_job(ids[0], response(packets[0]))
    assert node.pending_graph_count == 2
    node.clear_pending()
    assert node.pending is None and node.pending_graph_bytes == 0
    assert all(p.grad is None for p in node.model.parameters())
    with pytest.raises(RuntimeError): node.step_group()


def test_payload_is_device_detached_copy_and_preserves_whole_batch_prototypes(monkeypatch):
    from utils.na_sfl_hybrid import sample_indices
    node = make_client(monkeypatch, slow=True)
    data = batches()
    ids, packets = uploads(node, data)
    assert [p['selected_sd'].shape[0] for p in packets] == [13, 6]
    for job, (images, labels), p in zip(ids, data, packets):
        for value in p.values():
            if isinstance(value, torch.Tensor):
                assert value.device == node.device and not value.requires_grad
        assert torch.equal(p['selected_indices'], sample_indices(len(images), .1, 42, job))
        assert p['counts'].sum().item() == len(images)
        with torch.no_grad():
            z = node.model(images)
            expected = torch.stack([z[labels == c].mean(0) for c in p['labels']])
        torch.testing.assert_close(p['activation'], expected)
    # Changing a queued copy must not alter the retained graph.
    retained = node.pending[ids[0]].prototypes.detach().clone()
    packets[0]['activation'].zero_()
    torch.testing.assert_close(node.pending[ids[0]].prototypes, retained)
    node.clear_pending()
    assert node.pending is None and node.version == 0


def test_abort_partial_backward_clears_gradient_and_cannot_step(monkeypatch):
    node = make_client(monkeypatch)
    ids, packets = uploads(node, batches())
    node.backward_job(ids[1], response(packets[1]))
    assert node.pending_graph_count == 1
    node.clear_pending()
    assert node.pending is None and all(p.grad is None for p in node.model.parameters())
    with pytest.raises(RuntimeError): node.step_group()
    assert node.version == 0


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'nan'])
def test_fast_invalid_reply_preserves_existing_accumulated_gradient(monkeypatch, bad):
    node = make_client(monkeypatch)
    ids, packets = uploads(node, batches())
    node.backward_job(ids[1], response(packets[1]))
    saved = [p.grad.clone() for p in node.model.parameters()]
    good = response(packets[0])
    invalid = good.clone()
    if bad == 'shape': invalid = invalid[:1]
    if bad == 'dtype': invalid = invalid.double()
    if bad == 'nan': invalid[0] = float('nan')
    with pytest.raises(ValueError): node.backward_job(ids[0], invalid)
    for p, expected in zip(node.model.parameters(), saved):
        torch.testing.assert_close(p.grad, expected)
    assert node.pending_graph_count == 1 and node.version == 0
    node.backward_job(ids[0], good)
    node.step_group()


def test_upload_identity_label_rejection_and_duplicate_preparation(monkeypatch):
    node = make_client(monkeypatch, slow=True)
    images, labels = batches((56,))[0]
    job = (1, 0, 0)
    node.begin_group([job])
    begun = node.begin(job, images)
    with pytest.raises(ValueError):
        node.prepare_upload(job, {**begun, 'client_version': 99}, labels)
    with pytest.raises(ValueError):
        node.prepare_upload(job, begun, labels[:1])
    assert node.pending_graph_count == 1
    packet = node.prepare_upload(job, begun, labels)
    with pytest.raises(RuntimeError): node.prepare_upload(job, begun, labels)
    node.backward_job(job, response(packet))
    node.step_group()


def test_parameter_change_between_forward_and_step_is_detected(monkeypatch):
    node = make_client(monkeypatch)
    ids, packets = uploads(node, batches())
    for job, packet in zip(ids, packets): node.backward_job(job, response(packet))
    with torch.no_grad(): node.model[0].weight.add_(1)
    with pytest.raises(RuntimeError): node.step_group()
    assert node.version == 0
    node.clear_pending()


def test_forward_cuda_events_bracket_only_model_call_before_finite_check_and_clone(monkeypatch):
    from utils import na_sfl_pipeline as api
    node = make_client(monkeypatch)
    events = []
    class Event:
        def __init__(self, **kwargs): self.index = sum(e[0] == 'create' for e in events); events.append(('create', self.index))
        def record(self): events.append(('record', self.index))
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    # Only the hardware boundary is substituted: the production timed wrapper
    # runs the actual CPU model, while CUDA Event.record order stays observable.
    original = api._timed
    monkeypatch.setattr(api, '_timed', lambda device, fn: (fn(), 0.))
    activation, elapsed, pair = api._timed_model_forward(torch.device('cuda:0'),
        lambda: (events.append(('model',)), torch.ones(2, requires_grad=True))[1])
    assert events == [('create', 0), ('create', 1), ('record', 0), ('model',), ('record', 1)]
    assert pair[0].index == 0 and pair[1].index == 1
    assert elapsed == 0. and activation.requires_grad
    monkeypatch.setattr(api, '_timed', original)

    # The real begin path invokes finite validation and upload clone after the
    # exact model-span helper returned its two events.
    trace = []
    original_forward = api._timed_model_forward
    def forward(device, operation):
        value, elapsed, _ = original_forward(device, operation)
        trace.append('model_span_closed')
        return value, elapsed, pair
    monkeypatch.setattr(api, '_timed_model_forward', forward)
    original_finite = api._finite
    def finite(value, name):
        trace.append('finite')
        return original_finite(value, name)
    monkeypatch.setattr(api, '_finite', finite)
    original_clone = torch.Tensor.clone
    def clone(value, *args, **kwargs):
        trace.append('clone')
        return original_clone(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'clone', clone)
    node.begin_group([(1, 0, 0)])
    begun = node.begin((1, 0, 0), batches((2,))[0][0])
    assert begun['forward_cuda_events'] is pair
    assert trace == ['model_span_closed', 'finite', 'clone']
