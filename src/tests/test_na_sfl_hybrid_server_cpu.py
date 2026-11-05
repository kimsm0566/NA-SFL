"""Small synthetic mathematical contracts, not CIFAR/ResNet CPU experiments."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F


def config(q=.1):
    from utils.na_sfl_config import load_config
    return load_config(overrides={'method': 'na_sfl_proto_random_sd', 'random_sd': {
        'sample_ratio': q, 'sample_seed': 42}, 'n_clients': 2, 'slow_client_ids': [0],
        'samples_per_client': 56, 'batch_size': 56, 'weight_decay': 0.})


@pytest.fixture
def cpu(monkeypatch):
    from utils import sflv2, na_sfl_prototype
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(na_sfl_prototype, '_timed', lambda device, fn: (fn(), 0.))


def test_hybrid_defaults_are_dual_08_02_and_old_v3_unchanged():
    from utils.na_sfl_config import load_config
    cfg = config()
    assert cfg['bn_policy'] == 'dual_running_stats_shared_affine'
    assert cfg['prototype']['fast_sd_weight'] == .8
    assert cfg['prototype']['fast_proto_weight'] == .2
    old = load_config(overrides={'method': 'na_sfl_v3_proto'})
    assert old['bn_policy'] == 'shared_original'
    assert old['prototype']['fast_sd_weight'] == old['prototype']['fast_proto_weight'] == .5


@pytest.mark.parametrize('change', [{'sample_ratio': -1}, {'sample_ratio': 1.01},
    {'sample_ratio': float('nan')}, {'sample_ratio': True}, {'sample_seed': -1},
    {'sample_seed': True}, {'unknown': 1}])
def test_invalid_random_sd_settings_rejected(change):
    from utils.na_sfl_config import load_config
    with pytest.raises(ValueError):
        load_config(overrides={'method': 'na_sfl_proto_random_sd',
            'random_sd': {'sample_ratio': .1, 'sample_seed': 42, **change}})


def packet(z, y, job=(1, 0, 0), q=.1):
    from utils.na_sfl_prototype import make_local_prototypes
    from utils.na_sfl_hybrid import sample_indices
    p, c, n = make_local_prototypes(z, y)
    idx = sample_indices(len(z), q, 42, job).to(torch.int32)
    return {'job_id': job, 'job_kind': 'HYBRID_JOB', 'activation': p.detach(),
        'labels': c, 'counts': n, 'actual_batch_size': len(z), 'sample_ratio': q,
        'selected_sd': z.detach()[idx.long()], 'selected_labels': y[idx.long()],
        'selected_indices': idx, 'return_cut_gradient': True}


def top():
    return nn.Sequential(nn.Conv2d(2, 3, 1), nn.BatchNorm2d(3),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())


def test_hybrid_two_leaf_gradients_match_joint_forward_and_single_adam(cpu):
    from utils.na_sfl_prototype import PrototypeServer, make_local_prototypes
    from utils.na_sfl_bn import ServerBatchNorm
    torch.manual_seed(12)
    cfg = {**config(), 'num_classes': 3, 'boundary_shape_per_sample': (2, 2, 2)}
    bottom = nn.Conv2d(1, 2, 1); reference_bottom = copy.deepcopy(bottom)
    model = top(); reference_top = copy.deepcopy(model)
    server = PrototypeServer(model, cfg)
    # Seed a prior completed round to exercise nonzero count-weighted alignment.
    previous = torch.randn(3, 2, 2, 2)
    server.start_round(1)
    server.memory.accumulate(previous, torch.arange(3), torch.tensor([3, 8, 2]))
    server.finish_round(); server.start_round(2)
    images = torch.randn(56, 1, 2, 2); labels = torch.tensor([0]*31+[1]*19+[2]*6)
    z = bottom(images); z.retain_grad()
    request = packet(z, labels, (2, 0, 0))
    index = request['selected_indices'].long()
    ref_z = reference_bottom(images); ref_z.retain_grad()
    p, classes, counts = make_local_prototypes(ref_z, labels)
    bn = ServerBatchNorm(reference_top, 'dual_running_stats_shared_affine')
    opt = torch.optim.Adam(reference_top.parameters(), lr=.001)
    with bn.domain('sd'):
        sd = F.cross_entropy(reference_top(ref_z[index]), labels[index])
    with bn.domain('proto'):
        ce = (F.cross_entropy(reference_top(p), classes, reduction='none')*counts).sum()/56
    alignment = ((p-previous).square().flatten(1).mean(1)*counts).sum()/56
    loss = .8*sd+.2*(ce+.1*alignment)
    loss.backward(); opt.step()
    item = server.process_batch([request])['jobs'][0]
    reply = item['cut_gradient']
    local_p = make_local_prototypes(z, labels)[0]
    torch.autograd.backward([z[index], local_p],
        [reply['grad_selected_sd'], reply['grad_prototypes']])
    assert item['L_total'] == pytest.approx(loss.item())
    assert item['L_SD'] == pytest.approx(sd.item())
    assert item['L_Align'] == pytest.approx(alignment.item())
    assert item['sd_count'] == 6 and item['samples'] == 56
    assert item['proto_counts'] == [31, 19, 6]
    assert torch.allclose(z.grad, ref_z.grad, atol=1e-7, rtol=2e-5)
    for a,b in zip(bottom.parameters(),reference_bottom.parameters()):
        assert torch.allclose(a.grad,b.grad,atol=1e-7,rtol=2e-5)
    for a,b in zip(model.parameters(),reference_top.parameters()):
        assert torch.allclose(a,b,atol=1e-7,rtol=2e-5)
    assert server.version == 1
    assert all(s['step'].item()==1 for s in server.optimizer.state.values())
    assert server.batchnorm.summary()['forward_calls']=={'sd':1,'proto':1}
    stats=server.finish_round()
    assert stats['sample_counts_by_class']==[31,19,6]  # Never count sampled SD twice.
    # Every unselected feature still receives the class-mean path exactly once.
    mask=torch.ones(56,dtype=torch.bool); mask[index]=False
    cgrad=reply['grad_prototypes'][labels]/counts[labels,None,None,None]
    assert torch.allclose(z.grad[mask],cgrad[mask],atol=1e-7)
    assert torch.allclose(z.grad[index],cgrad[index]+reply['grad_selected_sd'],atol=1e-7)


@pytest.mark.parametrize('bad', ['duplicate_index','negative_index','out_of_range',
    'wrong_selected_label','wrong_count','wrong_ratio','nan_sd','wrong_dtype'])
def test_hybrid_invalid_packet_rejected_before_any_update(cpu,bad):
    from utils.na_sfl_prototype import PrototypeServer
    cfg={**config(),'num_classes':3,'boundary_shape_per_sample':(2,2,2)}
    server=PrototypeServer(top(),cfg); server.start_round(1)
    req=packet(torch.randn(56,2,2,2),torch.tensor([0]*40+[2]*16))
    if bad=='duplicate_index': req['selected_indices'][1]=req['selected_indices'][0]
    if bad=='negative_index': req['selected_indices'][0]=-1
    if bad=='out_of_range': req['selected_indices'][0]=56
    if bad=='wrong_selected_label': req['selected_labels'][0]=1
    if bad=='wrong_count': req['counts'][0]+=1
    if bad=='wrong_ratio': req['sample_ratio']=.2
    if bad=='nan_sd': req['selected_sd'][0,0,0,0]=float('nan')
    if bad=='wrong_dtype': req['selected_sd']=req['selected_sd'].double()
    with pytest.raises(ValueError): server.process_batch([req])
    assert server.version==0 and not server.optimizer.state
    assert server.batchnorm.summary()['forward_calls']=={'sd':0,'proto':0}


def test_q_zero_uses_exact_original_dual_proto_only_loss(cpu):
    from utils.na_sfl_prototype import PrototypeServer, make_local_prototypes
    cfg={**config(0),'num_classes':3,'boundary_shape_per_sample':(2,2,2)}
    a=PrototypeServer(top(),cfg)
    old={**cfg,'method':'na_sfl_v3_proto'}
    b=PrototypeServer(copy.deepcopy(a.model),old)
    z,y=torch.randn(56,2,2,2),torch.tensor([0]*40+[2]*16)
    p,c,n=make_local_prototypes(z,y)
    req={'job_id':(1,0,0),'job_kind':'PROTO_JOB','activation':p,'labels':c,
        'counts':n,'actual_batch_size':56,'return_cut_gradient':True}
    for s in (a,b): s.start_round(1)
    left=a.process_batch([req])['jobs'][0];right=b.process_batch([req])['jobs'][0]
    assert left['L_total']==right['L_total']
    assert torch.equal(left['cut_gradient'],right['cut_gradient'])
    assert all(torch.equal(v,b.model.state_dict()[k]) for k,v in a.model.state_dict().items())
