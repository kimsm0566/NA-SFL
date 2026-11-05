"""Typed v3/v4 tensors carried by the existing checked TCP frame protocol."""
import torch


_KINDS = {'SD_JOB', 'PROTO_JOB', 'SD_GRADIENT', 'PROTO_GRADIENT', 'HYBRID_JOB', 'HYBRID_GRADIENT'}
_FIELDS = {'kind', 'run_id', 'round', 'client_id', 'batch_id', 'job_id',
           'actual_batch_size', 'feature_shape'}


def validate_message(tensors, metadata):
    if not isinstance(metadata, dict) or metadata.get('kind') not in _KINDS:
        raise ValueError('invalid typed job metadata')
    hybrid = metadata['kind'].startswith('HYBRID_')
    fields = _FIELDS | {'sample_ratio', 'sample_seed'} if hybrid else _FIELDS
    if set(metadata) != fields:
        raise ValueError('invalid typed job metadata')
    run = metadata['run_id']
    if not isinstance(run, str) or not run or len(run) > 256:
        raise ValueError('invalid typed run identity')
    for field in ('round', 'client_id', 'batch_id', 'actual_batch_size'):
        value = metadata[field]
        if type(value) is not int or value < (1 if field in ('round', 'actual_batch_size') else 0):
            raise ValueError('invalid typed ' + field)
    identity = f'{run}:{metadata["round"]}:{metadata["client_id"]}:{metadata["batch_id"]}'
    if metadata['job_id'] != identity:
        raise ValueError('typed job identity mismatch')
    shape = metadata['feature_shape']
    if not isinstance(shape, list) or len(shape) != 3 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError('invalid typed feature shape')
    kind = metadata['kind']
    if hybrid:
        _validate_hybrid(tensors, metadata)
        return
    keys = {'SD_JOB': {'sd', 'labels'}, 'PROTO_JOB': {'prototypes', 'classes', 'counts'},
            'SD_GRADIENT': {'gradient'}, 'PROTO_GRADIENT': {'gradient', 'classes'}}[kind]
    if not isinstance(tensors, dict) or set(tensors) != keys:
        raise ValueError('typed tensor keys do not match kind')
    name = 'sd' if kind == 'SD_JOB' else 'prototypes' if kind == 'PROTO_JOB' else 'gradient'
    feature = tensors[name]
    if (not isinstance(feature, torch.Tensor) or feature.dtype != torch.float32
            or feature.ndim != 4 or list(feature.shape[1:]) != shape or feature.shape[0] < 1
            or not torch.isfinite(feature).all().item()):
        raise ValueError('invalid typed feature/gradient tensor')
    n = feature.shape[0]
    if kind in ('SD_JOB', 'SD_GRADIENT') and n != metadata['actual_batch_size']:
        raise ValueError('SD size differs from actual batch size')
    for name in ('labels', 'classes'):
        if name not in tensors:
            continue
        labels = tensors[name]
        if (labels.dtype != torch.int64 or labels.shape != (n,)
                or (labels < 0).any().item() or (labels >= 10).any().item()):
            raise ValueError('invalid typed labels/classes')
        if name == 'classes' and n > 1 and not (labels[1:] > labels[:-1]).all().item():
            raise ValueError('prototype classes must be sorted and distinct')
    if kind == 'PROTO_JOB':
        counts = tensors['counts']
        if (counts.dtype != torch.int64 or counts.shape != (n,) or (counts <= 0).any().item()
                or counts.sum().item() != metadata['actual_batch_size']):
            raise ValueError('invalid prototype counts or batch size')


def request_message(packet):
    row = packet['row']
    job = row['job_id']
    if (not isinstance(job, (list, tuple)) or len(job) != 3
            or any(type(x) is not int for x in job) or job[1] != row['client_id']):
        raise ValueError('typed job identity differs from Client')
    meta = dict(kind=packet['job_kind'], run_id=row['run_id'], round=job[0], client_id=job[1],
                batch_id=job[2], job_id=f'{row["run_id"]}:{job[0]}:{job[1]}:{job[2]}',
                actual_batch_size=row['samples'], feature_shape=list(packet['activation'].shape[1:]))
    if meta['kind'] == 'SD_JOB':
        tensors = {'sd': packet['activation'], 'labels': packet['labels']}
    elif meta['kind'] == 'PROTO_JOB':
        tensors = {'prototypes': packet['activation'], 'classes': packet['labels'], 'counts': packet['counts']}
    elif meta['kind'] == 'HYBRID_JOB':
        required = ('counts', 'selected_sd', 'selected_labels', 'selected_indices', 'sample_ratio')
        if any(key not in packet for key in required):
            raise ValueError('Hybrid request is missing fields')
        meta.update(sample_ratio=packet['sample_ratio'], sample_seed=packet.get('sample_seed', 42))
        if packet.get('actual_batch_size', meta['actual_batch_size']) != meta['actual_batch_size']:
            raise ValueError('Hybrid actual batch size differs from row')
        tensors = {'prototypes': packet['activation'], 'classes': packet['labels'],
            **{key: packet[key] for key in ('counts', 'selected_sd', 'selected_labels', 'selected_indices')}}
    else:
        raise ValueError('expected typed request kind')
    validate_message(tensors, meta)
    return tensors, meta


def reply_message(request_metadata, gradient, classes=None):
    if request_metadata['kind'] == 'HYBRID_JOB':
        meta = {**request_metadata, 'kind': 'HYBRID_GRADIENT'}
        validate_message(gradient, meta)
        return gradient, meta
    kind = 'PROTO_GRADIENT' if request_metadata['kind'] == 'PROTO_JOB' else 'SD_GRADIENT'
    tensors = {'gradient': gradient}
    if kind == 'PROTO_GRADIENT':
        tensors['classes'] = classes
    meta = {**request_metadata, 'kind': kind}
    validate_message(tensors, meta)
    return tensors, meta


def _validate_hybrid(tensors, metadata):
    from utils.na_sfl_hybrid import sample_count
    m = sample_count(metadata['actual_batch_size'], metadata['sample_ratio'])
    if m == 0 or type(metadata['sample_seed']) is not int or metadata['sample_seed'] < 0:
        raise ValueError('Hybrid request requires a positive sample ratio and valid seed')
    job = metadata['kind'] == 'HYBRID_JOB'
    keys = ({'prototypes', 'classes', 'counts', 'selected_sd', 'selected_labels', 'selected_indices'} if job
            else {'grad_prototypes', 'classes', 'grad_selected_sd', 'selected_indices'})
    if not isinstance(tensors, dict) or set(tensors) != keys:
        raise ValueError('Hybrid tensor keys do not match kind')
    proto = tensors['prototypes' if job else 'grad_prototypes']
    selected = tensors['selected_sd' if job else 'grad_selected_sd']
    for feature in (proto, selected):
        if (not isinstance(feature, torch.Tensor) or feature.dtype != torch.float32 or feature.ndim != 4
                or list(feature.shape[1:]) != metadata['feature_shape'] or feature.shape[0] < 1
                or not torch.isfinite(feature).all().item()):
            raise ValueError('invalid Hybrid feature or gradient')
    n = proto.shape[0]
    if selected.shape[0] != m:
        raise ValueError('Hybrid SD count differs from sampling ratio')
    classes, indices = tensors['classes'], tensors['selected_indices']
    if (not isinstance(classes, torch.Tensor) or classes.dtype != torch.int64 or classes.shape != (n,)
            or (classes < 0).any().item() or (classes >= 10).any().item()
            or (n > 1 and not (classes[1:] > classes[:-1]).all().item())):
        raise ValueError('invalid Hybrid classes')
    if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int32 or indices.shape != (m,)
            or (indices < 0).any().item() or (indices >= metadata['actual_batch_size']).any().item()
            or indices.unique().numel() != m):
        raise ValueError('invalid Hybrid selected indices')
    if job:
        counts, labels = tensors['counts'], tensors['selected_labels']
        if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.int64 or counts.shape != (n,)
                or (counts <= 0).any().item() or counts.sum().item() != metadata['actual_batch_size']):
            raise ValueError('invalid Hybrid full-batch counts')
        if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64 or labels.shape != (m,)
                or not torch.isin(labels, classes).all().item()):
            raise ValueError('invalid Hybrid selected labels')
        if any((labels == c).sum().item() > count.item() for c, count in zip(classes, counts)):
            raise ValueError('Hybrid selected class count exceeds full-batch count')
