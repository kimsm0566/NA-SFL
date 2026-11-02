"""Typed v3/v4 tensors carried by the existing checked TCP frame protocol."""
import torch


_KINDS = {'SD_JOB', 'PROTO_JOB', 'SD_GRADIENT', 'PROTO_GRADIENT'}
_FIELDS = {'kind', 'run_id', 'round', 'client_id', 'batch_id', 'job_id',
           'actual_batch_size', 'feature_shape'}


def validate_message(tensors, metadata):
    if not isinstance(metadata, dict) or set(metadata) != _FIELDS or metadata['kind'] not in _KINDS:
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
    else:
        raise ValueError('expected typed request kind')
    validate_message(tensors, meta)
    return tensors, meta


def reply_message(request_metadata, gradient, classes=None):
    kind = 'PROTO_GRADIENT' if request_metadata['kind'] == 'PROTO_JOB' else 'SD_GRADIENT'
    tensors = {'gradient': gradient}
    if kind == 'PROTO_GRADIENT':
        tensors['classes'] = classes
    meta = {**request_metadata, 'kind': kind}
    validate_message(tensors, meta)
    return tensors, meta
