"""Capacity-balanced class-wise Dirichlet; immutable local CIFAR10 input."""
from pathlib import Path

import numpy as np


def balanced_partitions(labels, n_clients, samples_per_client, alpha, seed, partition='dirichlet'):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not len(labels) or labels.dtype.kind not in 'iu'
            or type(n_clients) is not int or n_clients < 1
            or type(samples_per_client) is not int or samples_per_client < 1
            or n_clients * samples_per_client > len(labels)
            or not np.isfinite(alpha) or alpha <= 0 or partition not in ('iid', 'dirichlet')):
        raise ValueError('invalid balanced partition request')
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(labels))[:n_clients * samples_per_client]
    if partition == 'iid':
        return [p.copy() for p in np.split(selected, n_clients)]
    classes = np.unique(labels[selected])
    pools = [selected[labels[selected] == cls] for cls in classes]
    row_totals = np.array([len(p) for p in pools], dtype=np.float64)
    # Iterative proportional fitting preserves the drawn class/client preference
    # cross-ratios while imposing exact selected-class and client capacity margins.
    # Flooring at1e-12 makes extreme alpha draws numerically well defined.
    matrix = np.maximum(rng.dirichlet(np.full(n_clients, alpha), size=len(classes)), 1e-12)
    for _ in range(20000):
        matrix *= (row_totals / matrix.sum(axis=1))[:, None]
        matrix *= (samples_per_client / matrix.sum(axis=0))[None, :]
        if np.max(np.abs(matrix.sum(axis=1) - row_totals)) < 1e-8:
            break
    else:
        raise ValueError('balanced Dirichlet matrix did not converge')
    counts = np.floor(matrix).astype(np.int64)
    rows = row_totals.astype(np.int64) - counts.sum(axis=1)
    cols = samples_per_client - counts.sum(axis=0)
    fractions = matrix - counts
    while rows.sum():
        scores = np.where((rows[:, None] > 0) & (cols[None, :] > 0), fractions, -np.inf)
        row, col = np.unravel_index(np.argmax(scores), scores.shape)
        if not np.isfinite(scores[row, col]):
            raise RuntimeError('balanced integer allocation has inconsistent margins')
        counts[row, col] += 1
        rows[row] -= 1; cols[col] -= 1
        fractions[row, col] -= 1.
    parts = [[] for _ in range(n_clients)]
    for pool, allocation in zip(pools, counts):
        for cid, ids in enumerate(np.split(rng.permutation(pool), np.cumsum(allocation)[:-1])):
            parts[cid].extend(ids.tolist())
    result = [rng.permutation(p).astype(np.int64) for p in parts]
    if any(len(p) != samples_per_client for p in result) or len(np.unique(np.concatenate(result))) != len(selected):
        raise RuntimeError('balanced partition violated exact disjoint quotas')
    return result


def load_local_cifar10(data_path):
    from torchvision.datasets import CIFAR10
    train = CIFAR10(str(Path(data_path) / 'cifar10'), train=True, download=False)
    test = CIFAR10(str(Path(data_path) / 'cifar10'), train=False, download=False)
    return train.data, np.asarray(train.targets, dtype=np.int64), test.data, np.asarray(test.targets, dtype=np.int64)
