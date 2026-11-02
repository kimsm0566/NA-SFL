"""Bounded CPU residual snapshots for NA-SFLv4 Pseudo-SD training.

SHA256-keyed priority sampling is an order-independent uniform reservoir: the
smallest K pseudorandom priorities survive. Separate keys choose pseudo donors
without replacement. Neither operation consumes model/DataLoader RNG state.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch

from utils.na_sfl_prototype import _feature_shape, validate_prototypes
from utils.sflv2 import _finite


_ALGORITHM = 'sha256_keyed_priority_v1'


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).hexdigest()


def _job(job, round_id):
    if (not isinstance(job, tuple) or len(job) != 3
            or any(type(i) is not int or i < 0 for i in job) or job[0] != round_id):
        raise ValueError('residual job must match active round/client/batch')


class ResidualBank:
    """Only previous-completed-round donors are visible to current pseudo jobs.

    Entries contain one detached CPU FP32 residual plus its source job/sample ID
    and priority. Snapshot and builder each have at most K entries per class;
    a full SD batch is never retained. State stores the keyed RNG algorithm and
    seed rather than a mutable generator position.
    """

    def __init__(self, num_classes, feature_shape, *, seed=42,
                 max_residuals_per_class=64, min_residuals_per_class=2):
        if type(num_classes) is not int or num_classes < 1:
            raise ValueError('num_classes must be positive')
        if type(seed) is not int or seed < 0:
            raise ValueError('residual seed must be a nonnegative integer')
        if type(max_residuals_per_class) is not int or max_residuals_per_class < 2:
            raise ValueError('residual capacity must be at least two')
        if (type(min_residuals_per_class) is not int or min_residuals_per_class < 2
                or min_residuals_per_class > max_residuals_per_class):
            raise ValueError('minimum donors must be between two and residual capacity')
        self.num_classes, self.feature_shape = num_classes, _feature_shape(feature_shape)
        self.seed, self.capacity = seed, max_residuals_per_class
        self.minimum = min_residuals_per_class
        self.capacity_memory_bytes = num_classes * self.capacity * math.prod(self.feature_shape) * 4
        self.bank = [[] for _ in range(num_classes)]
        self.builder = [[] for _ in range(num_classes)]
        self.version, self.round_id, self.bank_round_id, self.active = 0, None, None, False
        self.seen_jobs = set()

    def _require_active(self):
        if not self.active:
            raise RuntimeError('residual operation requires active round')

    def start_round(self, round_id):
        if self.active:
            raise RuntimeError('previous residual round remains active')
        if (type(round_id) is not int or round_id < 0
                or (self.round_id is not None and round_id <= self.round_id)):
            raise ValueError('residual round must increase')
        self.round_id, self.active = round_id, True
        self.builder = [[] for _ in range(self.num_classes)]
        self.seen_jobs.clear()

    @torch.no_grad()
    def accumulate(self, sd, labels, protos, classes, counts, job_id):
        self._require_active(); _job(job_id, self.round_id)
        if job_id in self.seen_jobs:
            raise ValueError('duplicate Fast residual job')
        if (not isinstance(sd, torch.Tensor) or sd.dtype != torch.float32
                or tuple(sd.shape[1:]) != self.feature_shape or len(sd) < 1
                or not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
                or labels.device != sd.device or labels.shape != (len(sd),)):
            raise ValueError('invalid residual SD/labels')
        validate_prototypes(protos, classes, counts, self.feature_shape, self.num_classes, len(sd))
        if protos.device != sd.device:
            raise ValueError('residual SD/prototypes device mismatch')
        _finite(sd, 'Fast residual SD')
        if not torch.equal(torch.unique(labels, sorted=True), classes):
            raise ValueError('residual classes differ from Fast SD labels')
        pending = {}
        for row, c in enumerate(classes.tolist()):
            indices = torch.nonzero(labels == c, as_tuple=False).flatten().tolist()
            if len(indices) != counts[row].item():
                raise ValueError('residual class counts differ from Fast SD labels')
            if len(indices) < 2:
                continue
            candidates = list(self.builder[c])
            for index in indices:
                donor_id = (*job_id, index)
                candidates.append({'id': list(donor_id),
                    'priority': _key('bank', self.seed, self.round_id, c, donor_id), 'index': index})
            selected = sorted(candidates, key=lambda entry: (entry['priority'], entry['id']))[:self.capacity]
            entries = []
            for entry in selected:
                if 'index' in entry:
                    residual = (sd[entry['index']].detach() - protos[row].detach()).cpu().clone()
                    _finite(residual, 'Fast residual')
                    entries.append({'id': entry['id'], 'priority': entry['priority'], 'residual': residual})
                else:
                    entries.append(entry)
            pending[c] = entries
        for c, entries in pending.items():
            self.builder[c] = entries
        self.seen_jobs.add(job_id)

    def make_pseudo(self, protos, classes, counts, job_id, config):
        self._require_active(); _job(job_id, self.round_id)
        validate_prototypes(protos, classes, counts, self.feature_shape, self.num_classes,
                            int(counts.sum().item()))
        maximum, minimum = config['pseudo_samples_per_class'], config['min_residuals_per_class']
        if type(maximum) is not int or maximum < 2 or type(minimum) is not int or minimum < 2:
            raise ValueError('pseudo sample count and minimum donors must be at least two')
        if type(config['pseudo_gradient_to_client']) is not bool or type(config['residual_recenter']) is not bool:
            raise ValueError('pseudo gradient/recenter flags must be bool')
        scale = config['residual_scale']
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
            raise ValueError('residual scale must be finite and nonnegative')
        generated, labels, weights, fallbacks, reasons, donors = [], [], [], [], {}, []
        eligible_count = 0
        for row, c in enumerate(classes.tolist()):
            available = self.bank[c]
            if len(available) < minimum:
                fallbacks.append(c)
                reasons[str(c)] = ('empty_previous_round_bank' if not available else 'insufficient_same_class_residuals')
                continue
            selected = sorted(available, key=lambda e: (_key('pseudo', self.seed, self.round_id,
                job_id, c, e['id']), e['id']))[:min(maximum, len(available))]
            residuals = torch.stack([entry['residual'] for entry in selected]).to(protos.device).detach()
            if config['residual_recenter']:
                residuals = residuals - residuals.mean(dim=0, keepdim=True)
            base = protos[row] if config['pseudo_gradient_to_client'] else protos[row].detach()
            pseudo = base.unsqueeze(0) + scale * residuals
            _finite(pseudo, 'pseudo SD')
            generated.append(pseudo)
            labels.append(torch.full((len(selected),), c, dtype=torch.int64, device=protos.device))
            count = int(counts[row].item()); eligible_count += count
            weights.append(torch.full((len(selected),), count / len(selected),
                dtype=torch.float32, device=protos.device))
            donors.append({'class': c, 'count': len(selected), 'available_count': len(available),
                'selected_ids': [entry['id'][:] for entry in selected],
                'selected_priorities': [entry['priority'] for entry in selected]})
        return {'activation': torch.cat(generated) if generated else None,
            'labels': torch.cat(labels) if labels else None,
            'weights': torch.cat(weights) / eligible_count if weights else None,
            'pseudo_samples_generated': sum(len(tensor) for tensor in generated),
            'pseudo_fallback_classes': fallbacks, 'pseudo_fallback_reasons': reasons,
            'pseudo_donors': donors, 'bank_version': self.version,
            'pseudo_gradient_to_client': config['pseudo_gradient_to_client']}

    def _bytes(self, bank):
        return sum(entry['residual'].numel() * entry['residual'].element_size()
                   for entries in bank for entry in entries)

    def stats(self):
        sizes = [len(entries) for entries in self.bank]
        return {'version': self.version, 'source_round': self.bank_round_id,
            'residual_bank_size_by_class': sizes,
            'class_donor_coverage': [size >= self.minimum for size in sizes],
            'memory_bytes': self._bytes(self.bank), 'builder_memory_bytes': self._bytes(self.builder),
            'capacity_memory_bytes': self.capacity_memory_bytes,
            'max_residuals_per_class': self.capacity, 'seed': self.seed,
            'selection_algorithm': _ALGORITHM,
            'selected_donors': [{'class': c, 'ids': [entry['id'][:] for entry in entries],
                'priorities': [entry['priority'] for entry in entries]}
                for c, entries in enumerate(self.bank)]}

    def finish_round(self):
        self._require_active()
        self.bank, self.builder = self.builder, [[] for _ in range(self.num_classes)]
        self.bank_round_id, self.active = self.round_id, False
        self.version += 1
        return self.stats()

    @staticmethod
    def _copy_bank(bank):
        return [[{'id': entry['id'][:], 'priority': entry['priority'],
                  'residual': entry['residual'].detach().cpu().clone()} for entry in entries] for entries in bank]

    def state_dict(self):
        return {'num_classes': self.num_classes, 'feature_shape': self.feature_shape,
            'capacity': self.capacity, 'minimum': self.minimum,
            'selection_rng_state': {'algorithm': _ALGORITHM, 'seed': self.seed},
            'version': self.version, 'round_id': self.round_id,
            'bank_round_id': self.bank_round_id, 'active': self.active,
            'seen_jobs': sorted(self.seen_jobs),
            'bank': self._copy_bank(self.bank), 'builder': self._copy_bank(self.builder)}

    def load_state_dict(self, state):
        if (state['num_classes'] != self.num_classes or tuple(state['feature_shape']) != self.feature_shape
                or state['capacity'] != self.capacity or state['minimum'] != self.minimum
                or state['selection_rng_state'] != {'algorithm': _ALGORITHM, 'seed': self.seed}):
            raise ValueError('residual checkpoint layout/capacity/RNG differs')
        version, round_id, source, active = state['version'], state['round_id'], state['bank_round_id'], state['active']
        if (type(version) is not int or version < 0 or type(active) is not bool
                or any(value is not None and (type(value) is not int or value < 0) for value in (round_id, source))
                or (active and round_id is None) or (version == 0 and source is not None)
                or (version > 0 and source is None) or (source is not None and (round_id is None or source > round_id))
                or (active and source is not None and source >= round_id)
                or (not active and source != round_id)):
            raise ValueError('invalid residual checkpoint lifecycle')
        seen = set()
        for job in state['seen_jobs']:
            job = tuple(job); _job(job, round_id)
            if job in seen:
                raise ValueError('duplicate residual checkpoint job')
            seen.add(job)
        loaded = {}
        for name, expected_round in (('bank', source), ('builder', round_id)):
            bank = state[name]
            if not isinstance(bank, list) or len(bank) != self.num_classes:
                raise ValueError('residual checkpoint class layout differs')
            if not active and name == 'builder' and any(bank):
                raise ValueError('inactive residual checkpoint has a nonempty builder')
            ids = set()
            for c, entries in enumerate(bank):
                if not isinstance(entries, list) or len(entries) > self.capacity:
                    raise ValueError('residual checkpoint class capacity exceeded')
                for entry in entries:
                    donor = entry['id']; residual = entry['residual']
                    if (not isinstance(donor, list) or len(donor) != 4
                            or any(type(i) is not int or i < 0 for i in donor)
                            or expected_round is None or donor[0] != expected_round
                            or tuple(donor) in ids):
                        raise ValueError('invalid residual checkpoint donor ID')
                    ids.add(tuple(donor))
                    if (not isinstance(residual, torch.Tensor) or residual.dtype != torch.float32
                            or tuple(residual.shape) != self.feature_shape or residual.device.type != 'cpu'
                            or residual.requires_grad or residual.grad_fn is not None):
                        raise ValueError('residual checkpoint must contain detached CPU FP32 features')
                    _finite(residual, 'residual checkpoint')
                    if entry['priority'] != _key('bank', self.seed, donor[0], c, tuple(donor)):
                        raise ValueError('residual checkpoint donor priority differs')
                    if name == 'builder' and tuple(donor[:3]) not in seen:
                        raise ValueError('residual builder donor has no saved source job')
                keys = [(entry['priority'], entry['id']) for entry in entries]
                if keys != sorted(keys):
                    raise ValueError('residual checkpoint priority order differs')
            if self._bytes(bank) > self.capacity_memory_bytes:
                raise ValueError('residual checkpoint byte budget exceeded')
            loaded[name] = self._copy_bank(bank)
        self.bank, self.builder = loaded['bank'], loaded['builder']
        self.version, self.round_id, self.bank_round_id, self.active = version, round_id, source, active
        self.seen_jobs = seen
