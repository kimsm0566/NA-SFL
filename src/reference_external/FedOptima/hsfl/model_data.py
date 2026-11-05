"""Matched HSFL CIFAR models and local data for the original FedOptima runtime.

Both the HSFL ``src`` directory and the FedOptima directory must be on
``sys.path``. Importing this module does not inspect CUDA or load datasets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.data import build_loader, partition_metadata
from data.na_sfl_data import balanced_partitions, load_local_cifar10
from models.model import ClientModel, FullModel, ServerModel, load_matching_state


# torchvision augmentation uses the process CPU RNG. All adapter augmentation
# and model construction share this lock; the fork never touches CUDA RNGs.
CPU_RNG_LOCK = threading.RLock()


def _seed(seed: int, *parts) -> int:
    """The exact SFLV2Trainer._seed derivation, without constructing a trainer."""
    text = ':'.join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63 - 1)


class AuxiliaryNetwork(nn.Module):
    """Original ResNet auxiliary Conv/BN/ReLU/pool/dense idea at cut2.

    The author's networks/model_config.py ResNet18 auxiliary uses32 output
    channels and maxpool2. The changed input128x16x16 therefore needs a2048
    feature classifier. Layer defaults and ResNet initialization follow
    networks/layers.py and networks/resnet.py, without importing NLP packages.
    """

    model_type = 'auxiliary_net'

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(32 * 8 * 8, 10)
        # Original ResNet overrides Conv weights and BN affine only. In
        # particular, Conv bias and Dense retain their constructor defaults.
        nn.init.kaiming_normal_(self.features[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.features[1].weight, 1)
        nn.init.constant_(self.features[1].bias, 0)

    def forward(self, inputs):
        return self.classifier(torch.flatten(self.features(inputs), start_dim=1))


def make_models(seed=42, cut=2):
    """Return complete/client/server/auxiliary_net CPU modules for FedOptima.

    Complete initializes from seed exactly as NASFLTrainer does. Client and
    Server load its corresponding ordered state keys; their random constructor
    weights are discarded. No state key is renamed and no CUDA call is made.
    Runtime owns moving/deep-copying these modules onto the selected device.
    """
    if type(cut) is not int or cut != 2:
        raise ValueError('the matched FedOptima adapter requires cut=2')
    with CPU_RNG_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        complete = FullModel('resnet18_cifar')
        state = complete.state_dict()
        server = ServerModel('resnet18_cifar', cut)
        load_matching_state(server, state)
        client = ClientModel('resnet18_cifar', cut)
        load_matching_state(client, state)
        # Aux follows FedOptima, rather than the different NA auxiliary head.
        torch.random.default_generator.manual_seed(_seed(seed, 'fedoptima_auxiliary_initialization'))
        auxiliary = AuxiliaryNetwork()
    result = {'complete': complete, 'client': client, 'server': server, 'auxiliary_net': auxiliary}
    for name, model in result.items():
        model.model_type = name
    return result


class EpochLoader:
    """Two-tensor batches with isolated deterministic per-client epochs.

    One client owns one loader. ``epoch`` is1-based and increments when an
    iterator first advances; ``batch_index`` is0-based. ``last_batch_size`` and
    ``last_sample_ids`` describe the most recently yielded batch. Accounting
    must use those sizes (or len(labels)), because the last batch has56 samples.

    FedOptima clients have independent epochs. Epoch e uses NA's round e,
    local_epoch0 seed contract. This reproduces the same first-epoch order and
    augmentation, without claiming asynchronously advanced epochs align with
    the global30000-visit evaluation rounds.
    """

    def __init__(self, dataset, *, seed, client_id, batch_size, train):
        self.dataset = dataset
        self.seed, self.client_id = seed, client_id
        self.batch_size, self.train = batch_size, train
        self.num_workers, self.drop_last = 0, False
        self.epoch, self.batch_index, self.last_batch_size = 0, None, 0
        self.last_sample_ids = []
        self._epoch_lock = threading.Lock()

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        with self._epoch_lock:
            self.epoch += 1
            epoch = self.epoch
        key = ('shuffle', epoch, self.client_id, 0) if self.train else ('evaluation',)
        generator = torch.Generator(device='cpu').manual_seed(_seed(self.seed, *key))
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.train,
                            drop_last=False, num_workers=0, pin_memory=True, generator=generator)
        iterator = iter(loader)
        index = 0
        while True:
            with CPU_RNG_LOCK, torch.random.fork_rng(devices=[]):
                if self.train:
                    torch.random.default_generator.manual_seed(
                        _seed(self.seed, 'augmentation', epoch, self.client_id, 0, index))
                try:
                    images, labels, ids = next(iterator)
                except StopIteration:
                    break
            self.batch_index, self.last_batch_size = index, len(labels)
            self.last_sample_ids = ids.tolist()
            yield images, labels
            index += 1


def make_loaders(seed=42, data_path='/workspace/HSFL/src/datasets', batch_size=128):
    """Return ten training loaders, official Test loader and JSON-safe identity.

    CIFAR is loaded locally with download=False. The exact NA balanced
    Dirichletalpha.1 allocation selects30000 disjoint train IDs,3000/client.
    The metadata data_sha256 uses the same four arrays+str(partitions) digest
    as NASFLTrainer.run; partition_sha256 uses the same compact JSON as NA.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    images, labels, test_images, test_labels = load_local_cifar10(Path(data_path))
    if len(images) != 50000 or len(test_images) != 10000:
        raise ValueError('canonical CIFAR10 train50000/test10000 splits required')
    parts = balanced_partitions(labels, 10, 3000, .1, seed, 'dirichlet')
    ids = [part.tolist() for part in parts]
    training = []
    for client_id, part in enumerate(parts):
        generator = torch.Generator(device='cpu').manual_seed(_seed(seed, 'shuffle', 1, client_id, 0))
        dataset = build_loader(images[part], labels[part], part, batch_size, True, 0, generator).dataset
        training.append(EpochLoader(dataset, seed=seed, client_id=client_id, batch_size=batch_size, train=True))
    generator = torch.Generator(device='cpu').manual_seed(_seed(seed, 'evaluation'))
    test_dataset = build_loader(test_images, test_labels, np.arange(len(test_labels)),
                                batch_size, False, 0, generator).dataset
    test = EpochLoader(test_dataset, seed=seed, client_id=None, batch_size=batch_size, train=False)
    data_digest = hashlib.sha256()
    for value in (images, labels, test_images, test_labels):
        data_digest.update(np.ascontiguousarray(value).tobytes())
    data_digest.update(str(ids).encode())
    metadata = {
        'seed': seed, 'batch_size': batch_size, 'n_clients': 10, 'samples_per_client': 3000,
        'samples_per_round': 30000, 'drop_last': False, 'num_workers': 0,
        'partition_ids': ids, 'train_labels': labels.tolist(),
        'partition_metadata': partition_metadata(labels, parts, 'dirichlet', .1, seed),
        'partition_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
        'data_sha256': data_digest.hexdigest(),
        'augmentation': 'NA RandomCrop32 padding4 + RandomHorizontalFlip + ToTensor + Normalize',
        'epoch_seed_contract': 'NA round=client_epoch(1-based), client_id, local_epoch=0',
        'test_samples': len(test_labels), 'local_download': False,
    }
    return training, test, metadata
