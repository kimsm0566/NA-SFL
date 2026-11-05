"""FedOptima matched model/data contracts; no model forward or backward."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / 'reference_external' / 'FedOptima'))
from data.na_sfl_data import balanced_partitions
from data.data import build_loader
from models.model import FullModel
from utils.sflv2 import SFLV2Trainer


def adapter():
    try:
        return importlib.import_module('hsfl.model_data')
    except ModuleNotFoundError as exc:
        pytest.fail(f'FedOptima model/data adapter is missing: {exc}')


def digest(model):
    result = hashlib.sha256()
    for key, value in model.state_dict().items():
        array = value.detach().contiguous().numpy()
        result.update(key.encode() + str(array.dtype).encode() + str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


@pytest.fixture(scope='module')
def models():
    return adapter().make_models()


@pytest.fixture(scope='module')
def loaders():
    return adapter().make_loaders(data_path=SRC / 'datasets')


def test_models_match_na_initial_state_and_ordered_split_keys(models):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        expected = FullModel('resnet18_cifar').state_dict()
    assert set(models) == {'complete', 'client', 'server', 'auxiliary_net'}
    for name, model in models.items():
        assert isinstance(model, nn.Module) and model.model_type == name
    assert list(models['client'].state_dict()) + list(models['server'].state_dict()) == list(expected)
    for side in ('complete', 'client', 'server'):
        assert all(torch.equal(tensor, expected[key]) for key, tensor in models[side].state_dict().items())
    assert models['client'].features.conv1.kernel_size == (3, 3)
    assert list(models['client'].features._modules)[-1] == 'layer2'
    assert models['complete'].classifier.fc.out_features == 10


def test_model_initialization_preserves_global_cpu_rng():
    before = torch.get_rng_state().clone()
    first = adapter().make_models(seed=41)
    assert torch.equal(before, torch.get_rng_state())
    second = adapter().make_models(seed=41)
    assert digest(first['client']) == digest(second['client'])
    assert digest(first['auxiliary_net']) == digest(second['auxiliary_net'])
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('cut', [1, 3, True, 0])
def test_unsupported_cuts_rejected(cut):
    with pytest.raises(ValueError, match='cut'):
        adapter().make_models(cut=cut)


def test_auxiliary_uses_original_conv_bn_pool_classifier_concept(models):
    modules = list(models['auxiliary_net'].modules())
    conv = next(layer for layer in modules if isinstance(layer, nn.Conv2d))
    bn = next(layer for layer in modules if isinstance(layer, nn.BatchNorm2d))
    pool = next(layer for layer in modules if isinstance(layer, nn.MaxPool2d))
    classifier = next(layer for layer in modules if isinstance(layer, nn.Linear))
    assert (conv.in_channels, conv.out_channels, conv.kernel_size) == (128, 32, (3, 3))
    assert conv.bias is not None and bn.num_features == 32
    assert any(isinstance(layer, nn.ReLU) for layer in modules)
    assert pool.kernel_size == 2 and pool.stride == 2
    assert (classifier.in_features, classifier.out_features) == (32 * 8 * 8, 10)


def test_local_partitions_and_baseline_identity(loaders, models):
    training, test, metadata = loaders
    assert len(training) == 10 and len(test.dataset) == 10000
    expected = balanced_partitions(metadata['train_labels'], 10, 3000, .1, 42)
    assert [len(loader.dataset) for loader in training] == [3000] * 10
    assert [len(loader) for loader in training] == [24] * 10
    assert all(loader.num_workers == 0 and not loader.drop_last for loader in training)
    assert all(np.array_equal(loader.dataset.sample_ids, ids) for loader, ids in zip(training, expected))
    assert len(np.unique(np.concatenate(expected))) == 30000
    ids = [part.tolist() for part in expected]
    assert json.loads(json.dumps(metadata))['partition_ids'] == ids
    assert metadata['partition_sha256'] == hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    # Existing raw results are independent evidence, when available locally.
    baselines = sorted((SRC / 'results' / '20261104_proto_random_sd10_seed42' / 'run_100round').glob('*/summary.json'))
    if baselines:
        baseline = json.loads(baselines[0].read_text())
        assert metadata['partition_sha256'] == baseline['partition_sha256']
        assert metadata['data_sha256'] == baseline['data_sha256']
        assert digest(models['client']) == baseline['initial_model_hashes']['client']
        assert digest(models['server']) == baseline['initial_model_hashes']['server']


def test_epoch_retains_partial_batch_and_exposes_sample_accounting(loaders):
    training, _, _ = loaders
    loader = training[9]
    sizes, seen = [], []
    for images, labels in loader:
        sizes.append(len(labels))
        assert images.shape == (len(labels), 3, 32, 32)
        assert loader.last_batch_size == len(labels)
        seen.extend(loader.last_sample_ids)
    assert sizes == [128] * 23 + [56]
    assert sorted(seen) == sorted(loader.dataset.sample_ids.tolist())
    assert loader.epoch == 1 and loader.last_batch_size == 56


def test_concurrent_augmentation_is_deterministic_and_rng_isolated(loaders):
    training, _, _ = loaders
    before = torch.get_rng_state().clone()

    def take(loader):
        batch = next(iter(loader))
        return batch, list(loader.last_sample_ids)

    serial = [take(training[cid]) for cid in (0, 1)]
    # Independent newly constructed loaders start from the same epoch seeds.
    repeated, _, _ = adapter().make_loaders(data_path=SRC / 'datasets')
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(take, repeated[:2]))
    for (first, first_ids), (second, second_ids) in zip(serial, parallel):
        assert first_ids == second_ids
        assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.equal(before, torch.get_rng_state())


def test_first_epoch_matches_na_shuffle_and_augmentation(loaders):
    training, _, _ = loaders
    loader = training[3]
    # Use the existing NA seed method and data builder as independent oracles.
    class SeedContract:
        args = {'seed': 42}
        _seed = SFLV2Trainer._seed

    oracle = SeedContract()
    dataset = loader.dataset
    expected_loader = build_loader(dataset.images, dataset.labels, dataset.sample_ids,
                                   128, True, 0,
                                   torch.Generator().manual_seed(oracle._seed('shuffle', 1, 3, 0)))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(oracle._seed('augmentation', 1, 3, 0, 0))
        images, labels, ids = next(iter(expected_loader))
    actual_images, actual_labels = next(iter(loader))
    assert torch.equal(actual_images, images) and torch.equal(actual_labels, labels)
    assert loader.last_sample_ids == ids.tolist()
    next(iter(loader))
    assert loader.epoch == 2 and loader.last_sample_ids != ids.tolist()


def test_test_loader_repeats_without_augmentation_and_no_rng_consumption(loaders):
    _, loader, _ = loaders
    before = torch.get_rng_state().clone()
    first = next(iter(loader))
    second = next(iter(loader))
    assert len(first) == 2 and all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('batch_size', [0, -1, True])
def test_invalid_batch_size_rejected(batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        adapter().make_loaders(batch_size=batch_size)
