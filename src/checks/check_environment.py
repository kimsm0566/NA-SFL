"""Read-only installation/data checks; never allocate CUDA tensors or start Mininet."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', action='store_true', help='check Mininet imports/tools/root, without creating links')
    parser.add_argument('--data-path', type=Path, help='verify existing local CIFAR10; never download')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    checks = []
    def record(name, passed, observed, expected):
        checks.append({'name': name, 'passed': bool(passed), 'observed': observed, 'expected': expected})
    record('Python', platform.python_version() == '3.10.12', platform.python_version(), '3.10.12')
    record('Architecture', platform.machine() == 'x86_64', platform.machine(), 'x86_64')
    for line in (root / 'requirements.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '-')):
            continue
        name, expected = line.split('==', 1)
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed = None
        record('package:' + name, observed == expected, observed, expected)
    try:
        import torch
        import torchvision
        record('torch CUDA build', torch.version.cuda == '12.1', torch.version.cuda, '12.1')
        record('torchvision import', True, torchvision.__version__, 'imports successfully')
        record('CUDA remains uninitialized', not torch.cuda.is_initialized(), torch.cuda.is_initialized(), False)
    except Exception as error:
        record('torch/torchvision import', False, str(error), 'imports successfully')
    if args.network:
        try:
            from mininet.net import Mininet
            from mininet.nodelib import LinuxBridge
            from mininet.link import TCLink
            observed = metadata.version('mininet')
            record('Mininet', observed == '2.3.1b4', observed, '2.3.1b4; pinned source commit in Dockerfile')
        except Exception as error:
            record('Mininet import', False, str(error), 'imports successfully')
        for tool in ('ip', 'tc', 'brctl', 'ping', 'mn', 'mnexec', 'ethtool', 'mpirun', 'unshare', 'mount'):
            value = shutil.which(tool)
            record('tool:' + tool, value is not None, value, 'available')
        record('network process root', os.geteuid() == 0, os.geteuid(), 0)
    if args.data_path is not None:
        try:
            from torchvision.datasets import CIFAR10
            train = CIFAR10(str(args.data_path / 'cifar10'), train=True, download=False)
            test = CIFAR10(str(args.data_path / 'cifar10'), train=False, download=False)
            record('local CIFAR10', len(train) == 50000 and len(test) == 10000,
                   {'train': len(train), 'test': len(test)}, {'train': 50000, 'test': 10000})
        except Exception as error:
            record('local CIFAR10', False, str(error), 'canonical local data; download=False')
    result = {'passed': all(c['passed'] for c in checks), 'checks': checks,
              'scope': 'read-only dependency/import/data checks; no GPU model, Mininet topology or training'}
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
