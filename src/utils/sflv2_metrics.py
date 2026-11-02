"""Source snapshots and final manifests used by NA-SFL."""
import hashlib
import json
import os
from pathlib import Path
import shutil

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PRUNED = {'results', 'datasets', '__pycache__', '.git', '.pytest_cache'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def snapshot_sources(run_dir):
    """Prune large data/result trees before traversal, retaining active source/config only."""
    hashes = {}
    target = Path(run_dir) / 'source_snapshot'
    for folder, dirs, files in os.walk(SOURCE_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in PRUNED and not d.startswith('.'))
        for name in sorted(files):
            path = Path(folder) / name
            if path.suffix not in ('.py', '.yaml', '.yml', '.sh', '.c', '.h', '.patch'):
                continue
            relative = path.relative_to(SOURCE_ROOT).as_posix()
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            hashes[relative] = sha256(destination)
    return hashes


def check_sources_unchanged(hashes):
    for name, digest in hashes.items():
        if not (SOURCE_ROOT / name).is_file() or sha256(SOURCE_ROOT / name) != digest:
            raise RuntimeError(f'source changed during execution: {name}')


def seal_manifest(run_dir, source_hashes, *, kind='training'):
    root = Path(run_dir)
    check_sources_unchanged(source_hashes)
    artifacts = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
                 if p.is_file() and p.name != 'manifest.json' and 'source_snapshot' not in p.parts}
    write_json(root / 'manifest.json', {'schema_version': 1, 'kind': kind,
               'source_sha256': source_hashes, 'artifacts': artifacts})
