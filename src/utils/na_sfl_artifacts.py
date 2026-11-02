"""NA convergence reports and project-wide final-checkpoint retention."""
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import fcntl
from pathlib import Path
import json

from utils.deit_artifacts import json_file
from utils.sflv2_metrics import sha256


@contextmanager
def _registry_lock(registry_path):
    path = Path(registry_path).with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def target_summary(rows, accuracy, *, initial_evaluation=None):
    if initial_evaluation is not None:
        rows = [{'round': 0, **initial_evaluation}, *rows]
    result = {'accuracy': accuracy, 'status': 'not_configured' if accuracy is None else 'not_reached', 'round': None}
    timed = accuracy is not None and any('elapsed_wall_s' in row for row in rows)
    if timed:
        result['elapsed_wall_s'] = None
    if accuracy is not None:
        for row in rows:
            if row['test_accuracy'] >= accuracy:
                result.update(status='reached', round=row['round'])
                if timed:
                    result['elapsed_wall_s'] = row['elapsed_wall_s']
                break
    return result


def retain_final_checkpoint(registry_path, entry, allowed_root):
    """Only user-authorized model retention; all non-model artifacts survive."""
    with _registry_lock(registry_path):
        registry_path, allowed_root = Path(registry_path), Path(allowed_root).resolve()
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
        previous = registry.get('final_checkpoints', [])
        for item in [*previous, entry]:
            path = Path(item['path']).resolve()
            if not path.is_relative_to(allowed_root) or path.suffix not in ('.pt', '.pth', '.ckpt', '.safetensors'):
                raise ValueError('checkpoint retention path is outside the allowed results tree')
        new_path = Path(entry['path'])
        if not new_path.is_file() or new_path.stat().st_size != entry['bytes'] or sha256(new_path) != entry['sha256']:
            raise ValueError('new final checkpoint identity does not match registration')
        same = [item for item in previous if item['experiment_id'] == entry['experiment_id']
                and item['path'] != entry['path']]
        candidates = [item for item in previous if item['experiment_id'] != entry['experiment_id']]
        candidates.append(entry)
        candidates.sort(key=lambda item: (datetime.fromisoformat(item['final_saved_at']), item['experiment_id']), reverse=True)
        keep, remove = candidates[:5], same + candidates[5:]
        now = datetime.now(timezone(timedelta(hours=9))).isoformat()
        deletions = []
        for item in remove:
            path = Path(item['path'])
            if path.exists():
                path.unlink()
            deletions.append({**item, 'deleted_at': now,
                'reason': 'same experiment replacement or project latest-five final-model limit'})
        registry.update(final_checkpoints=keep, final_count=len(keep), global_limit=5, updated_at=now)
        registry['retention_deletions'] = registry.get('retention_deletions', []) + deletions
        json_file(registry_path, registry)
        return registry


def discard_failed_checkpoint(run_dir, registry_path, allowed_root):
    """A failed launch is never a retained final experiment model."""
    with _registry_lock(registry_path):
        path = (Path(run_dir) / 'checkpoints' / 'final.pt').resolve()
        if not path.is_relative_to(Path(allowed_root).resolve()):
            raise ValueError('failed checkpoint lies outside allowed results')
        registry_path = Path(registry_path)
        if registry_path.exists():
            registry = json.loads(registry_path.read_text())
            entries = registry.get('final_checkpoints', [])
            removed = [item for item in entries if Path(item['path']).resolve() == path]
            registry['final_checkpoints'] = [item for item in entries if Path(item['path']).resolve() != path]
            registry['final_count'] = len(registry['final_checkpoints'])
            registry['retention_deletions'] = registry.get('retention_deletions', []) + [
                {**item, 'reason': 'launch failed after final save; incomplete models are not retained'} for item in removed]
            json_file(registry_path, registry)
        path.unlink(missing_ok=True)


def sweep_status(records):
    complete = bool(records) and all(item['completed'] for item in records)
    return {'status': 'completed' if complete else 'stopped_at_round_boundary',
            'completed': complete, 'completed_dispatch': True}


def register_checkpoint(run_dir, config, checkpoint, experiment_id, *, final_saved_at):
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(checkpoint)
    entry = {'experiment_id': experiment_id, 'run_dir': str(run_dir), 'path': str(checkpoint),
        'bytes': checkpoint.stat().st_size, 'sha256': sha256(checkpoint),
        'final_saved_at': final_saved_at,
        'scope': 'NA live_async round-boundary final; Client/Server/Aux and optimizers bundled',
        'method': config['method'], 'seed': config['seed'], 'aux_client_count': config['aux_client_count']}
    registry = retain_final_checkpoint(root / 'results' / 'checkpoint_registry.json', entry, root / 'results')
    json_file(Path(run_dir) / 'checkpoint_retention.json', {
        'registered': entry, 'retained_final_count': registry['final_count'],
        'policy': 'one final per experiment; project latest five',
        'retention_deletions': registry.get('retention_deletions', [])})
    return entry


def render_run(root, rows, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    x = [row['round'] for row in rows]
    axes[0].plot(x, [row['test_accuracy'] for row in rows], marker='.')
    axes[0].set(xlabel='Communication round', ylabel='Global test accuracy (%)')
    axes[1].plot(x, [row['test_ce_loss'] for row in rows], label='Test CE')
    prototype_method = config['method'] in ('na_sfl_v3_proto', 'na_sfl_v4_residual')
    axes[1].plot(x, [row['train_server_ce_loss'] for row in rows],
                 label='Online combined server loss' if prototype_method else 'Online server CE')
    axes[1].set(xlabel='Communication round', ylabel='Loss' if prototype_method else 'Cross-entropy loss')
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle(f"{config['method']} / {config['server_policy']} / auxiliary-gradient clients {config['aux_client_count']}/10 / seed{config['seed']}")
    fig.savefig(root / 'learning_curves.png', dpi=180)
    plt.close(fig)
    partition = json.loads((root / 'partition.json').read_text())
    histogram = [[labels.count(cls) for cls in range(10)] for labels in partition['labels_by_client']]
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    heatmap = axis.imshow(histogram, aspect='auto', cmap='Blues')
    axis.set(xlabel='CIFAR10 class', ylabel='Client ID', xticks=range(10),
             yticks=range(len(histogram)), title='Fixed per-client class distribution')
    fig.colorbar(heatmap, ax=axis, label='Training samples')
    fig.savefig(root / 'class_distribution.png', dpi=180)
    plt.close(fig)


def render_suite(root, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for item in records:
        rows = json.loads((Path(item['run_dir']) / 'round_metrics.json').read_text())
        label = f"k={item['aux_client_count']}"
        axes[0].plot([r['round'] for r in rows], [r['test_accuracy'] for r in rows], label=label)
        axes[1].plot([r['round'] for r in rows], [r['test_ce_loss'] for r in rows], label=label)
    axes[0].set(ylabel='Global test accuracy (%)')
    axes[1].set(ylabel='Global test CE loss')
    for axis in axes:
        axis.set_xlabel('Communication round')
        axis.grid(alpha=.2)
        axis.legend(ncol=3, fontsize=8)
    fig.suptitle('NA-SFLv1 / one seed / actual live FCFS orders may differ')
    fig.savefig(root / 'count_learning_curves.png', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot([r['aux_client_count'] for r in records], [r['final_test_accuracy'] for r in records], marker='o')
    axis.set(xlabel='Auxiliary-gradient clients (all10 heads trained)', ylabel='Final global test accuracy (%)')
    axis.grid(alpha=.2)
    fig.savefig(root / 'count_final_accuracy.png', dpi=180)
    plt.close(fig)


def render_comparison(root, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for item in records:
        rows = json.loads((Path(item['run_dir']) / 'round_metrics.json').read_text())
        for axis, x in zip(axes, ('round', 'elapsed_wall_s')):
            axis.plot([r[x] for r in rows], [r['test_accuracy'] for r in rows],
                      label=item['method'] + '/' + item['server_policy'])
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
            axis.set_ylabel('Global test accuracy (%)')
    axes[0].set_xlabel('Communication round')
    axes[1].set_xlabel('Elapsed wall seconds (initial broadcast through evaluation)')
    fig.savefig(Path(root) / 'comparison_learning_curves.png', dpi=180)
    plt.close(fig)
