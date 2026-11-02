"""NA-SFL command-line options for the single experiment entrypoint."""
import argparse
import math
from pathlib import Path
import sys

from utils.na_sfl_config import load_config


def build_parser():
    parser = argparse.ArgumentParser(description='GPU NA-SFL: CIFAR10 ResNet18 with live_async workers')
    subparsers = parser.add_subparsers(dest='trainer', required=True)
    na = subparsers.add_parser('na-sfl-v1', help='selective auxiliary gradients; all clients train auxiliary heads')
    na.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs' / 'na_sfl_v1.yaml'))
    na.add_argument('--method', choices=('splitfedv2', 'na_sfl_v1', 'na_sfl_v2', 'hybrid_no_aux_avg',
                    'na_sfl_v3_proto', 'na_sfl_v4_residual'), default=None)
    na.add_argument('--server-policy', choices=('individual', 'concat'), default=None)
    for flag in ('aux-client-count', 'rounds', 'seed', 'device', 'max-batches-per-client', 'server-max-batch-samples'):
        na.add_argument('--' + flag, type=int, default=None)
    na.add_argument('--result-path', default=None)
    na.add_argument('--target-accuracy', type=float, default=None)
    na.add_argument('--resume', default=None)
    na.add_argument('--allow-unprofiled-gpu', action=argparse.BooleanOptionalAction, default=None,
                    help='explicitly allow an NVIDIA GPU other than the two measured research models; busy/free-memory checks remain')
    na.add_argument('--smoke-test', action='store_true', help='at most2 rounds; verify/discard own checkpoint and preserve retained models')
    na.add_argument('--alignment-weight-beta', type=float, default=None)
    na.add_argument('--pseudo-loss-weight-gamma', type=float, default=None)
    na.add_argument('--pseudo-gradient-to-client', action=argparse.BooleanOptionalAction, default=None)
    na.add_argument('--residual-recenter', action=argparse.BooleanOptionalAction, default=None)
    na.add_argument('--stop-after-rounds', type=int, default=None)
    na.add_argument('--max-wall-time-s', type=float, default=None, help='stop only at a completed round boundary')
    dispatch = na.add_mutually_exclusive_group()
    dispatch.add_argument('--sweep', action='store_true')
    dispatch.add_argument('--comparison', action='store_true', help='four Server/Aux × individual/concat conditions')
    na.add_argument('--client-counts', default='0,1,2,3,4,5,6,7,8,9,10')
    return parser


def parse_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0].startswith('-'):
        arguments.insert(0, 'na-sfl-v1')
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        args.config = load_config(args.config, {key: getattr(args, key) for key in
            ('method', 'server_policy', 'server_max_batch_samples', 'aux_client_count', 'rounds', 'seed', 'device', 'allow_unprofiled_gpu',
             'max_batches_per_client', 'result_path', 'target_accuracy')
            if getattr(args, key) is not None})
        options = {}
        if args.alignment_weight_beta is not None:
            options['prototype'] = {**args.config['prototype'], 'alignment_weight_beta': args.alignment_weight_beta}
        changes = {key: getattr(args, key) for key in ('pseudo_loss_weight_gamma', 'pseudo_gradient_to_client', 'residual_recenter')
                   if getattr(args, key) is not None}
        if changes:
            options['v4'] = {**args.config['v4'], **changes}
        if args.smoke_test:
            options.update(smoke_test=True, rounds=min(args.config['rounds'], 2))
        if options:
            from utils.na_sfl_config import DEFAULTS
            args.config = load_config(overrides={**{k: args.config[k] for k in DEFAULTS}, **options})
        args.client_counts = [int(value) for value in args.client_counts.split(',')]
        if (not args.client_counts or len(set(args.client_counts)) != len(args.client_counts)
                or any(not 0 <= count <= args.config['n_clients'] for count in args.client_counts)):
            raise ValueError('client counts must be unique integers in [0,n_clients]')
        if args.stop_after_rounds is not None and args.stop_after_rounds < 1:
            raise ValueError('stop-after-rounds must be positive')
        if args.max_wall_time_s is not None and (not math.isfinite(args.max_wall_time_s) or args.max_wall_time_s <= 0):
            raise ValueError('max-wall-time-s must be positive and finite')
        if args.resume is not None and (args.sweep or args.comparison or not Path(args.resume).is_file()):
            raise ValueError('resume requires an existing checkpoint and a single count')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return args
