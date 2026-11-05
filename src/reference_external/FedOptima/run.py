import argparse
import copy
import logging
import os
import sys
import time
import json

import torch.cuda
import torch.multiprocessing as mp

os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count())

import config
from node import run_client, run_server


parser = argparse.ArgumentParser(description='Arguments for running multiple experiments.')
parser.add_argument('--on_server', '-s',
                    help='split mini-batch across replicas in the same client, default False', action='store_true')
parser.add_argument('--client_index', '-i', type=int, default=-1,
                    help='index of the client, default -1')
parser.add_argument('--hsfl-experiment', action='store_true',
                    help='opt-in matched HSFL model/data/Mininet experiment')
parser.add_argument('--hsfl-smoke', action='store_true', help='short actual matched training before a fresh main run')
parser.add_argument('--hsfl-result-path', default=None)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    if args.hsfl_experiment:
        from hsfl.runtime import run as run_matched
        return run_matched(result_path=args.hsfl_result_path, smoke=args.hsfl_smoke)
    with open("config.json", "r") as f:
        config_s = json.load(f)
    this_config = config.Config(**config_s)
    if sys.platform == "linux":
        mp.set_start_method("spawn")
        mp.set_sharing_strategy('file_system')
    if args.on_server:
        logger.info("Running on server.")
        run_func = run_server.run
    else:
        logger.info("Running on client.")
        run_func = run_client.run
    this_config.CLIENT_INDEX = args.client_index
    run_func(this_config)


if __name__ == "__main__":
    main()
