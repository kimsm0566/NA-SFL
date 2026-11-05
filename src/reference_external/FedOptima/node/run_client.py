import argparse
import gc
import logging
import os
import platform
import psutil
import random
import time
from time import sleep
from threading import Thread

import torch
from torch.autograd.profiler import profile as torch_profile

import config
from communication.one_to_one_communicator import OneToOneCommunicator
from communication.message import Message
from node.async_client import AsyncClient
from node.client import Client
from node.simulated_client import SimulatedClient

from util.util import set_random_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run(client_config=None):
    set_random_seed(client_config.CLIENT_INDEX+1994)
    if client_config is None:
        parser = argparse.ArgumentParser(description='Client-side arguments.')
        parser.add_argument('--client_index', '-i', type=int, default=0, help='the index of client, required')
        parser.add_argument('--port', '-p', type=int, default=51000)
        args = parser.parse_args()
        client_config = config.Config(
            client_index=args.client_index,
            port=args.port,
        )
    local_test_accuracy = run_a_client(client_config)
    logger.info('Local Test Accuracy: {}'.format(local_test_accuracy))


def run_a_client(client_config):
    if client_config.ASYNCHRONISATION:
        client = AsyncClient(client_config)
    else:
        if client_config.SIMULATED_CLIENT_NUM > 1:
            client = SimulatedClient(client_config)
        else:
            client = Client(client_config)
    logger.info('Running on device: {}'.format(client.device))
    client.connect()
    # set bandwidth
    bandwidth_reset = False
    if not client_config.SIMULATE_BANDWIDTH and client_config.UPLINK_BANDWIDTH and client_config.DOWNLINK_BANDWIDTH:
        if client_config.UPLINK_BANDWIDTH > 0 and client_config.DOWNLINK_BANDWIDTH > 0:
            os.system('sudo wondershaper -c -a eth0')
            os.system('sudo wondershaper -a eth0 -u {} -d {}'.format(
                client_config.UPLINK_BANDWIDTH * 1024, client_config.DOWNLINK_BANDWIDTH * 1024))
            bandwidth_reset = True
    if client.config.DYNAMIC_BANDWIDTH:
        dynamic_bandwidth_thread = Thread(target=update_bandwidth, args=(client_config,), daemon=True)
        dynamic_bandwidth_thread.start()
    else:
        set_random_seed(1994)
    # set CPU number
    if not torch.cuda.is_available() and platform.system() != 'Darwin' and not platform.node().startswith('nvnano'):
        p = psutil.Process(os.getpid())
        if client_config.CLIENT_CPU_NUM != os.cpu_count():
            all_cpus = list(range(os.cpu_count()))
            if client_config.IS_SIMULATED:
                start_index = sum(client_config.CLIENT_CPU_NUM_LIST[:client_config.CLIENT_INDEX])
                end_index = start_index + client_config.CLIENT_CPU_NUM
                allocated_cpus = all_cpus[start_index: end_index]
            else:
                allocated_cpus = all_cpus[-client_config.CLIENT_CPU_NUM:]
            p.cpu_affinity(allocated_cpus)
        logger.info("Available CPU number: {}; Allocated CPU cores: {}".format(os.cpu_count(), p.cpu_affinity()))
    # profile
    if (client.config.PROFILE_ITER_NUM > 0 and client.config.UPLINK_BANDWIDTH > 0
            and client.config.DOWNLINK_BANDWIDTH > 0):
        client.profile()
        client.send_client_profile_res()
        client.receive_recommend_parameters()
    client.receive_net()
    if not client_config.ASYNCHRONISATION:
        client.receive_sync_flag()
    # Train.
    with torch_profile(enabled=client.config.ENABLE_PROFILER) as prof:
        # Async communication setup.
        comm = None
        sender_thread = None
        receiver_thread = None
        if client.config.ASYNC_COMM:
            comm = OneToOneCommunicator(client.socket,
                                        time_difference=client.config.TIME_DIFFERENCE,
                                        config=client_config,
                                        timing=client.client_timing)
            sender_thread = Thread(target=comm.run_sender)
            sender_thread.start()
            receiver_thread = Thread(target=comm.run_receiver)
            receiver_thread.start()
            client.set_comm(comm)
        client.train()
    if client.config.ENABLE_PROFILER:
        print(prof.key_averages().table(sort_by="cpu_time_total"))
        if not os.path.exists(client.config.PROFILE_DIR):
            os.mkdir(client.config.PROFILE_DIR)
        profile_file = os.path.join(
            client.config.PROFILE_DIR,
            '{}_client_{}_trace.json'.format(client.config.EXPERIMENT_NAME, client.config.CLIENT_INDEX))
        prof.export_chrome_trace(profile_file)
    # Test.
    if client.config.ENABLE_TEST:
        logger.info("Start testing...")
        client.test()
        logger.info("Testing done.")
    # Statistics.
    client.send_client_communication_amount()
    client.send_client_timing()
    # End
    if client.config.ASYNC_COMM:
        comm.send(Message(title="CLOSE_COMMUNICATION_CHANNEL", close_comm=True))
        sender_thread.join()
        comm.receive()
        receiver_thread.join()
        client.communication_amount['TRAINING'] = comm.amount
        print("Logit communication: {} MB".format(comm.logit_amount / 1024 / 1024 * 2))
    # Save checkpoint.
    if client.config.MODEL_SAVE:
        if not os.path.exists(client.config.MODEL_DIR):
            os.mkdir(client.config.MODEL_DIR)
        torch.save(client.global_client_model,
                   os.path.join(client.config.MODEL_DIR,
                                client.config.MODEL_NAME + '_on_client{}.pth'.format(client.config.CLIENT_INDEX)))
    if not client_config.ASYNCHRONISATION:
        client.receive_sync_flag()
    client.socket.close()
    if bandwidth_reset:
        os.system('sudo wondershaper -c -a eth0')
    test_accuracy = client.test_accuracy
    del client
    gc.collect()
    return test_accuracy


def update_bandwidth(client_config):
    while True:
        sleep(client_config.BANDWIDTH_CHANGE_INTERVAL)
        drop = random.uniform(0, 1) < client_config.DEVICE_DROP_RATE
        if drop:
            client_config.UPLINK_BANDWIDTH = 0
            client_config.DOWNLINK_BANDWIDTH = 0
        else:
            choice_ind = random.randint(0, len(client_config.DYNAMIC_BANDWIDTH) - 1)
            client_config.UPLINK_BANDWIDTH = client_config.DYNAMIC_BANDWIDTH[choice_ind]
            client_config.DOWNLINK_BANDWIDTH = client_config.DYNAMIC_BANDWIDTH[choice_ind]


if __name__ == '__main__':
    run()
