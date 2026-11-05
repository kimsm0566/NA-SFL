import logging
import os
import platform
import psutil
import time
from copy import deepcopy
from threading import Thread

import torch
from torch.multiprocessing import Process, Queue, Lock
from torch.autograd.profiler import profile as torch_profile

from util.aggregator import Aggregator
from util.async_aggregator import AsyncAggregator
from communication.one_to_one_communicator import OneToOneCommunicator
from util.profile import recommend_parameter
from communication.message import Message
from util.util import draw_timeline, get_net, print_log, save_stats_csv, set_random_seed, split_nets
from node.node import Node
from node.server_model import ServerModel
from node.simulated_server_model import SimulatedServerModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
set_random_seed(1994)


class Server(Node):
    def __init__(self, config):
        super().__init__(config)
        # self.device = 'cuda:{}'.format(torch.cuda.device_count() - 1) if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.socket.bind(("", self.config.SERVER_PORT))
        self.socket.listen()
        self.stats = []
        self.net = None
        self.client_net = None
        self.complete_net = None

    def run(self):
        logger.info('Running on device: {}'.format(self.device))
        # Set cpu number
        if not torch.cuda.is_available() and platform.system() != 'Darwin':
            p = psutil.Process(os.getpid())
            if self.config.SERVER_CPU_NUM != os.cpu_count():
                all_cpus = list(range(os.cpu_count()))
                allocated_cpus = all_cpus[-self.config.SERVER_CPU_NUM:]
                p.cpu_affinity(allocated_cpus)
            logger.info("Available CPU number: {}; Allocated CPU cores: {}".format(os.cpu_count(), p.cpu_affinity()))

        # Train server models in parallel
        all_processes = []
        sync = Queue(self.config.CLIENT_NUM)
        stats = Queue(self.config.CLIENT_NUM)
        aggregator = AsyncAggregator() if self.config.ASYNC_AGG else Aggregator(
            self.config.CLIENT_NUM, self.config.CLIENT_NUM)
        shared_info = Queue(self.config.CLIENT_NUM - 1)
        test_shared_info = Queue(self.config.CLIENT_NUM - 1)
        lock = Lock()
        logger.info("Waiting for {} incoming connections.".format(self.config.CLIENT_NUM))
        for _ in range(self.config.CLIENT_NUM):
            single_process = Process(target=self.run_a_process,
                                     args=(aggregator, stats, sync, shared_info, test_shared_info, lock))
            single_process.start()
            all_processes.append(single_process)
        for single_process in all_processes:
            single_process.join()
        while not stats.empty():
            stat = stats.get()
            self.stats.append(stat)
        test_accuracy = self.stats[0]["test_accuracy"]
        split_points = []
        parallel_batch_nums = []
        total_comm_amount = 0
        for stat in self.stats:
            split_points.append(stat['split_point'])
            parallel_batch_nums.append(stat['parallel_batch_num'])
            total_comm_amount += stat['server_communication_amount'] + stat['client_communication_amount']
        training_time = self.stats[0]["training_time"]["server"]["total"]
        logger.info("Total training time: {}".format(training_time))
        print_log(self.config, test_accuracy, training_time,
                  total_comm_amount, split_points, parallel_batch_nums)
        return

    def connect(self):
        (client_socket, (ip_address, port)) = self.socket.accept()
        logger.info("New connection added: " + ip_address + ":" + str(port))
        Message(
            title="MSG_SERVER_CONFIG",
            content=self.config,
        ).send_to(client_socket)
        Message(
            title="MSG_SERVER_TIME",
            content=time.time()
        ).send_to(client_socket)
        client_config = Message(
            from_socket=client_socket, expected_title="MSG_CLIENT_CONFIG").content
        self.config.update(
            client_config,
            exceptions=["CLIENT_INDEX",
                        "BATCH_SIZE",
                        "DATA_LENGTH",
                        "LOCAL_DATA_LENGTH",
                        "TEST_DATA_LENGTH",
                        "LOCAL_TEST_DATA_LENGTH",
                        "SERVER_ADDRESS",
                        "DATASET_PATH"]
        )
        self.config.TIME_DIFFERENCE *= -1
        self.config.get_vocab()
        self.config.get_model_info()
        Message(
            title="MSG_CONNECTION_STATE",
            content="Client {} connected to the server.".format(client_config.CLIENT_INDEX)
        ).send_to(client_socket)
        if self.config.SIMULATED_CLIENT_NUM > 1:
            server_model = SimulatedServerModel(
                client_config.CLIENT_INDEX,
                ip_address,
                port,
                self.socket,
                client_socket,
                self.device,
                deepcopy(self.net),
                deepcopy(self.client_net),
                deepcopy(self.complete_net),
                deepcopy(self.criterion),
                self.config,
            )
        else:
            server_model = ServerModel(
                client_config.CLIENT_INDEX,
                ip_address,
                port,
                self.socket,
                client_socket,
                self.device,
                deepcopy(self.net),
                deepcopy(self.client_net),
                deepcopy(self.complete_net),
                deepcopy(self.criterion),
                self.config,
            )
        return server_model

    def _distribute_client_model(self, client_sock):
        Message(
            title="MSG_INITIAL_MODEL",
            content=self.client_net
        ).send_to(client_sock)

    def _accumulate_communication(self, communication_amount):
        for title, amount in communication_amount.items():
            self.communication_amount[title] = (self.communication_amount.get(title, 0) + amount)

    def run_a_process(self, aggregator, stats, sync, shared_info, test_shared_info, lock):
        server_model = self.connect()
        run_profiler = False
        if server_model.config.PROFILE_ITER_NUM > 0:
            if server_model.config.UPLINK_BANDWIDTH > 0 and server_model.config.DOWNLINK_BANDWIDTH > 0:
                run_profiler = True
            else:
                raise Exception("Uplink and downlink bandwidth need to be given when running profiler.")
        if self.config.LAYER_NUM_ON_CLIENT != -1 or run_profiler:
            torch.set_num_threads(max(1, self.config.SERVER_CPU_NUM // self.config.CLIENT_NUM))
        else:
            torch.set_num_threads(1)
        # profile
        profiling_time = 0
        if run_profiler:
            logging.info("Running profiler...")
            profiler_start_time = time.time()
            server_model.profile()
            server_model.receive_client_profile_res()
            if server_model.config.PROFILE_NUM_ON_CLIENT == -1:
                layer_num = server_model.config.LAYER_NUM
            else:
                layer_num = server_model.config.PROFILE_NUM_ON_CLIENT + 1
            split_point, batch_num = recommend_parameter(
                server_model.client_profile_res, server_model.server_profile_res,
                server_model.config.UPLINK_BANDWIDTH, server_model.config.DOWNLINK_BANDWIDTH,
                layer_num, server_model.config.BATCH_SIZE)
            server_model.config.LAYER_NUM_ON_CLIENT = split_point
            server_model.config.PIPE_BATCH_NUM = batch_num
            server_model.send_recommend_parameters()
            profiling_time = time.time() - profiler_start_time
            logging.info("Profile time: {}".format(profiling_time))
            logging.info("Split point: {}; Pipe batch num: {}".format(split_point, batch_num))
        complete_net = get_net(server_model.config, model_type="complete")
        server_net = get_net(server_model.config, model_type="server")
        client_net = get_net(server_model.config, model_type="client")
        split_nets(client_net, server_net, complete_net)
        server_model.update_net(server_net, client_net, complete_net)
        if server_model.config.ASYNC_AGG and server_model.client_index == 0:
            aggregator.init_global_model(complete_net.state_dict())
        Message(
            title="MSG_INITIAL_MODEL",
            content=client_net
        ).send_to(server_model.client_socket)
        # Sync the start time for all clients.
        sync.put(server_model.client_index)
        while not (sync.full() and sync.full() and sync.full() and sync.full()):
            time.sleep(0.01)
        server_model.send_sync_flag()
        # Train.
        logger.info("Client {} start training...".format(server_model.client_index))
        with torch_profile(enabled=server_model.config.ENABLE_PROFILER) as prof:
            comm = None
            comm_thread = None
            comm_thread2 = None
            if server_model.config.ASYNC_COMM:
                comm = OneToOneCommunicator(server_model.client_socket,
                                            config=server_model.config,
                                            time_difference=server_model.config.TIME_DIFFERENCE)
                comm_thread = Thread(target=comm.run_receiver)
                comm_thread.start()
                comm_thread2 = Thread(target=comm.run_sender)
                comm_thread2.start()
                server_model.set_comm(comm)
            if server_model.client_index == 0:
                server_model.set_data_loader()
            server_model.train(aggregator, shared_info)
        logger.info("Client {} training done.".format(server_model.client_index))
        if server_model.config.ENABLE_PROFILER:
            print(prof.key_averages().table(sort_by="cpu_time_total"))
            if not os.path.exists(server_model.config.PROFILE_DIR):
                os.mkdir(server_model.config.PROFILE_DIR)
            profile_file = os.path.join(
                server_model.config.PROFILE_DIR,
                '{}_server_{}_trace.json'.format(server_model.config.EXPERIMENT_NAME, server_model.client_index))
            prof.export_chrome_trace(profile_file)
        # Test.
        if self.config.ENABLE_TEST:
            logger.info("Client {} start testing...".format(server_model.client_index))
            server_model.test(test_shared_info)
            logger.info("Client {} testing done.".format(server_model.client_index))
        # Statistics.
        server_model.receive_client_communication_amount()
        server_model.receive_client_timing()
        # End
        if server_model.config.ASYNC_COMM:
            comm.receive()
            comm_thread.join()
            comm.send(Message(title="CLOSE_COMMUNICATION_CHANNEL", close_comm=True))
            comm_thread2.join()
            server_model.server_communication_amount["TRAINING"] = comm.amount
        # Save checkpoint.
        if self.config.MODEL_SAVE:
            if not os.path.exists(self.config.MODEL_DIR):
                os.mkdir(self.config.MODEL_DIR)
            if self.config.LAYER_NUM_ON_CLIENT != -1:
                torch.save(server_model.global_server_model,
                           os.path.join(self.config.MODEL_DIR, self.config.MODEL_NAME + '_on_server.pth'))
        if not os.path.exists(server_model.config.RESULT_DIR):
            os.mkdir(server_model.config.RESULT_DIR)
        if not os.path.exists(server_model.config.FIG_DIR):
            os.mkdir(server_model.config.FIG_DIR)
        # fig_save_address = server_replica.config.EXPERIMENT_NAME + "_client_{}".format(server_replica.client_index)
        # draw_timeline(
        #     start_iter=0,
        #     end_iter=5,
        #     server_timing=server_replica.server_timing,
        #     client_timing=server_replica.client_timing,
        #     time_difference=server_replica.config.TIME_DIFFERENCE,
        #     save_address=os.path.join(server_replica.config.FIG_DIR, fig_save_address),
        #     show=False
        # )
        if not os.path.exists(self.config.RESULT_DIR):
            os.mkdir(self.config.RESULT_DIR)
        if server_model.config.SIMULATED_CLIENT_NUM > 1:
            multi_stat = []
            for simulate_index in range(server_model.config.SIMULATED_CLIENT_NUM):
                stat = {
                    "experiment_name": server_model.config.EXPERIMENT_NAME,
                    "client_index": server_model.client_index,
                    "simulate_index": simulate_index,
                    "training_time": {
                        "server": {
                            "total": round(server_model.server_timings[simulate_index].elapsed_time, 2),
                            "converge": None,
                            "best step": None,
                            "forward": round(server_model.server_timings[simulate_index].accumulate("forward"), 2),
                            "backward": round(server_model.server_timings[simulate_index].accumulate("backward"), 2),
                            "update": round(server_model.server_timings[simulate_index].accumulate("update"), 2),
                            "aggregate": round(server_model.server_timings[simulate_index].accumulate("aggregate"), 2),
                            "idle": round(server_model.server_timings[simulate_index].accumulate("idle"), 2),
                        },
                        "client": {
                            "total": round(server_model.client_timings[simulate_index].elapsed_time, 2),
                            "data_load": round(server_model.client_timings[simulate_index].accumulate("load"), 2),
                            "forward": round(server_model.client_timings[simulate_index].accumulate("forward"), 2),
                            "backward": round(server_model.client_timings[simulate_index].accumulate("backward"), 2),
                            "update": round(server_model.client_timings[simulate_index].accumulate("update"), 2),
                            "idle": round(server_model.client_timings[simulate_index].accumulate("idle"), 2),
                            "straggler": round(server_model.client_timings[simulate_index].accumulate("straggler"), 2)
                        },
                        "upload": round(server_model.server_timings[simulate_index].accumulate("upload"), 2),
                        "download": round(server_model.client_timings[simulate_index].accumulate("download"), 2),
                    },
                    "test_accuracy": server_model.test_acc,
                    "server_communication_amount": sum(server_model.server_communication_amount.values()),
                    "client_communication_amount": server_model.client_communication_amount,
                    "train_loss": server_model.train_loss,
                    "train_acc": server_model.train_acc,
                    "val_loss": server_model.val_loss,
                    "val_acc": server_model.val_acc,
                    "time_line": server_model.time_line,
                    "split_point": server_model.config.LAYER_NUM_ON_CLIENT,
                    "parallel_batch_num": server_model.config.PIPE_BATCH_NUM,
                    "profiling_time": profiling_time
                }
                lock.acquire()
                save_stats_csv(stat, server_model.config.RESULT_SYNC_CSV_ADDRESS)
                lock.release()
                # print(stat)
                multi_stat.append(stat)
            stat = multi_stat[0]
        else:
            stat = {
                "experiment_name": server_model.config.EXPERIMENT_NAME,
                "client_index": server_model.client_index,
                "training_time": {
                    "server": {
                        "total": round(server_model.server_timing.elapsed_time, 2),
                        "converge": round(server_model.converge_time, 2)
                            if server_model.client_index == 0 and server_model.converge_time is not None else None,
                        "best step": server_model.best_val_step if server_model.client_index == 0 else None,
                        "forward": round(server_model.server_timing.accumulate("forward"), 2),
                        "backward": round(server_model.server_timing.accumulate("backward"), 2),
                        "update": round(server_model.server_timing.accumulate("update"), 2),
                        "aggregate": round(server_model.server_timing.accumulate("aggregate"), 2),
                        "idle": round(server_model.server_timing.accumulate("idle"), 2),
                    },
                    "client": {
                        "total": round(server_model.client_timing.elapsed_time, 2),
                        "data_load": round(server_model.client_timing.accumulate("load"), 2),
                        "forward": round(server_model.client_timing.accumulate("forward"), 2),
                        "backward": round(server_model.client_timing.accumulate("backward"), 2),
                        "update": round(server_model.client_timing.accumulate("update"), 2),
                        "idle": round(server_model.client_timing.accumulate("idle"), 2),
                        "straggler": round(server_model.client_timing.accumulate("straggler"), 2)
                    },
                    "upload": round(server_model.server_timing.accumulate("upload"), 2),
                    "download": round(server_model.client_timing.accumulate("download"), 2),
                },
                "test_accuracy": server_model.test_acc,
                "server_communication_amount": sum(server_model.server_communication_amount.values()),
                "client_communication_amount": server_model.client_communication_amount,
                "train_loss": server_model.train_loss,
                "train_acc": server_model.train_acc,
                "val_loss": server_model.val_loss,
                "val_acc": server_model.val_acc,
                "time_line": server_model.time_line,
                "split_point": server_model.config.LAYER_NUM_ON_CLIENT,
                "parallel_batch_num": server_model.config.PIPE_BATCH_NUM,
                "profiling_time": profiling_time
            }
            lock.acquire()
            save_stats_csv(stat, server_model.config.RESULT_SYNC_CSV_ADDRESS)
            lock.release()
        del stat["train_loss"]
        del stat["train_acc"]
        del stat["val_loss"]
        del stat["val_acc"]
        stats.put(stat)
        # Ending.
        while not (stats.full() and stats.full() and stats.full() and stats.full()):
            time.sleep(0.1)
        server_model.send_sync_flag()
        server_model.server_socket.close()
        return
