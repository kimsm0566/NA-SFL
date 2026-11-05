import logging
import os
import platform
import psutil
import time
from copy import deepcopy
from threading import Thread

import torch
from torch.multiprocessing import Pipe
from torch.autograd.profiler import profile as torch_profile

from communication.one_to_many_communicator import OneToManyCommunicator
from communication.message import Message
from communication.task_scheduler import TaskScheduler
from util.util import get_net, save_multi_stats_csv, set_random_seed, split_nets
from data_preprocessing.data_reader import DataReader
from node.node import Node
from node.async_server_model import AsyncServerModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
set_random_seed(1994)


class AsyncServer(Node):
    def __init__(self, config):
        super().__init__(config)
        # self.device = 'cuda:{}'.format(torch.cuda.device_count() - 1) if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.socket.bind(("", self.config.SERVER_PORT))
        self.socket.listen()
        self.stats = []
        self.net = None
        self.client_net = None
        self.client_sockets = {}
        self.client_configs = {}

    def run(self):
        logger.info('Running on device: {}'.format(self.device))
        # Connect to all clients.
        logger.info("Waiting for {} incoming connections.".format(self.config.CLIENT_NUM))
        connect_threads = [Thread(target=self.connect) for _ in range(self.config.CLIENT_NUM)]
        for connect_thread in connect_threads:
            connect_thread.start()
        for connect_thread in connect_threads:
            connect_thread.join()

        # Communicator
        client_sockets = []
        time_differences = []
        for client_index in range(self.config.CLIENT_NUM):
            client_sockets.append(self.client_sockets[client_index])
            time_differences.append(-self.client_configs[client_index].TIME_DIFFERENCE)
        comm = OneToManyCommunicator(client_sockets, time_differences)
        sender = Thread(target=comm.run_sender)
        sender.start()
        receiver = Thread(target=comm.run_receiver)
        receiver.start()

        # Initialise Network
        complete_net = get_net(self.config, model_type="complete")
        server_net = get_net(self.config, model_type="server")
        client_net = get_net(self.config, model_type="client")
        split_nets(client_net, server_net, complete_net)
        auxiliary_net = get_net(self.config, model_type=self.config.AUXILIARY) if self.config.AUXILIARY else None
        for client_socket in client_sockets:
            Message(
                title="MSG_INITIAL_MODEL",
                content=[client_net, auxiliary_net]
            ).send_to(client_socket)

        # Build server-side models
        scheduler_connect, model_connect = Pipe()
        data_reader = DataReader(self.config, 'test')
        val_data_loader, test_data_loader = data_reader.get_data_loader()
        server_model = AsyncServerModel(
            self.device,
            deepcopy(server_net),
            deepcopy(client_net),
            deepcopy(auxiliary_net),
            deepcopy(self.criterion),
            model_connect,
            self.config,
            val_data_loader,
            test_data_loader,
            comm
        )

        # Task scheduler
        act_confirm = True if self.config.AUXILIARY and not self.config.ASYNC_AGG else False
        act_check = True if self.config.AUXILIARY and self.config.ASYNC_AGG else False
        task_scheduler = TaskScheduler(
            self.config.CLIENT_NUM,
            scheduler_connect,
            comm,
            self.config.CLIENT_DATA_LENGTH,
            self.config.SORT_ALGO,
            act_confirm,
            act_check,
            self.config.ACT_BUFFER,
            server_model.server_timing,
            self.config.ASYNC_AGG,
            self.config.AGG_ALL_MODELS,
            self.config.ACT_PARALLEL_NUM)
        scheduler_put_thread = Thread(target=task_scheduler.put)
        scheduler_put_thread.start()
        scheduler_get_thread = Thread(target=task_scheduler.get)
        scheduler_get_thread.start()

        # Set cpu number
        if not torch.cuda.is_available() and platform.system() != 'Darwin':
            p = psutil.Process(os.getpid())
            if self.config.SERVER_CPU_NUM != os.cpu_count():
                all_cpus = list(range(os.cpu_count()))
                allocated_cpus = all_cpus[-self.config.SERVER_CPU_NUM:]
                p.cpu_affinity(allocated_cpus)
            logger.info("Available CPU number: {}; Allocated CPU cores: {}".format(os.cpu_count(), p.cpu_affinity()))

        # Train server models in parallel
        server_stats = self.model_training(server_model)

        # stats
        test_accuracy = server_stats["test_acc"]
        client_communication_amount = [
            task_scheduler.general_messages[client_index].get().content for client_index in range(self.config.CLIENT_NUM)]
        client_timings = [
            task_scheduler.general_messages[client_index].get().content for client_index in range(self.config.CLIENT_NUM)]
        self.stats = {
            "experiment_name": self.config.EXPERIMENT_NAME,
            "test_accuracy": test_accuracy,
            "server_communication_amount": comm.amounts,
            "client_communication_amount": client_communication_amount,
            "training_time": {
                "total": server_stats["total"],
                "server": {
                    "total": server_stats["total"],
                    "converge": server_stats["converge"],
                    "best step": server_stats["best step"],
                    "forward": server_stats["forward"],
                    "backward": server_stats["backward"],
                    "update": server_stats["update"],
                    "aggregate": server_stats["aggregate"],
                    "idle": server_stats["idle"],
                    "upload": server_stats["upload"],
                },
                "client": [{
                    "index": client_index,
                    "total": client_timings[client_index].elapsed_time,
                    "data_load": client_timings[client_index].accumulate("load"),
                    "forward": client_timings[client_index].accumulate("forward"),
                    "backward": client_timings[client_index].accumulate("backward"),
                    "update": client_timings[client_index].accumulate("update"),
                    "idle": client_timings[client_index].accumulate("idle"),
                    "download": client_timings[client_index].accumulate("idle"),
                    "straggler": client_timings[client_index].accumulate("straggler"),
                } for client_index in range(self.config.CLIENT_NUM)],
            },
            "train_loss": server_stats["train_loss"],
            "train_acc": server_stats["train_acc"],
            "val_loss": server_stats["val_loss"],
            "val_acc": server_stats["val_acc"],
            "time_line": server_stats["time_line"],
            "split_point": self.config.LAYER_NUM_ON_CLIENT,
            "parallel_batch_num": self.config.PIPE_BATCH_NUM,
        }
        print(self.stats)
        training_time = server_stats["total"]
        logger.info("Total training time: {}".format(training_time))
        logger.info("Global test accuracy: {}".format(test_accuracy))

        # Save results.
        if not os.path.exists(self.config.RESULT_DIR):
            os.mkdir(self.config.RESULT_DIR)
        save_multi_stats_csv(self.stats, self.config.RESULT_ASYNC_CSV_ADDRESS)

        # End
        for client_index, client_socket in self.client_sockets.items():
            comm.send(Message(title="CLOSE_COMMUNICATION_CHANNEL", close_comm=True), client_index)
        sender.join()
        receiver.join()
        scheduler_put_thread.join()
        scheduler_get_thread.join()
        self.disconnect()

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
        self.config.get_vocab()
        self.config.get_model_info()
        client_config = Message(
            from_socket=client_socket, expected_title="MSG_CLIENT_CONFIG").content
        Message(
            title="MSG_CONNECTION_STATE",
            content="Client {} connected to the server.".format(client_config.CLIENT_INDEX)
        ).send_to(client_socket)
        self.client_sockets[client_config.CLIENT_INDEX] = client_socket
        self.client_configs[client_config.CLIENT_INDEX] = client_config

    def model_training(self, server_model):
        if self.config.LAYER_NUM_ON_CLIENT != -1:
            torch.set_num_threads(max(1, self.config.SERVER_CPU_NUM))
        else:
            torch.set_num_threads(1)
        # Train.
        logger.info("Start training...")
        with torch_profile(enabled=server_model.config.ENABLE_PROFILER) as prof:
            server_model.server_timing.start()
            server_model.train()
            server_model.server_timing.stop()
        logger.info("Training done.")
        # Test.
        if self.config.ENABLE_TEST:
            logger.info("Start testing...")
            server_model.test()
            logger.info("Testing done.")
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
        fig_save_address = server_model.config.EXPERIMENT_NAME
        # draw_timeline(
        #     start_iter=0,
        #     end_iter=5,
        #     server_timing=server_model.py.server_timing,
        #     client_timing=server_model.py.client_timing,
        #     time_difference=server_model.py.config.TIME_DIFFERENCE,
        #     save_address=os.path.join(server_model.py.config.FIG_DIR, fig_save_address),
        #     show=False
        # )
        stat = {
            "total": round(server_model.server_timing.elapsed_time, 2),
            "converge": round(server_model.converge_time, 2),
            "best step": server_model.best_val_step,
            "forward": round(server_model.server_timing.accumulate("forward"), 2),
            "backward": round(server_model.server_timing.accumulate("backward"), 2),
            "update": round(server_model.server_timing.accumulate("update"), 2),
            "aggregate": round(server_model.server_timing.accumulate("aggregate"), 2),
            "idle": round(server_model.server_timing.accumulate("idle"), 2),
            "upload": round(server_model.server_timing.accumulate("upload"), 2),
            "train_loss": server_model.train_loss,
            "train_acc": server_model.train_acc,
            "val_loss": server_model.val_loss,
            "val_acc": server_model.val_acc,
            "test_acc": server_model.test_acc,
            "time_line": server_model.time_line
        }
        return stat

    def disconnect(self):
        for client_socket in self.client_sockets.values():
            client_socket.close()

    def _accumulate_communication(self, communication_amount):
        for title, amount in communication_amount.items():
            self.communication_amount[title] = self.communication_amount.get(title, 0) + amount
