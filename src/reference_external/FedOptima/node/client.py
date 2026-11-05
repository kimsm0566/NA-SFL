import logging
import time

import tqdm
import torch
import statistics

from node.node import Node
from communication.message import Message
from util.timing import Timing
from util.util import get_optimiser, split_batch
from util.profile import Profiler
from data_preprocessing.data_reader import DataReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Client(Node):
    def __init__(self, client_config):
        super().__init__(client_config)
        # self.device = 'cuda:{}'.format(torch.cuda.device_count() - 1) if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.port = None
        self.test_accuracy = 0
        self.global_client_model = None
        self.client_timing = Timing('client')
        self.profile_res = None
        self.comm = None
        self.data_loader = None

    def connect(self):
        logger.info('Connecting to Server.')
        self.socket.connect((self.config.SERVER_ADDRESS, self.config.SERVER_PORT))
        self.port = self.socket.getsockname()[1]
        server_config = Message(from_socket=self.socket, expected_title="MSG_SERVER_CONFIG", config=self.config).content
        self.config.update(server_config,
                           exceptions=["CLIENT_INDEX", "INDEX", "SERVER_ADDRESS", "DATASET_PATH"])
        self.config.client_init()
        server_time = Message(from_socket=self.socket, expected_title="MSG_SERVER_TIME", config=self.config).content
        self.config.TIME_DIFFERENCE = server_time - time.time()
        Message(title="MSG_CLIENT_CONFIG", content=self.config, config=self.config).send_to(self.socket)
        self.config.get_vocab()
        self.config.get_model_info()
        connection_state = Message(
            from_socket=self.socket, expected_title="MSG_CONNECTION_STATE", config=self.config).content
        logger.info(connection_state)

    def set_comm(self, comm):
        self.comm = comm

    def set_data_loader(self):
        data_reader = DataReader(self.config, set_name='train')
        self.data_loader = data_reader.get_data_loader()

    def receive_net(self):
        self.net = Message(from_socket=self.socket, expected_title="MSG_INITIAL_MODEL", config=self.config).content
        self.optimiser = get_optimiser(self.net, self.config.LAYER_NUM_ON_CLIENT, self.config.LEARNING_RATE,
                                       self.config.OPTIM, self.config.WEIGHT_DECAY)
        self.global_client_model = self.net.state_dict()

    def train(self):
        self.set_data_loader()
        self.client_timing.start()
        early_stop = False
        for epoch_index in range(self.config.EPOCH_NUM):
            # early stopping
            if epoch_index > 0 and self.config.EARLY_STOP and self.receive_stop_signal():
                early_stop = True
                break
            epoch_timing = self.client_timing.break_down('epoch', epoch_index)
            epoch_timing.start()
            # torch.set_num_threads(self.config.CLIENT_CPU_NUM)
            self.net.to(self.device)
            if self.config.LAYER_NUM_ON_CLIENT == -1:
                self.train_on_client(epoch_index, epoch_timing.start_time)
            elif self.config.LAYER_NUM_ON_CLIENT == 0:
                self.train_on_server(epoch_index, epoch_timing.start_time)
            else:
                self.train_split(epoch_index, epoch_timing.start_time)
            self.global_client_model = self.net.state_dict()
            epoch_timing.stop()
        if self.config.EARLY_STOP and not early_stop:
            self.receive_stop_signal()
        self.client_timing.stop()
        self.client_timing.accumulate()

    def train_on_client(self, epoch_index, start_time, *args, **kwargs):
        # train
        self.net.train()
        bar = tqdm.tqdm(self.data_loader)
        train_loss = 0.
        correct = 0
        total = 0
        total_intersection = [0 for _ in range(self.config.NUM_CLASS)]
        total_union = [0 for _ in range(self.config.NUM_CLASS)]
        batch_idx = 0
        iter_timing = self.client_timing.break_down('iter', epoch_index, 0)
        iter_timing.start()
        load_timing = self.client_timing.break_down('load', epoch_index, 0)
        load_timing.start()
        for inputs, targets in bar:
            load_timing.stop()
            bar.set_description("Epoch {}/{}".format(epoch_index + 1, self.config.EPOCH_NUM))
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # forward pass
            forward_timing = self.client_timing.break_down('forward', epoch_index, batch_idx)
            forward_timing.start()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            forward_timing.stop()
            train_loss += loss.item()
            if self.config.TASK_TYPE == 'semantic_segmentation':
                predicted_masks = torch.argmax(outputs, dim=1)
                for i in range(self.config.NUM_CLASS):
                    intersection = ((predicted_masks == i) & (targets == i)).sum().item()
                    union = ((predicted_masks == i) | (targets == i)).sum().item()
                    total_intersection[i] += intersection
                    total_union[i] += union
            else:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            # backward pass
            backward_timing = self.client_timing.break_down('backward', epoch_index, batch_idx)
            backward_timing.start()
            loss.backward()
            backward_timing.stop()
            if self.config.ASYNC_COMM:
                self.comm.send(Message(
                    title='MSG_LOCAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                    content=loss.item(),
                    config=self.config
                ))
            else:
                Message(
                    title='MSG_LOCAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                    content=loss.item(),
                    config=self.config
                ).send_to(self.socket)
            if self.config.CLIP_GRAD_NORM:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.CLIP_GRAD_NORM)
            # param update
            update_timing = self.client_timing.break_down('update', epoch_index, batch_idx)
            update_timing.start()
            self.optimiser.step()
            self.optimiser.zero_grad()
            update_timing.stop()
            # Aggregation.
            straggler_timing = self.client_timing.break_down('straggler', epoch_index, batch_idx)
            straggler_timing.start()
            self.aggregation(epoch_index, batch_idx)
            straggler_timing.stop()
            iter_timing.stop()
            bar.set_postfix({"train_loss": "{:.2f}".format(loss),
                             "time": "{:.2f}".format(iter_timing.stop_time - start_time)})
            batch_idx += 1
            iter_timing = self.client_timing.break_down('iter', epoch_index, batch_idx)
            iter_timing.start()
            load_timing = self.client_timing.break_down('load', epoch_index, batch_idx)
            load_timing.start()
        load_timing.stop()
        iter_timing.stop()
        if self.config.ASYNC_COMM:
            self.comm.send(Message(
                title='MSG_LOCAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                end=True,
                config=self.config
            ))
        else:
            Message(
                title='MSG_LOCAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                end=True,
                config=self.config
            ).send_to(self.socket)
        train_loss /= batch_idx
        if self.config.TASK_TYPE == 'semantic_segmentation':
            train_acc = statistics.mean(
                [100. * total_intersection[i] / total_union[i]
                 for i in range(1, self.config.NUM_CLASS) if total_union[i]])
        else:
            train_acc = 100. * correct / total if total else 0
        if self.config.ASYNC_COMM:
            self.comm.send(Message(
                title='MSG_LOCAL_TRAIN_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                content=[train_loss, train_acc],
                config=self.config
            ))
        else:
            Message(
                title='MSG_LOCAL_TRAIN_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                content=[train_loss, train_acc],
                config=self.config
            ).send_to(self.socket)

    def train_on_server(self, epoch_index, start_time, *args, **kwargs):
        # train
        bar = tqdm.tqdm(self.data_loader)
        current_loss = 0.
        batch_idx = 0
        iter_timing = self.client_timing.break_down('iter', epoch_index, 0)
        iter_timing.start()
        load_timing = self.client_timing.break_down('load', epoch_index, 0)
        load_timing.start()
        for inputs, targets in bar:
            load_timing.stop()
            idle_timing = self.client_timing.break_down('idle', epoch_index, batch_idx)
            idle_timing.start()
            bar.set_description("Epoch {}/{}".format(epoch_index + 1, self.config.EPOCH_NUM))
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                content=[inputs.cpu(), targets.cpu()],
                config=self.config
            ).send_to(self.socket)
            idle_timing.stop()
            # Aggregation.
            straggler_timing = self.client_timing.break_down('straggler', epoch_index, batch_idx)
            straggler_timing.start()
            self.aggregation(epoch_index, batch_idx)
            straggler_timing.stop()
            iter_timing.stop()
            bar.set_postfix({"train_loss": "{:.2f}".format(current_loss),
                             "time": "{:.2f}".format(iter_timing.stop_time - start_time)})
            batch_idx += 1
            iter_timing = self.client_timing.break_down('iter', epoch_index, batch_idx)
            iter_timing.start()
            load_timing = self.client_timing.break_down('load', epoch_index, batch_idx)
            load_timing.start()
        load_timing.stop()
        iter_timing.stop()
        if self.config.ASYNC_COMM:
            self.comm.send(Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                end=True,
                config=self.config))
        else:
            self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                end=True,
                config=self.config
            ).send_to(self.socket)

    def train_split(self, epoch_index, start_time, *args, **kwargs):
        # train
        self.net.train()
        bar = tqdm.tqdm(self.data_loader)
        loss = 0.
        iter_index = 0
        outputs_list = []
        # forward pass
        iter_timing = self.client_timing.break_down('iter', epoch_index, iter_index)
        iter_timing.start()
        load_timing = self.client_timing.break_down('load', epoch_index, iter_index)
        load_timing.start()
        for inputs, targets in bar:
            load_timing.stop()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            bar.set_description("Epoch {}/{}".format(epoch_index + 1, self.config.EPOCH_NUM))
            # forward pass
            for pipe_idx, (pipe_inputs, pipe_targets) in enumerate(
                    split_batch(inputs, targets, partition_num=self.config.PIPE_BATCH_NUM, keep_last=False)):
                forward_timing = self.client_timing.break_down('forward', epoch_index, iter_index)
                forward_timing.start()
                pipe_outputs = self.net(pipe_inputs)
                forward_timing.stop()
                # upload activation
                idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
                idle_timing.start()
                if self.config.ASYNC_COMM:
                    self.comm.send(Message(
                        title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                        content=[pipe_outputs, pipe_targets],
                        msg_type="train",
                        config=self.config))
                else:
                    self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                        title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                        content=[pipe_outputs, pipe_targets],
                        msg_type="train",
                        config=self.config
                    ).send_to(self.socket)
                idle_timing.stop()
                outputs_list.append(pipe_outputs)
            # backward pass
            for pipe_outputs in outputs_list:
                idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
                idle_timing.start()
                if self.config.ASYNC_COMM:
                    message = self.comm.receive()
                else:
                    message = Message(
                        from_socket=self.socket,
                        expected_title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX),
                        time_difference=self.config.TIME_DIFFERENCE,
                        config=self.config
                    )
                download_timing = self.client_timing.break_down('download', epoch_index, iter_index)
                download_timing.set_start_time(message.send_time)
                download_timing.set_stop_time(message.receive_time)
                gradients, loss = message.content
                idle_timing.stop()
                backward_timing = self.client_timing.break_down('backward', epoch_index, iter_index)
                gradients = gradients.to(self.device)
                backward_timing.start()
                pipe_outputs.backward(gradients)
                backward_timing.stop()
            if self.config.CLIP_GRAD_NORM:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.CLIP_GRAD_NORM)
            update_timing = self.client_timing.break_down('update', epoch_index, iter_index)
            update_timing.start()
            self.optimiser.step()
            self.optimiser.zero_grad()
            update_timing.stop()
            # Aggregation.
            straggler_timing = self.client_timing.break_down('straggler', epoch_index, iter_index)
            straggler_timing.start()
            self.aggregation(epoch_index, iter_index)
            straggler_timing.stop()
            outputs_list.clear()
            iter_timing.stop()
            iter_index += 1
            bar.set_postfix({"train_loss": "{:.2f}".format(loss),
                             "time": "{:.2f}".format(iter_timing.stop_time - start_time)})
            iter_timing = self.client_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            load_timing = self.client_timing.break_down('load', epoch_index, iter_index)
            load_timing.start()
        load_timing.stop()
        iter_timing.stop()
        if self.config.ASYNC_COMM:
            self.comm.send(Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                msg_type="train",
                end=True,
                config=self.config))
        else:
            self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                end=True,
                config=self.config
            ).send_to(self.socket)

    def aggregation(self, epoch_index, iter_index):
        if (iter_index + 1) % self.config.AGGREGATION_FREQ == 0:
            self.net.to('cpu')
            self.global_client_model = self.net.state_dict()
            self.send_client_params()
            self.receive_client_params(epoch_index, iter_index)
            self.net.load_state_dict(self.global_client_model)
            self.net.to(self.device)

    def test(self):
        if self.comm:
            loss_message = self.comm.receive().content
        else:
            loss_message = Message(
                from_socket=self.socket,
                expected_title='MSG_SERVER_LOSS_SERVER_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX),
                config=self.config
            ).content
        self.test_accuracy = loss_message

    def profile(self):
        profiler = Profiler(self.config)
        self.profile_res = profiler.run()

    def send_client_profile_res(self):
        Message(
            title='MSG_PROFILE_RES_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
            content=self.profile_res,
            config=self.config
        ).send_to(self.socket)

    def send_client_params(self):
        if self.comm:
            self.comm.send(Message(
                title='MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                msg_type='agg',
                content=self.global_client_model,
                config=self.config
            ))
        else:
            self.communication_amount["AGGREGATION"] = self.communication_amount.get("AGGREGATION", 0) + Message(
                title='MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                msg_type='agg',
                content=self.global_client_model,
                config=self.config
            ).send_to(self.socket)

    def receive_client_params(self, epoch_index, iter_index):
        if self.comm:
            message = self.comm.receive(msg_type='agg')
        else:
            message = Message(
                from_socket=self.socket,
                expected_title='MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX),
                time_difference=self.config.TIME_DIFFERENCE,
                config=self.config
            )
        download_timing = self.client_timing.break_down('download', epoch_index, iter_index)
        download_timing.set_start_time(message.send_time)
        download_timing.set_stop_time(message.receive_time)
        self.global_client_model = message.content

    def send_client_communication_amount(self):
        if self.comm:
            self.comm.send(Message(
                title="MSG_COMMUNICATION_AMOUNT_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
                content=self.comm.amount,
                config=self.config))
        else:
            Message(
                title="MSG_COMMUNICATION_AMOUNT_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
                content=sum(self.communication_amount.values()),
                config=self.config
            ).send_to(self.socket)

    def send_client_timing(self):
        if self.comm:
            self.comm.send(Message(
                title="MSG_TIMING_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
                content=self.client_timing,
                config=self.config))
        else:
            Message(
                title="MSG_TIMING_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
                content=self.client_timing,
                config=self.config
            ).send_to(self.socket)

    def receive_sync_flag(self):
        return Message(
            from_socket=self.socket,
            expected_title='MSG_SYNC_FLAG_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX),
            config=self.config
        ).content

    def receive_recommend_parameters(self):
        self.config.LAYER_NUM_ON_CLIENT, self.config.PIPE_BATCH_NUM = Message(
            from_socket=self.socket,
            expected_title='MSG_RECOMMEND_PARAMS_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX),
            config=self.config
        ).content

    def receive_stop_signal(self):
        if self.comm:
            stop_signal = self.comm.receive().content
        else:
            stop_signal = Message(
                from_socket=self.socket,
                expected_title="MSG_STOP_SIGNAL_TO_CLIENT_{}".format(self.config.CLIENT_INDEX),
                config=self.config
            ).content
        return stop_signal
