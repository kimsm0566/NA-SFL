import logging
import torch
import tqdm
import time
import statistics
from copy import deepcopy
from queue import Empty
from communication.message import Message
from util.timing import Timing
from util.util import get_optimiser, merge_models, split_models
from util.profile import Profiler
from data_preprocessing.data_reader import DataReader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ServerModel:
    def __init__(self,
                 client_index,
                 ip_address,
                 port,
                 server_socket,
                 client_socket,
                 device,
                 net,
                 client_net,
                 complete_net,
                 criterion,
                 config):
        self.client_index = client_index
        self.ip_address = ip_address
        self.port = port
        self.server_socket = server_socket
        self.client_socket = client_socket
        self.device = device
        self.net = net
        self.client_net = client_net
        self.complete_net = complete_net
        self.optimiser = None
        self.criterion = criterion
        self.config = config
        self.comm = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.start_time = None
        self.loss = None
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.time_line = []
        self.test_loss = 0.
        self.test_acc = 0.
        self.global_client_model = None
        self.global_server_model = None
        self.global_model = None
        self.best_val_step = None
        self.best_client_model = None
        self.best_server_model = None
        self.converge_time = None
        self.server_communication_amount = {}
        self.client_communication_amount = {}
        self.server_timing = Timing('server')
        self.client_timing = None
        self.server_profile_res = None
        self.client_profile_res = None
        self.client_version = 0

    def update_net(self, net, client_net, complete_net):
        self.net = net
        self.client_net = client_net
        self.complete_net = complete_net
        self.optimiser = get_optimiser(self.net, self.config.LAYER_NUM_ON_CLIENT, self.config.LEARNING_RATE,
                                       self.config.OPTIM, self.config.WEIGHT_DECAY)
        self.global_server_model = self.net.state_dict()

    def set_comm(self, comm):
        self.comm = comm

    def set_data_loader(self):
        data_reader = DataReader(self.config, 'test')
        self.val_data_loader, self.test_data_loader = data_reader.get_data_loader()

    def train(self, aggregator, shared_info):
        self.server_timing.start()
        self.start_time = time.time()
        if self.client_index == 0:
            bar = tqdm.trange(self.config.EPOCH_NUM)
            bar.set_description("Training")
        else:
            bar = range(self.config.EPOCH_NUM)
        for epoch_index in bar:
            epoch_start_time = time.time()
            epoch_timing = self.server_timing.break_down('epoch', epoch_index)
            epoch_timing.start()
            if self.config.LAYER_NUM_ON_CLIENT == -1:
                end = self.train_on_client(epoch_index, aggregator, shared_info)
            elif self.config.LAYER_NUM_ON_CLIENT == 0:
                end = self.train_on_server(epoch_index, aggregator, shared_info)
            else:
                end = self.train_split(epoch_index, aggregator, shared_info)
            epoch_timing.stop()
            if self.client_index == 0:
                postfix = {
                    'total_time': "{:.2f}".format(time.time() - self.start_time),
                    'epoch_time': "{:.2f}".format(time.time() - epoch_start_time),
                    'train_loss': "{:.2f}".format(self.train_loss[-1]),
                    'train_acc': "{:.2f}".format(self.train_acc[-1]),
                }
                if self.config.ENABLE_VAL and self.val_loss:
                    postfix.update({
                        'val_loss': "{:.2f}".format(self.val_loss[-1]),
                        'val_acc': "{:.2f}".format(self.val_acc[-1]),
                        'best_step': str(self.best_val_step),
                        "best_acc": 0 if self.best_val_step is None else self.val_acc[self.best_val_step]
                    })
                bar.set_postfix(postfix)
            if self.config.EARLY_STOP and end:
                break
        self.server_timing.stop()

    def train_on_client(self, epoch_index, aggregator, shared_info):
        # train
        iter_index = 0
        idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
        idle_timing.start()
        while True:
            iter_timing = self.server_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            if self.comm:
                message = self.comm.receive()
            else:
                message = Message(from_socket=self.client_socket,
                                  expected_title='MSG_LOCAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.client_index))
            if message.end:
                break
            idle_timing.stop()
            self.loss = message.content
            # Client model aggregation.
            if (iter_index + 1) % self.config.AGGREGATION_FREQ == 0:
                idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
                idle_timing.start()
                self.receive_client_params(epoch_index, iter_index)
                idle_timing.stop()
                aggregate_timing = self.server_timing.break_down('aggregate', epoch_index, iter_index)
                self.global_client_model = self.aggregation(self.global_client_model, aggregator, aggregate_timing)
                self.send_client_params()
            iter_index += 1
            idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
            idle_timing.start()
            iter_timing.stop()
        idle_timing.stop()
        if self.comm:
            message = self.comm.receive()
        else:
            message = Message(from_socket=self.client_socket,
                              expected_title='MSG_LOCAL_TRAIN_LOSS_CLIENT_{}_TO_SERVER'.format(self.client_index))
        train_loss, train_acc = message.content
        self.train_loss.append(round(train_loss, 2))
        self.train_acc.append(round(train_acc, 2))
        # val
        if self.config.ENABLE_VAL:
            if (epoch_index + 1) % self.config.VAL_FREQ == 0:
                return self.val(shared_info, asynchronous=self.config.ASYNC_AGG)
            else:
                self.send_stop_signal(False)
        return False

    def train_on_server(self, epoch_index, aggregator, shared_info):
        # train
        self.net.to(self.device)
        self.net.train()
        iter_index = 0
        train_loss = 0.
        correct = 0
        total = 0
        total_intersection = [0 for _ in range(self.config.NUM_CLASS)]
        total_union = [0 for _ in range(self.config.NUM_CLASS)]
        idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
        idle_timing.start()
        while True:
            iter_timing = self.server_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            message = Message(
                from_socket=self.client_socket,
                expected_title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.client_index),
                time_difference=self.config.TIME_DIFFERENCE
            )
            if message.end:
                break
            idle_timing.stop()
            upload_timing = self.server_timing.break_down('upload', epoch_index, iter_index)
            upload_timing.set_start_time(message.send_time)
            upload_timing.set_stop_time(message.receive_time)
            forward_timing = self.server_timing.break_down('forward', epoch_index, iter_index)
            forward_timing.start()
            activations, labels = message.content
            inputs, targets = activations.to(self.device), labels.to(self.device)
            inputs.retain_grad()
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
            backward_timing = self.server_timing.break_down('backward', epoch_index, iter_index)
            backward_timing.start()
            loss.backward()
            backward_timing.stop()
            if self.config.CLIP_GRAD_NORM:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.CLIP_GRAD_NORM)
            self.loss = loss.item()
            update_timing = self.server_timing.break_down('update', epoch_index, iter_index)
            update_timing.start()
            self.optimiser.step()
            self.optimiser.zero_grad()
            update_timing.stop()
            # Server model aggregation.
            if (iter_index + 1) % self.config.AGGREGATION_FREQ == 0:
                self.net.to('cpu')
                aggregate_timing = self.server_timing.break_down('aggregate', epoch_index, iter_index)
                self.global_server_model = self.aggregation(self.net.state_dict(), aggregator, aggregate_timing)
                self.net.load_state_dict(self.global_server_model)
                self.net.to(self.device)
            iter_index += 1
            idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
            idle_timing.start()
            iter_timing.stop()
        idle_timing.stop()
        train_loss /= iter_index
        self.train_loss.append(round(train_loss, 2))
        if self.config.TASK_TYPE == 'semantic_segmentation':
            train_acc = statistics.mean(
                [100. * total_intersection[i] / total_union[i]
                 for i in range(1, self.config.NUM_CLASS) if total_union[i]])
        else:
            train_acc = 100. * correct / total if total else 0
        self.train_acc.append(round(train_acc, 2))
        # val
        if self.config.ENABLE_VAL:
            if (epoch_index + 1) % self.config.VAL_FREQ == 0:
                return self.val(shared_info, asynchronous=self.config.ASYNC_AGG)
            else:
                self.send_stop_signal(False)
        return False

    def train_split(self, epoch_index, aggregator, shared_info):
        # train
        self.net.to(self.device)
        self.net.train()
        iter_index = 0
        train_loss = 0.
        correct = 0
        total = 0
        total_intersection = [0 for _ in range(self.config.NUM_CLASS)]
        total_union = [0 for _ in range(self.config.NUM_CLASS)]
        while True:
            iter_timing = self.server_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
            idle_timing.start()
            end = False
            pipe_loss = 0.
            pipe_batch_num = self.config.BATCH_SIZE // (self.config.BATCH_SIZE // self.config.PIPE_BATCH_NUM)
            for pipe_idx in range(pipe_batch_num):
                if self.comm:
                    message = self.comm.receive()
                else:
                    message = Message(
                        from_socket=self.client_socket,
                        expected_title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.client_index),
                        time_difference=self.config.TIME_DIFFERENCE
                    )
                if message.end:
                    end = True
                    break
                upload_timing = self.server_timing.break_down('upload', epoch_index, iter_index)
                upload_timing.set_start_time(message.send_time)
                upload_timing.set_stop_time(message.receive_time)
                idle_timing.stop()
                forward_timing = self.server_timing.break_down('forward', epoch_index, iter_index)
                forward_timing.start()
                activations, labels = message.content
                inputs, targets = activations.to(self.device), labels.to(self.device)
                inputs.retain_grad()
                outputs = self.net(inputs)
                if self.config.IMMEDIATE_UPDATE:
                    loss = self.criterion(outputs, targets)
                    pipe_loss = loss.item()
                    train_loss += loss.item() / self.config.PIPE_BATCH_NUM
                else:
                    loss = self.criterion(outputs, targets) / self.config.PIPE_BATCH_NUM
                    pipe_loss += loss.item()
                    train_loss += loss.item()
                forward_timing.stop()
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
                backward_timing = self.server_timing.break_down('backward', epoch_index, iter_index)
                backward_timing.start()
                loss.backward()
                backward_timing.stop()
                if self.config.CLIP_GRAD_NORM:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.CLIP_GRAD_NORM)
                self.loss = loss.item()
                if self.config.IMMEDIATE_UPDATE:
                    update_timing = self.server_timing.break_down('update', epoch_index, iter_index)
                    update_timing.start()
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                    update_timing.stop()
                # Send gradients to client.
                idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
                idle_timing.start()
                if self.comm:
                    self.comm.send(Message(
                        title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(self.client_index),
                        content=[inputs.grad, pipe_loss],
                    ))
                else:
                    self.server_communication_amount["TRAINING"] = self.server_communication_amount.get(
                        "TRAINING", 0) + Message(
                        title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(self.client_index),
                        content=[inputs.grad, pipe_loss]
                    ).send_to(self.client_socket)
            if end:
                idle_timing.stop()
                break
            if not self.config.IMMEDIATE_UPDATE:
                update_timing = self.server_timing.break_down('update', epoch_index, iter_index)
                update_timing.start()
                self.optimiser.step()
                self.optimiser.zero_grad()
                update_timing.stop()
            # Model aggregation.
            if (iter_index + 1) % self.config.AGGREGATION_FREQ == 0:
                idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
                idle_timing.start()
                self.receive_client_params(epoch_index, iter_index)
                idle_timing.stop()
                aggregate_timing = self.server_timing.break_down('aggregate', epoch_index, iter_index)
                self.net.to('cpu')
                self.global_model = self.complete_net.state_dict()
                self.global_server_model = self.net.state_dict()
                merge_models(self.global_client_model, self.global_server_model, self.global_model)
                if self.config.ASYNC_AGG:
                    self.global_model = self.async_aggregation(self.global_model, aggregator, aggregate_timing)
                else:
                    self.global_model = self.aggregation(self.global_model, aggregator, aggregate_timing)
                split_models(self.global_client_model, self.global_server_model, self.global_model)
                self.send_client_params()
                self.net.load_state_dict(self.global_server_model)
                self.net.to(self.device)
            iter_index += 1
            iter_timing.stop()
        train_loss /= iter_index
        self.train_loss.append(round(train_loss, 2))
        if self.config.TASK_TYPE == 'semantic_segmentation':
            train_acc = statistics.mean(
                [100. * total_intersection[i] / total_union[i]
                 for i in range(1, self.config.NUM_CLASS) if total_union[i]])
        else:
            train_acc = 100. * correct / total if total else 0
        self.train_acc.append(round(train_acc, 2))
        # val
        if self.config.ENABLE_VAL:
            if (epoch_index + 1) % self.config.VAL_FREQ == 0:
                return self.val(shared_info, asynchronous=self.config.ASYNC_AGG)
            else:
                self.send_stop_signal(False)
        return False

    def aggregation(self, model, aggregator, timing=None):
        aggregator.add_model(model)
        if self.client_index == 0:
            aggregator.aggregate(timing)
        global_model = aggregator.get_global_models()
        aggregator.task_done()
        return global_model

    def async_aggregation(self, model, aggregator, timing=None):
        global_model, self.client_version = aggregator.aggregate(model, self.client_version, timing)
        return global_model

    def val(self, shared_info, asynchronous=False):
        if self.client_index != 0:
            if self.config.EARLY_STOP:
                block = False if asynchronous else True
                try:
                    end = shared_info.get(block)
                except Empty:
                    end = False
                if end is None:
                    end = False
                self.send_stop_signal(end)
                return end
            return False
        self.client_net.load_state_dict(self.global_client_model)
        self.client_net.to(self.device)
        self.client_net.eval()
        self.net.load_state_dict(self.global_server_model)
        self.net.to(self.device)
        self.net.eval()
        val_loss = 0.
        val_iter_index = 0
        total = 0
        correct = 0
        total_intersection = [0 for _ in range(self.config.NUM_CLASS)]
        total_union = [0 for _ in range(self.config.NUM_CLASS)]
        with torch.no_grad():
            for inputs, targets in self.val_data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                activations = self.client_net(inputs)
                outputs = self.net(activations)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                if self.config.TASK_TYPE == 'semantic_segmentation':
                    predicted_masks = torch.argmax(outputs, dim=1)
                    for i in range(1, self.config.NUM_CLASS):
                        intersection = ((predicted_masks == i) & (targets == i)).sum().item()
                        union = ((predicted_masks == i) | (targets == i)).sum().item()
                        total_intersection[i] += intersection
                        total_union[i] += union
                else:
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                val_iter_index += 1
            val_loss /= val_iter_index
            if self.config.TASK_TYPE == 'semantic_segmentation':
                val_acc = statistics.mean(
                    [100. * total_intersection[i] / total_union[i]
                     for i in range(1, self.config.NUM_CLASS) if total_union[i]])
            else:
                val_acc = 100. * correct / total if total else 0
            current_time = (time.time() - self.start_time) / self.config.SIMULATED_CLIENT_NUM
        val_step = len(self.val_acc)
        self.val_loss.append(round(val_loss, 2))
        self.val_acc.append(round(val_acc, 2))
        self.time_line.append(round(current_time))
        end = False
        if self.best_val_step is None or val_acc > self.val_acc[self.best_val_step]:
            self.best_val_step = val_step
            self.best_client_model = deepcopy(self.global_client_model)
            self.best_server_model = deepcopy(self.global_server_model)
            self.converge_time = current_time
        if self.config.EARLY_STOP:
            end = self.check_stop_condition(val_step)
            if not asynchronous or end:
                for _ in range(self.config.CLIENT_NUM - 1):
                    shared_info.put(end)
            self.send_stop_signal(end)
        return end

    def test(self, shared_info):
        if self.client_index != 0:
            self.test_loss, self.test_acc = shared_info.get()
        else:
            if self.config.ENABLE_VAL:
                self.client_net.load_state_dict(self.best_client_model)
                self.net.load_state_dict(self.best_server_model)
            else:
                self.client_net.load_state_dict(self.global_client_model)
                self.net.load_state_dict(self.global_server_model)
            self.client_net.to(self.device)
            self.client_net.eval()
            self.net.to(self.device)
            self.net.eval()
            with torch.no_grad():
                test_iter_index = 0
                correct = 0
                total = 0
                total_intersection = [0 for _ in range(self.config.NUM_CLASS)]
                total_union = [0 for _ in range(self.config.NUM_CLASS)]
                for inputs, targets in self.test_data_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    activations = self.client_net(inputs)
                    outputs = self.net(activations)
                    loss = self.criterion(outputs, targets)
                    self.test_loss += loss.item()
                    if self.config.TASK_TYPE == 'semantic_segmentation':
                        predicted_masks = torch.argmax(outputs, dim=1)
                        for i in range(1, self.config.NUM_CLASS):
                            intersection = ((predicted_masks == i) & (targets == i)).sum().item()
                            union = ((predicted_masks == i) | (targets == i)).sum().item()
                            total_intersection[i] += intersection
                            total_union[i] += union
                    else:
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                    test_iter_index += 1
                self.test_loss /= test_iter_index
                if self.config.TASK_TYPE == 'semantic_segmentation':
                    self.test_acc = statistics.mean(
                        [100. * total_intersection[i] / total_union[i]
                         for i in range(1, self.config.NUM_CLASS) if total_union[i]])
                else:
                    self.test_acc = 100. * correct / total if total else 0
            for _ in range(self.config.CLIENT_NUM - 1):
                shared_info.put([self.test_loss, self.test_acc])
        # Send test loss and acc to client.
        self.server_communication_amount["TESTING"] = self.server_communication_amount.get(
            "TESTING", 0) + Message(
            title='MSG_SERVER_LOSS_SERVER_TO_CLIENT_{}'.format(self.client_index),
            content=self.test_acc
        ).send_to(self.client_socket)

    def profile(self):
        profiler = Profiler(self.config)
        self.server_profile_res = profiler.run()

    def send_recommend_parameters(self):
        Message(
            title='MSG_RECOMMEND_PARAMS_TO_CLIENT_{}'.format(self.client_index),
            content=[self.config.LAYER_NUM_ON_CLIENT, self.config.PIPE_BATCH_NUM]
        ).send_to(self.client_socket)

    def send_client_params(self):
        if self.comm:
            self.comm.send(Message(
                title='MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}'.format(self.client_index),
                content=self.global_client_model,
                msg_type='agg'
            ))
        else:
            self.server_communication_amount["AGGREGATION"] = self.server_communication_amount.get(
                "AGGREGATION", 0) + Message(
                title='MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}'.format(self.client_index),
                content=self.global_client_model,
                msg_type='agg'
            ).send_to(self.client_socket)

    def send_sync_flag(self):
        Message(
            title='MSG_SYNC_FLAG_TO_CLIENT_{}'.format(self.client_index),
            content="Client {} synchronised.".format(self.client_index)
        ).send_to(self.client_socket)

    def receive_client_params(self, epoch_index, iter_index):
        if self.comm:
            message = self.comm.receive('agg')
        else:
            message = Message(
                from_socket=self.client_socket,
                expected_title='MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.client_index),
                time_difference=self.config.TIME_DIFFERENCE
            )
        upload_timing = self.server_timing.break_down('upload', epoch_index, iter_index)
        upload_timing.set_start_time(message.send_time)
        upload_timing.set_stop_time(message.receive_time)
        self.global_client_model = message.content

    def receive_client_communication_amount(self):
        if self.comm:
            self.client_communication_amount = self.comm.receive().content
        else:
            self.client_communication_amount = Message(
                from_socket=self.client_socket,
                expected_title="MSG_COMMUNICATION_AMOUNT_CLIENT_{}_TO_SERVER".format(self.client_index)
            ).content

    def receive_client_timing(self):
        if self.comm:
            self.client_timing = self.comm.receive().content
        else:
            self.client_timing = Message(
                from_socket=self.client_socket,
                expected_title="MSG_TIMING_CLIENT_{}_TO_SERVER".format(self.client_index)
            ).content

    def receive_client_profile_res(self):
        self.client_profile_res = Message(
            from_socket=self.client_socket,
            expected_title="MSG_PROFILE_RES_CLIENT_{}_TO_SERVER".format(self.client_index)
        ).content

    def check_stop_condition(self, val_step):
        if val_step + 1 >= self.config.MAX_VAL_STEP:
            return True
        if val_step - self.best_val_step >= self.config.NON_IMPROVE_STEP:
            return True
        return False

    def send_stop_signal(self, stop_signal):
        if self.comm:
            self.comm.send(Message(
                title="MSG_STOP_SIGNAL_TO_CLIENT_{}".format(self.client_index),
                content=stop_signal))
        else:
            Message(
                title="MSG_STOP_SIGNAL_TO_CLIENT_{}".format(self.client_index),
                content=stop_signal
            ).send_to(self.client_socket)
