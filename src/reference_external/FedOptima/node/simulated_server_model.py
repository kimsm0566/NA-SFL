import tqdm
import time
import torch
import statistics
from copy import deepcopy
from communication.message import Message
from node.server_model import ServerModel
from util.timing import Timing
from util.util import get_average_model, merge_models, split_models


class SimulatedServerModel(ServerModel):
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
        super().__init__(client_index, ip_address, port, server_socket, client_socket,
                         device, net, client_net, complete_net, criterion, config)
        self.avg_model = None
        self.server_timings = [Timing('server') for _ in range(self.config.SIMULATED_CLIENT_NUM)]
        self.client_timings = None

    def train(self, aggregator, shared_info):
        for server_timing in self.server_timings:
            server_timing.start()
        self.start_time = time.time()
        if self.client_index == 0:
            bar = tqdm.trange(self.config.EPOCH_NUM)
            bar.set_description("Training")
        else:
            bar = range(self.config.EPOCH_NUM)
        for epoch_index in bar:
            epoch_timings = []
            straggler_timings = []
            for simulate_index in range(self.config.SIMULATED_CLIENT_NUM):
                # change model
                self.net.load_state_dict(deepcopy(self.global_server_model))
                self.server_timing = self.server_timings[simulate_index]
                # train
                epoch_timing = self.server_timing.break_down('epoch', epoch_index)
                epoch_timings.append(epoch_timing)
                epoch_timing.start()
                self.net.to(self.device)
                if self.config.LAYER_NUM_ON_CLIENT == -1:
                    self.train_on_client(epoch_index, aggregator, shared_info)
                elif self.config.LAYER_NUM_ON_CLIENT == 0:
                    self.train_on_server(epoch_index, aggregator, shared_info)
                else:
                    self.train_split(epoch_index, aggregator, shared_info)
                self.average_model(simulate_index)
            # aggregation
            self.global_server_model = self.avg_model
            aggregate_timing = self.server_timings[0].break_down('aggregate', epoch_index, 0)
            if self.config.LAYER_NUM_ON_CLIENT == -1:
                self.receive_client_params(epoch_index, 0)
                self.global_client_model = self.aggregation(self.global_client_model, aggregator, aggregate_timing)
                self.send_client_params()
            elif self.config.LAYER_NUM_ON_CLIENT == 0:
                self.global_server_model = self.aggregation(self.global_server_model, aggregator, aggregate_timing)
            else:
                self.global_model = self.complete_net.state_dict()
                self.receive_client_params(epoch_index, 0)
                merge_models(self.global_client_model, self.global_server_model, self.global_model)
                self.global_model = self.aggregation(self.global_model, aggregator, aggregate_timing)
                split_models(self.global_client_model, self.global_server_model, self.global_model)
                self.send_client_params()
            self.avg_model = None
            for timing in straggler_timings:
                timing.stop()
            postfix = {
                'train_loss': "{:.2f}".format(self.train_loss[-1]),
                'train_acc': "{:.2f}".format(self.train_acc[-1]),
            }
            # val
            end = False
            if self.config.ENABLE_VAL:
                end = self.val(shared_info)
                if self.client_index == 0:
                    postfix.update({
                        'val_loss': "{:.2f}".format(self.val_loss[-1]),
                        'val_acc': "{:.2f}".format(self.val_acc[-1]),
                    })
            if self.client_index == 0:
                bar.set_postfix(postfix)
            for timing in epoch_timings:
                timing.stop()
            if self.config.EARLY_STOP and end:
                break
        for server_timing in self.server_timings:
            server_timing.stop()

    def train_on_client(self, epoch_index, aggregator, shared_info):
        # train
        iter_index = 0
        idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
        idle_timing.start()
        while True:
            iter_timing = self.server_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            message = Message(from_socket=self.client_socket,
                              expected_title='MSG_LOCAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.client_index))
            if message.end:
                break
            idle_timing.stop()
            self.loss = message.content
            iter_index += 1
            idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
            idle_timing.start()
            iter_timing.stop()
        idle_timing.stop()
        message = Message(from_socket=self.client_socket,
                          expected_title='MSG_LOCAL_TRAIN_LOSS_CLIENT_{}_TO_SERVER'.format(self.client_index))
        train_loss, train_acc = message.content
        self.train_loss.append(round(train_loss, 2))
        self.train_acc.append(round(train_acc, 2))

    def train_on_server(self, epoch_index, aggregator, shared_info):
        # train
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

    def train_split(self, epoch_index, aggregator, shared_info):
        # train
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

    def average_model(self, simulate_index):
        self.net.to('cpu')
        if simulate_index == 0:
            self.avg_model = deepcopy(self.net.state_dict())
        else:
            self.avg_model = get_average_model(
                [self.avg_model, self.net.state_dict()],
                [simulate_index, 1])
        self.net.to(self.device)

    def receive_client_timing(self):
        if self.comm:
            self.client_timings = self.comm.receive().content
        else:
            self.client_timings = Message(
                from_socket=self.client_socket,
                expected_title="MSG_TIMING_CLIENT_{}_TO_SERVER".format(self.client_index)
            ).content
