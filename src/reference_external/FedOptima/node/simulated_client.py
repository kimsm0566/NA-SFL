import torch
import tqdm
import statistics
from copy import deepcopy
from communication.message import Message
from node.client import Client
from util.timing import Timing
from util.util import split_batch, get_average_model
from data_preprocessing.data_reader import DataReader


class SimulatedClient(Client):
    def __init__(self, client_config):
        super().__init__(client_config)
        self.data_loaders = []
        self.client_timings = [Timing('client') for _ in range(self.config.SIMULATED_CLIENT_NUM)]
        self.avg_model = None

    def set_data_loader(self):
        data_reader = DataReader(self.config, 'train')
        for simulate_index in range(self.config.SIMULATED_CLIENT_NUM):
            client_index = self.config.CLIENT_INDEX * self.config.SIMULATED_CLIENT_NUM + simulate_index
            self.data_loaders.append(data_reader.get_data_loader(client_index))

    def train(self):
        self.set_data_loader()
        early_stop = False
        for timing in self.client_timings:
            timing.start()
        for epoch_index in range(self.config.EPOCH_NUM):
            # early stopping
            if epoch_index > 0 and self.config.EARLY_STOP and self.receive_stop_signal():
                early_stop = True
                break
            epoch_timings = []
            straggler_timings = []
            for simulate_index in range(self.config.SIMULATED_CLIENT_NUM):
                # change model
                self.client_timing = self.client_timings[simulate_index]
                self.net.load_state_dict(deepcopy(self.global_client_model))
                self.data_loader = self.data_loaders[simulate_index]
                # train
                epoch_timing = self.client_timing.break_down('epoch', epoch_index)
                epoch_timings.append(epoch_timing)
                epoch_timing.start()
                # torch.set_num_threads(self.config.CLIENT_CPU_NUM)
                self.net.to(self.device)
                if self.config.LAYER_NUM_ON_CLIENT == -1:
                    self.train_on_client(epoch_index, epoch_timing.start_time, simulate_index)
                elif self.config.LAYER_NUM_ON_CLIENT == 0:
                    self.train_on_server(epoch_index, epoch_timing.start_time, simulate_index)
                else:
                    self.train_split(epoch_index, epoch_timing.start_time, simulate_index)
                self.average_model(simulate_index)
                straggler_timing = self.client_timing.break_down('straggler', epoch_index)
                straggler_timings.append(straggler_timing)
                epoch_timing.stop()
            for timing in epoch_timings:
                timing.start()
            for timing in straggler_timings:
                timing.start()
            self.aggregation(epoch_index, 0)
            for timing in straggler_timings:
                timing.stop()
            for timing in epoch_timings:
                timing.stop()
        if self.config.EARLY_STOP and not early_stop:
            self.receive_stop_signal()
        for timing in self.client_timings:
            timing.stop()
            timing.accumulate()

    def train_on_client(self, epoch_index, start_time, simulate_index=None, *args, **kwargs):
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
            bar.set_description("Epoch {}/{} Model {}/{}".format(
                epoch_index+1, self.config.EPOCH_NUM, simulate_index+1, self.config.SIMULATED_CLIENT_NUM))
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
        Message(
            title='MSG_LOCAL_TRAIN_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
            content=[train_loss, train_acc],
            config=self.config
        ).send_to(self.socket)
        return False

    def train_on_server(self, epoch_index, start_time, simulate_index=None, *args, **kwargs):
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
            bar.set_description("Epoch {}/{} Model {}/{}".format(
                epoch_index + 1, self.config.EPOCH_NUM, simulate_index, self.config.SIMULATED_CLIENT_NUM))
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                content=[inputs.cpu(), targets.cpu()],
                config=self.config
            ).send_to(self.socket)
            idle_timing.stop()
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
        return False

    def train_split(self, epoch_index, start_time, simulate_index=None, *args, **kwargs):
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
            bar.set_description("Epoch {}/{} Model {}/{}".format(
                epoch_index + 1, self.config.EPOCH_NUM, simulate_index, self.config.SIMULATED_CLIENT_NUM))
            # forward pass
            for pipe_idx, (pipe_inputs, pipe_targets) in enumerate(
                    split_batch(inputs, targets, self.config.PIPE_BATCH_NUM)):
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
                        config=self.config
                    ).send_to(self.socket)
                idle_timing.stop()
                outputs_list.append(pipe_outputs)
            # backward pass
            for pipe_outputs in outputs_list:
                idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
                idle_timing.start()
                if self.config.ASYNC_COMM:
                    gradients, loss = self.comm.receive().content
                else:
                    gradients, loss = Message(
                        from_socket=self.socket,
                        expected_title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX),
                        time_difference=self.config.TIME_DIFFERENCE,
                        config=self.config
                    ).content
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
        return False

    def average_model(self, simulate_index):
        self.net.to('cpu')
        if simulate_index == 0:
            self.avg_model = deepcopy(self.net.state_dict())
        else:
            self.avg_model = get_average_model(
                [self.avg_model, self.net.state_dict()],
                [simulate_index, 1])
        self.net.to(self.device)

    def aggregation(self, epoch_index, iter_index):
        self.global_client_model = self.avg_model
        self.send_client_params()
        self.receive_client_params(epoch_index, iter_index)

    def send_client_timing(self):
        if self.comm:
            self.comm.send(Message(
                title="MSG_TIMING_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
                content=self.client_timings,
                config=self.config))
        else:
            Message(
                title="MSG_TIMING_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
                content=self.client_timings,
                config=self.config
            ).send_to(self.socket)
