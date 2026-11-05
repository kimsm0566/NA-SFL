import logging
import time
import tqdm
import torch
import statistics

from copy import deepcopy
from node.node import Node
from communication.message import Message
from util.timing import Timing
from util.util import get_optimiser, split_batch
from util.profile import Profiler
from data_preprocessing.data_reader import DataReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AsyncClient(Node):
    def __init__(self, client_config):
        super().__init__(client_config)
        self.port = None
        self.test_accuracy = 0
        self.global_client_model = None
        self.previous_client_model = None
        self.global_version = 0
        self.self_step_num = 0
        self.other_step_num = 0
        self.client_timing = Timing('client')
        self.profile_res = None
        self.comm = None
        self.data_loader = None
        self.auxiliary_net = None
        self.auxiliary_optimiser = None
        self.global_auxiliary_model = None
        self.lr_decay = False

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
        data_reader = DataReader(self.config, 'train')
        self.data_loader = data_reader.get_data_loader()

    def receive_net(self):
        self.net, self.auxiliary_net = Message(
            from_socket=self.socket, expected_title="MSG_INITIAL_MODEL", config=self.config).content
        self.optimiser = get_optimiser(
            self.net, self.config.LAYER_NUM_ON_CLIENT, self.config.LEARNING_RATE, self.config.OPTIM,
            self.config.WEIGHT_DECAY)
        self.global_client_model = self.net.state_dict()
        self.previous_client_model = deepcopy(self.global_client_model)
        if self.config.AUXILIARY is not None:
            self.global_auxiliary_model = self.net.state_dict()
            self.auxiliary_optimiser = get_optimiser(
                self.auxiliary_net, self.config.LAYER_NUM_ON_CLIENT, self.config.LEARNING_RATE, self.config.OPTIM,
                self.config.WEIGHT_DECAY)

    def train(self):
        self.set_data_loader()
        self.client_timing.start()
        epoch_index = 0
        while True:
            epoch_timing = self.client_timing.break_down('epoch', epoch_index)
            epoch_timing.start()
            # torch.set_num_threads(self.config.CLIENT_CPU_NUM)
            self.net.to(self.device)
            if self.auxiliary_net:
                self.auxiliary_net.to(self.device)
            if self.config.LAYER_NUM_ON_CLIENT == -1:
                end = self.train_on_client(epoch_index, epoch_timing.start_time)
            elif self.config.LAYER_NUM_ON_CLIENT == 0:
                end = self.train_on_server(epoch_index, epoch_timing.start_time)
            else:
                end = self.train_split(epoch_index, epoch_timing.start_time)
            self.global_client_model = self.net.state_dict()
            epoch_timing.stop()
            if end:
                break
            epoch_index += 1
        self.client_timing.stop()
        logger.info("Training done.")

    def train_on_client(self, epoch_index, start_time):
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
            bar.set_description("Epoch {}".format(epoch_index + 1))
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
            if self.config.CLIP_GRAD_NORM:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.CLIP_GRAD_NORM)
            # param update
            update_timing = self.client_timing.break_down('update', epoch_index, batch_idx)
            update_timing.start()
            self.optimiser.step()
            self.optimiser.zero_grad()
            update_timing.stop()
            # check stop condition
            end = self.comm.receive(msg_type='end', block=False)
            if end == -1:
                self.end_train(self.comm.final_client_model)
                return True
            # Aggregation.
            straggler_timing = self.client_timing.break_down('straggler', epoch_index, batch_idx)
            straggler_timing.start()
            # FedBuff
            if self.config.BUFFER_AGG > 1:
                end = self.fedbuff_aggregation(epoch_index, batch_idx)
            # FedAsync
            else:
                end = self.fedasync_aggregation(epoch_index, batch_idx)
            straggler_timing.stop()
            iter_timing.stop()
            bar.set_postfix({"train_loss": "{:.2f}".format(loss),
                             "time": "{:.2f}".format(iter_timing.stop_time - start_time)})
            if end:
                self.end_train(self.comm.final_client_model)
                return True
            batch_idx += 1
            iter_timing = self.client_timing.break_down('iter', epoch_index, batch_idx+1)
            iter_timing.start()
            load_timing = self.client_timing.break_down('load', epoch_index, batch_idx+1)
            load_timing.start()
        load_timing.stop()
        iter_timing.stop()
        train_loss /= batch_idx
        if self.config.TASK_TYPE == 'semantic_segmentation':
            train_acc = statistics.mean(
                [100. * total_intersection[i] / total_union[i]
                 for i in range(1, self.config.NUM_CLASS) if total_union[i]])
        else:
            train_acc = 100. * correct / total if total else 0
        # val
        self.comm.send(Message(
            title='MSG_LOCAL_VAL_REQUEST_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
            content=[train_loss, train_acc],
            msg_type="val",
            config=self.config))
        return False

    def train_on_server(self, epoch_index, start_time):
        raise NotImplementedError

    def train_split(self, epoch_index, start_time):
        # train
        self.net.train()
        if self.auxiliary_net:
            self.auxiliary_net.train()
        bar = tqdm.tqdm(self.data_loader)
        loss = 0.
        act_upload_interval = 5 if self.config.AUXILIARY and not self.config.ASYNC_AGG else 1
        act_check = True if self.config.AUXILIARY and self.config.ASYNC_AGG else False
        iter_index = 0
        # forward pass
        iter_timing = self.client_timing.break_down('iter', epoch_index, iter_index)
        iter_timing.start()
        load_timing = self.client_timing.break_down('load', epoch_index, iter_index)
        load_timing.start()
        for inputs, targets in bar:
            load_timing.stop()
            end = self.comm.receive(block=False)
            if end == -1:
                self.end_train(self.comm.final_client_model)
                return True
            bar.set_description("Epoch {}".format(epoch_index + 1))
            outputs_list = []
            # forward pass
            for pipe_idx, (pipe_inputs, pipe_targets) in enumerate(
                    split_batch(inputs, targets, self.config.PIPE_BATCH_NUM)):
                forward_timing = self.client_timing.break_down('forward', epoch_index, iter_index)
                forward_timing.start()
                pipe_inputs, pipe_targets = pipe_inputs.to(self.device), pipe_targets.to(self.device)
                pipe_outputs = self.net(pipe_inputs)
                forward_timing.stop()
                # print(self.comm.act_avail)
                # upload activation
                if act_upload_interval > 1:
                    if (iter_index + 1) % act_upload_interval == 0:
                        self.comm.send(Message(
                            title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                            content=[pipe_outputs.cpu(), pipe_targets.cpu()],
                            msg_type="train",
                            origin=self.config.CLIENT_INDEX,
                            config=self.config))
                elif not act_check or self.comm.act_avail:
                    self.comm.send(Message(
                        title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                        content=[pipe_outputs.cpu(), pipe_targets.cpu()],
                        msg_type="train",
                        origin=self.config.CLIENT_INDEX,
                        config=self.config))
                    self.comm.act_avail = False
                if self.config.AUXILIARY:
                    if act_upload_interval > 1 and (iter_index + 1) % act_upload_interval == 0:
                        idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
                        idle_timing.start()
                        if not (epoch_index == 0 and iter_index == 0):
                            server_receiver_confirm = self.comm.receive()
                            if server_receiver_confirm == -1:
                                self.end_train(self.comm.final_client_model)
                                idle_timing.stop()
                                return True
                            if server_receiver_confirm.title != "ACTIVATIONS_CONFIRM_TO_CLIENT_{}".format(
                                    self.config.CLIENT_INDEX):
                                raise "Should receive confirmation but received {}".format(server_receiver_confirm.title)
                        idle_timing.stop()
                    forward_timing = self.client_timing.break_down('forward', epoch_index, iter_index)
                    forward_timing.start()
                    local_loss = self.criterion(self.auxiliary_net(pipe_outputs), pipe_targets)
                    loss = local_loss.item()
                    forward_timing.stop()
                    backward_timing = self.client_timing.break_down('backward', epoch_index, iter_index)
                    backward_timing.start()
                    local_loss.backward()
                    backward_timing.stop()
                else:
                    outputs_list.append(pipe_outputs)
            # backward pass
            if self.config.AUXILIARY is None:
                for pipe_outputs in outputs_list:
                    idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
                    idle_timing.start()
                    message = self.comm.receive(msg_type='grad', batch_size=pipe_outputs.size(dim=0))
                    if message == -1:
                        self.end_train(self.comm.final_client_model)
                        idle_timing.stop()
                        return True
                    gradients, loss = message
                    idle_timing.stop()
                    backward_timing = self.client_timing.break_down('backward', epoch_index, iter_index)
                    gradients = gradients.to(self.device)
                    backward_timing.start()
                    pipe_outputs.backward(gradients)
                    backward_timing.stop()
            update_timing = self.client_timing.break_down('update', epoch_index, iter_index)
            update_timing.start()
            if self.auxiliary_optimiser:
                if self.config.CLIP_GRAD_NORM:
                    torch.nn.utils.clip_grad_norm_(self.auxiliary_net.parameters(), self.config.CLIP_GRAD_NORM)
                self.auxiliary_optimiser.step()
                self.auxiliary_optimiser.zero_grad()
            if self.config.CLIP_GRAD_NORM:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.CLIP_GRAD_NORM)
            self.optimiser.step()
            self.optimiser.zero_grad()
            self.self_step_num += 1
            update_timing.stop()
            del outputs_list
            # Aggregation.
            straggler_timing = self.client_timing.break_down('straggler', epoch_index, iter_index)
            straggler_timing.start()
            if self.config.ASYNC_AGG:
                # end = self.async_aggregation(epoch_index, iter_index)
                end = self.fedasync_aggregation(epoch_index, iter_index)
            else:
                end = self.sync_aggregation(epoch_index, iter_index)
            straggler_timing.stop()
            iter_timing.stop()
            iter_index += 1
            bar.set_postfix({"train_loss": "{:.2f}".format(loss),
                             "time": "{:.2f}".format(iter_timing.stop_time - start_time),
                             "step": self.self_step_num})
            if end:
                self.end_train(self.comm.final_client_model)
                return True
            iter_timing = self.client_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            load_timing = self.client_timing.break_down('load', epoch_index, iter_index)
            load_timing.start()
        load_timing.stop()
        iter_timing.stop()
        # val
        self.comm.send(Message(
            title='MSG_LOCAL_VAL_REQUEST_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
            msg_type="val",
            config=self.config))
        return False

    def async_aggregation(self, epoch_index, iter_idx):
        if (iter_idx + 1) % self.config.AGGREGATION_FREQ == 0:
            self.net.to('cpu')
            self.global_client_model = self.net.state_dict()
            if self.auxiliary_net:
                self.auxiliary_net.to('cpu')
                self.global_auxiliary_model = self.auxiliary_net.state_dict()
            self.comm.send(Message(
                title='MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                msg_type='agg',
                content=[[self.global_client_model, self.global_auxiliary_model],
                         [self.self_step_num, self.other_step_num]],
                origin=self.config.CLIENT_INDEX,
                config=self.config
            ))
            message = self.comm.receive(msg_type='agg', block=True)
            if message == -1:
                return True
            download_timing = self.client_timing.break_down('download', epoch_index, iter_idx)
            download_timing.set_start_time(message.send_time)
            download_timing.set_stop_time(message.receive_time)
            global_client_models, step_nums, lr_decay = message.content
            global_client_model, global_auxiliary_model = global_client_models
            self_step_num, other_step_num = step_nums
            # self_update_num = self.self_step_num - self_step_num
            # other_update_num = other_step_num - self.other_step_num
            self.global_client_model = global_client_model
            # self.global_client_model = get_average_model(
            #     [self.global_client_model, global_client_model],
            #     [self_update_num / (self_update_num + other_update_num),
            #      other_update_num / (self_update_num + other_update_num)])
            if global_auxiliary_model:
                # self.global_auxiliary_model = get_average_model(
                #     [self.global_auxiliary_model, global_auxiliary_model],
                #     [self_update_num / (self_update_num + other_update_num),
                #      other_update_num / (self_update_num + other_update_num)])
                self.global_auxiliary_model = global_auxiliary_model
            self.other_step_num = other_step_num
            self.net.load_state_dict(self.global_client_model)
            self.net.to(self.device)
            if self.auxiliary_net:
                self.auxiliary_net.load_state_dict(self.global_auxiliary_model)
                self.auxiliary_net.to(self.device)
            if lr_decay and not self.lr_decay:
                self.decay_learning_rate()
        return False

    def sync_aggregation(self, epoch_index, iter_idx):
        return self.fedasync_aggregation(epoch_index, iter_idx)

    def fedasync_aggregation(self, epoch_index, iter_idx):
        if (iter_idx + 1) % self.config.AGGREGATION_FREQ == 0:
            self.net.to('cpu')
            self.global_client_model = self.net.state_dict()
            if self.auxiliary_net:
                self.auxiliary_net.to('cpu')
                self.global_auxiliary_model = self.auxiliary_net.state_dict()
            self.comm.send(Message(
                title='MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                msg_type='agg',
                content=[self.global_client_model, self.global_auxiliary_model],
                origin=self.config.CLIENT_INDEX,
                config=self.config
            ))
            message = self.comm.receive(msg_type='agg', block=True)
            if message == -1:
                return True
            download_timing = self.client_timing.break_down('download', epoch_index, iter_idx)
            download_timing.set_start_time(message.send_time)
            download_timing.set_stop_time(message.receive_time)
            self.global_client_model, self.global_auxiliary_model = message.content
            self.net.load_state_dict(self.global_client_model)
            self.net.to(self.device)
            if self.auxiliary_net:
                self.auxiliary_net.load_state_dict(self.global_auxiliary_model)
                self.auxiliary_net.to(self.device)
        return False

    def fedbuff_aggregation(self, epoch_index, iter_idx):
        if (iter_idx + 1) % self.config.AGGREGATION_FREQ == 0:
            self.net.to('cpu')
            self.comm.send(Message(
                title='MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                msg_type='agg',
                content=self.net.state_dict(),
                origin=self.config.CLIENT_INDEX,
                config=self.config
            ))
            message = self.comm.receive(msg_type='agg', block=True)
            if message == -1:
                return True
            download_timing = self.client_timing.break_down('download', epoch_index, iter_idx)
            download_timing.set_start_time(message.send_time)
            download_timing.set_stop_time(message.receive_time)
            self.global_client_model = message.content
            self.net.load_state_dict(self.global_client_model)
            self.net.to(self.device)
        return False

    def end_train(self, final_client_model):
        if final_client_model is not None:
            self.global_client_model = final_client_model
            self.net.load_state_dict(self.global_client_model)

    def test(self):
        loss_message = self.comm.receive(msg_type='test')
        self.test_accuracy = loss_message.content

    def profile(self):
        profiler = Profiler(self.config)
        self.profile_res = profiler.run()

    def send_client_profile_res(self):
        Message(
            title='MSG_PROFILE_RES_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
            content=self.profile_res,
            config=self.config
        ).send_to(self.socket)

    def send_client_communication_amount(self):
        self.comm.send(Message(
            title="MSG_COMMUNICATION_AMOUNT_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
            content=self.comm.amount,
            config=self.config))

    def send_client_timing(self):
        self.comm.send(Message(
            title="MSG_TIMING_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
            content=self.client_timing,
            config=self.config))

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

    def decay_learning_rate(self):
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = self.config.LEARNING_RATE * 0.1
        if self.config.AUXILIARY is not None:
            for param_group in self.auxiliary_optimiser.param_groups:
                param_group['lr'] = self.config.LEARNING_RATE * 0.1
        print("Learning rate decayed to {}".format(self.config.LEARNING_RATE * 0.1))
        self.lr_decay = True
