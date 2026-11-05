import logging
import torch
import pickle
import time
import statistics
from copy import deepcopy
from tqdm import tqdm
from communication.message import Message
from util.timing import Timing
from util.util import get_optimiser, get_average_model, update_model, get_model_diff
from util.profile import Profiler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AsyncServerModel:
    def __init__(self,
                 device,
                 net,
                 client_net,
                 auxiliary_net,
                 criterion,
                 connect,
                 config,
                 val_data_loader,
                 test_data_loader,
                 comm):
        self.device = device
        self.net = net
        self.client_net = client_net
        self.auxiliary_net = auxiliary_net
        self.complete_net = None
        self.optimiser = get_optimiser(net, config.LAYER_NUM_ON_CLIENT, config.LEARNING_RATE, config.OPTIM,
                                       config.WEIGHT_DECAY)
        self.criterion = criterion
        self.connect = connect
        self.config = config
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.comm = comm
        self.loss = 0
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.time_line = []
        self.test_loss = 0.
        self.test_acc = 0.
        self.global_client_model = self.client_net.state_dict()
        self.global_client_version = 0
        self.client_versions = {client_index: 0 for client_index in range(config.CLIENT_NUM)}
        self.self_step_nums = {client_index: 0 for client_index in range(config.CLIENT_NUM)}
        self.other_step_nums = {client_index: 0 for client_index in range(config.CLIENT_NUM)}
        self.global_auxiliary_model = self.auxiliary_net.state_dict() if self.config.AUXILIARY else None
        self.val_step = 0
        self.best_val_step = None
        self.best_client_model = None
        self.best_server_model = None
        self.buffered_client_models = dict.fromkeys(range(self.config.CLIENT_NUM), None)
        self.converge_time = None
        self.server_communication_amount = {}
        self.client_communication_amount = {}
        self.server_timing = Timing('server')
        self.server_profile_res = None
        self.client_profile_res = None
        self.sync_final_model = False
        self.lr_decay = False

    def train(self):
        self.net.to(self.device)
        if self.config.LAYER_NUM_ON_CLIENT == -1:
            self.train_on_client()
        elif self.config.LAYER_NUM_ON_CLIENT == 0:
            self.train_on_server()
        else:
            self.train_split()

    def train_on_client(self):
        start_time = time.time()
        iter_index = 0
        val_message_num = 0
        buffered_messages = []
        with tqdm(total=self.config.MAX_VAL_STEP) as pbar:
            pbar.set_description("Training")
            while True:
                idle_timing = self.server_timing.break_down('idle', 0, iter_index)
                idle_timing.start()
                message_type, message = self.receive(self.config.BATCH_SIZE)
                idle_timing.stop()
                if message_type == 'agg':
                    upload_timing = self.server_timing.break_down('upload', 0, iter_index)
                    upload_timing.set_start_time(message.send_time)
                    upload_timing.set_start_time(message.receive_time)
                    # FedBuff
                    if self.config.BUFFER_AGG > 1:
                        buffered_messages.append(message)
                        self.fedbuff_aggregation(buffered_messages)
                    # FedAsync
                    else:
                        self.fedasync_aggregation(message)
                    continue
                if message_type == 'val':
                    val_message_num += 1
                    if val_message_num % self.config.CLIENT_NUM != 0:
                        continue
                    train_loss, train_acc = message.content
                    if self.global_client_model is None:
                        continue
                    end = self.val(start_time, train_loss, train_acc, pbar)
                    if end:
                        break
                    continue
                if message_type == 'test':
                    raise Exception("End the train loop before the test stage.")
                if message_type == 'general':
                    raise Exception("General message {} should not be received now.".format(message.content))
                raise Exception("Unrecognised message type: {}".format(message_type))

    def train_on_server(self):
        raise NotImplementedError

    def train_split(self):
        start_time = time.time()
        iter_index = 0
        train_loss = 0
        train_acc = 0
        buffered_messages = []
        val_message_num = 0
        parallel_outputs = {client_index: [] for client_index in range(self.config.CLIENT_NUM)}
        with tqdm(total=self.config.MAX_VAL_STEP) as pbar:
            pbar.set_description("Training")
            while True:
                iter_timing = self.server_timing.break_down('iter', 0, iter_index)
                iter_timing.start()
                idle_timing = self.server_timing.break_down('idle', 0, iter_index)
                idle_timing.start()
                message_type, message = self.receive(self.config.BATCH_SIZE)
                idle_timing.stop()
                # Train
                if message_type == 'train':
                    self.net.train()
                    activations, labels, activations_index = message
                    forward_timing = self.server_timing.break_down('forward', 0, iter_index)
                    forward_timing.start()
                    inputs, targets = activations.to(self.device), labels.to(self.device)
                    inputs.retain_grad()
                    outputs = self.net(inputs)
                    train_acc = self.calculate_acc(outputs, targets)
                    if self.config.LABEL_ON_DEVICE:
                        self.send_logit(outputs, activations_index)
                        origin = activations_index[0][1]
                        forward_timing.stop()
                        parallel_outputs[origin].append(outputs)
                        continue
                    loss = self.criterion(outputs, targets)
                    train_loss = loss.item()
                    forward_timing.stop()
                    backward_timing = self.server_timing.break_down('backward', 0, iter_index)
                    backward_timing.start()
                    loss.backward()
                    backward_timing.stop()
                    if self.config.CLIP_GRAD_NORM:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.CLIP_GRAD_NORM)
                    # Send gradients to client.
                    if self.config.AUXILIARY is None:
                        self.send_grad(inputs.grad, loss.item(), activations_index)
                    # Update
                    update_timing = self.server_timing.break_down('update', 0, iter_index)
                    update_timing.start()
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                    update_timing.stop()
                    iter_timing.stop()
                    iter_index += 1
                    continue
                if message_type == 'logit':
                    logit_grads, train_loss = message.content
                    logit_grads = logit_grads.to(self.device)
                    outputs = parallel_outputs[message.origin].pop(0)
                    backward_timing = self.server_timing.break_down('backward', 0, iter_index)
                    backward_timing.start()
                    outputs.backward(logit_grads)
                    backward_timing.stop()
                    if self.config.CLIP_GRAD_NORM:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.CLIP_GRAD_NORM)
                    # Send gradients to client.
                    if self.config.AUXILIARY is None:
                        self.send_grad(inputs.grad, loss.item(), activations_index)
                    # Update
                    if self.comm.act_parallel_num == 0:
                        update_timing = self.server_timing.break_down('update', 0, iter_index)
                        update_timing.start()
                        self.optimiser.step()
                        self.optimiser.zero_grad()
                        update_timing.stop()
                    iter_timing.stop()
                    iter_index += 1
                    continue
                # Model aggregation.
                if message_type == 'agg':
                    # Client model aggregation.
                    aggregate_timing = self.server_timing.break_down('aggregate', 0, iter_index)
                    aggregate_timing.start()
                    if isinstance(message, list):
                        origins = []
                        for m in message:
                            upload_timing = self.server_timing.break_down('upload', 0, iter_index)
                            upload_timing.set_start_time(m.send_time)
                            upload_timing.set_stop_time(m.receive_time)
                            origins.append(m.origin)
                    else:
                        upload_timing = self.server_timing.break_down('upload', 0, iter_index)
                        upload_timing.set_start_time(message.send_time)
                        upload_timing.set_stop_time(message.receive_time)
                    self.client_net.to('cpu')
                    if self.config.ASYNC_AGG:
                        if self.config.AGG_ALL_MODELS:
                            self.async_all_aggregation(message)
                        else:
                            # self.async_aggregation(message)
                            self.fedasync_aggregation(message)
                    else:
                        buffered_messages.append(message)
                        self.sync_aggregation(buffered_messages)
                    aggregate_timing.stop()
                    continue
                if message_type == 'val':
                    val_message_num += 1
                    if val_message_num % (self.config.CLIENT_NUM * self.config.VAL_FREQ) != 0:
                        continue
                    if self.global_client_model is None:
                        continue
                    end = self.val(start_time, train_loss, train_acc, pbar)
                    if end:
                        break
                    continue
                if message_type == 'test':
                    raise Exception("End the train loop before the test stage.")
                if message_type == 'general':
                    raise Exception("General message {} should not be received now.".format(message.content))
                raise Exception("Unrecognised message type: {}".format(message_type))

    def async_aggregation(self, message):
        (client_model, auxiliary_model), (self_step_num, other_step_num) = message.content
        # aggregation
        self_update_num = self_step_num - self.self_step_nums[message.origin]
        other_update_num = self.other_step_nums[message.origin] - other_step_num
        self.global_client_model = get_average_model(
            [client_model, self.global_client_model],
            [self_update_num / (self_update_num + other_update_num),
             other_update_num / (self_update_num + other_update_num)])
        if auxiliary_model:
            self.global_auxiliary_model = get_average_model(
                [auxiliary_model, self.global_auxiliary_model],
                [self_update_num / (self_update_num + other_update_num),
                 other_update_num / (self_update_num + other_update_num)])
        self.self_step_nums[message.origin] = self_step_num
        for client_index in range(self.config.CLIENT_NUM):
            if client_index != message.origin:
                self.other_step_nums[client_index] += self_update_num
        # send client params
        self.comm.send(Message(
            title="MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}".format(message.origin),
            content=[[self.global_client_model, self.global_auxiliary_model],
                     [self.self_step_nums[message.origin], self.other_step_nums[message.origin]],
                     self.lr_decay],
            msg_type='agg'), message.origin)

    def async_all_aggregation(self, messages):
        client_models = []
        auxiliary_models = []
        self_update_nums = []
        other_update_nums = []
        for message in messages:
            (client_model, auxiliary_model), (self_step_num, other_step_num) = message.content
            client_models.append(client_model)
            if auxiliary_model:
                auxiliary_models.append(auxiliary_model)
            self_update_num = self_step_num - self.self_step_nums[message.origin]
            other_update_num = self.other_step_nums[message.origin] - other_step_num
            self_update_nums.append(self_update_num)
            other_update_nums.append(other_update_num)
            self.self_step_nums[message.origin] = self_step_num
            for client_index in range(self.config.CLIENT_NUM):
                if client_index != message.origin:
                    self.other_step_nums[client_index] += self_update_num
        weights = [self_update_num / (sum(self_update_nums) + sum(other_update_nums))
                   for self_update_num in self_update_nums]
        client_models.append(self.global_client_model)
        if auxiliary_models:
            auxiliary_models.append(self.global_auxiliary_model)
        weights.append(sum(other_update_nums) / (sum(self_update_nums) + sum(other_update_nums)))
        self.global_client_model = get_average_model(client_models, weights)
        if auxiliary_models:
            self.global_auxiliary_model = get_average_model(auxiliary_models, weights)
        # send client params
        for message in messages:
            self.comm.send(Message(
                title="MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}".format(message.origin),
                content=[[self.global_client_model, self.global_auxiliary_model],
                         [self.self_step_nums[message.origin], self.other_step_nums[message.origin]],
                         self.lr_decay],
                msg_type='agg'), message.origin)

    def async_aggregation0(self, message):
        client_model, auxiliary_model = message.content
        difference = self.global_client_version - self.client_versions[message.origin]
        alpha = 1 / (difference + 1)
        self.global_client_model = get_average_model(
            [self.global_client_model, client_model],
            [1 - alpha, alpha]
        )
        if auxiliary_model:
            self.global_auxiliary_model = get_average_model(
                [self.global_auxiliary_model, auxiliary_model],
                [1 - alpha, alpha])
        self.global_client_version += 1
        self.client_versions[message.origin] = self.global_client_version
        # send client params
        self.comm.send(Message(
            title="MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}".format(message.origin),
            content=[self.global_client_model, self.global_auxiliary_model],
            msg_type='agg'), message.origin)

    def sync_aggregation(self, buffered_messages):
        if len(buffered_messages) >= self.config.CLIENT_NUM:
            client_models = []
            auxiliary_models = []
            for message in buffered_messages:
                client_model, auxiliary_model = message.content
                client_models.append(client_model)
                auxiliary_models.append(auxiliary_model)
            self.global_client_model = get_average_model(client_models)
            self.global_auxiliary_model = get_average_model(auxiliary_models)
            buffered_messages.clear()
            for client_index in range(self.config.CLIENT_NUM):
                self.self_step_nums[client_index] += 1
                self.comm.send(Message(
                    title="MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}".format(client_index),
                    content=[self.global_client_model, self.global_auxiliary_model],
                    msg_type='agg'), client_index)

    def fedasync_aggregation(self, message):
        client_model, auxiliary_model = message.content
        # FedAsync aggregation
        alpha = 0.9
        a = 10
        b = 4
        difference = self.global_client_version - self.client_versions[message.origin]
        staleness = 1 if difference <= b else 1 / (a * (difference - b) + 1)
        alpha *= staleness
        self.global_client_model = get_average_model(
            [self.global_client_model, client_model],
            [1 - alpha, alpha]
        )
        if auxiliary_model:
            self.global_auxiliary_model = get_average_model(
                [self.global_auxiliary_model, auxiliary_model],
                [1 - alpha, alpha])
        self.global_client_version += 1
        self.client_versions[message.origin] = self.global_client_version
        # send client params
        self.comm.send(Message(
            title="MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}".format(message.origin),
            content=[self.global_client_model, self.global_auxiliary_model],
            msg_type='agg'), message.origin)

    def fedbuff_aggregation(self, buffered_messages):
        # FedBuff aggregation
        if len(buffered_messages) >= self.config.BUFFER_AGG:
            buffered_models = []
            client_indices = []
            for message in buffered_messages:
                if self.buffered_client_models[message.origin] is None:
                    previous_model = self.global_client_model
                else:
                    previous_model = self.buffered_client_models[message.origin]
                client_model = get_model_diff(previous_model, message.content)
                buffered_models.append(client_model)
                client_indices.append(message.origin)
            avg_model_diff = get_average_model(buffered_models)
            self.global_client_model = update_model(
                self.global_client_model, avg_model_diff, self.config.GLOBAL_LEARNING_RATE)
            buffered_messages.clear()
            for client_index in client_indices:
                self.buffered_client_models[client_index] = self.global_client_model
                self.comm.send(Message(
                    title="MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}".format(client_index),
                    content=self.global_client_model,
                    msg_type='agg'), client_index)

    def val(self, start_time, train_loss, train_acc, pbar=None):
        self.client_net.load_state_dict(self.global_client_model)
        self.client_net.to(self.device)
        self.client_net.eval()
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
            current_time = time.time() - start_time
            if pbar:
                postfix = {
                    "time": "{:.2f}".format(current_time),
                    "train_loss": "{:.2f}".format(train_loss),
                    "train_acc": "{:.2f}".format(train_acc),
                    "val_loss": "{:.2f}".format(val_loss),
                    "val_acc": "{:.2f}".format(val_acc)
                }
            if val_loss is None or val_loss > 100:
                val_loss = self.val_loss[-1] if self.val_loss else 0
                val_acc = self.val_acc[-1] if self.val_acc else 0
            self.train_loss.append(round(train_loss, 2))
            self.train_acc.append(round(train_acc, 2))
            self.val_loss.append(round(val_loss, 2))
            self.val_acc.append(round(val_acc, 2))
            self.time_line.append(round(current_time))
            if self.best_val_step is None or val_acc > self.val_acc[self.best_val_step]:
                self.best_val_step = self.val_step
                self.best_client_model = deepcopy(self.global_client_model)
                self.best_server_model = deepcopy(self.net.state_dict())
                self.converge_time = current_time
            if pbar:
                postfix.update({
                    "best_step": str(self.best_val_step),
                    "best_acc": 0 if self.best_val_step is None else self.val_acc[self.best_val_step]
                })
                pbar.set_postfix(postfix)
                pbar.update(1)
            if self.check_stop_condition():
                self.send_stop_signal()
                return True
            self.val_step += 1
        return False

    def test(self):
        self.client_net.load_state_dict(self.best_client_model)
        self.client_net.to(self.device)
        self.client_net.eval()
        self.net.load_state_dict(self.best_server_model)
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
                outputs = self.net(self.client_net(inputs))
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
            # Send test loss and acc to client.
            for client_index in range(self.config.CLIENT_NUM):
                self.comm.send(Message(
                    title='MSG_SERVER_LOSS_SERVER_TO_CLIENT_{}'.format(client_index),
                    msg_type='test',
                    content=self.test_acc
                ), client_index)


    def calculate_acc(self, outputs, targets):
        if self.config.TASK_TYPE == 'semantic_segmentation':
            total_intersection = [0 for _ in range(self.config.NUM_CLASS)]
            total_union = [0 for _ in range(self.config.NUM_CLASS)]
            predicted_masks = torch.argmax(outputs, dim=1)
            for i in range(self.config.NUM_CLASS):
                intersection = ((predicted_masks == i) & (targets == i)).sum().item()
                union = ((predicted_masks == i) | (targets == i)).sum().item()
                total_intersection[i] += intersection
                total_union[i] += union
            acc = statistics.mean(
                [100. * total_intersection[i] / total_union[i]
                 for i in range(1, self.config.NUM_CLASS) if total_union[i]])
        else:
            _, predicted = outputs.max(1)
            total = targets.size(0)
            correct = predicted.eq(targets).sum().item()
            acc = 100. * correct / total if total else 0
        return acc

    def profile(self):
        profiler = Profiler(self.config)
        self.server_profile_res = profiler.run()

    def receive(self, batch_size):
        self.connect.send(batch_size)
        return pickle.loads(self.connect.recv())

    def send_grad(self, grads, loss, activation_index):
        grads_list = torch.split(grads, [act[0] for act in activation_index])
        origin_list = [act[1] for act in activation_index]
        for i in range(len(activation_index)):
            self.comm.send(Message(
                title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(origin_list[i]),
                content=[grads_list[i].cpu(), loss],
                msg_type="grad",
            ), origin_list[i])

    def send_logit(self, logits, activation_index):
        logit_list = torch.split(logits, [act[0] for act in activation_index])
        origin_list = [act[1] for act in activation_index]
        for i in range(len(activation_index)):
            client_logits = logit_list[i].cpu().detach().requires_grad_(True)
            self.comm.send(Message(
                title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(origin_list[i]),
                content=client_logits,
                msg_type="logit",
            ), origin_list[i])

    # def receive_logit(self):
    #     self.connect.send('logit')
    #     return pickle.loads(self.connect.recv())

    def check_stop_condition(self):
        if self.val_step + 1 >= self.config.MAX_VAL_STEP:
            return True
        if self.val_step - self.best_val_step >= 5 and self.config.LEARNING_RATE_DECAY:
            self.lr_decay = True
            for param_group in self.optimiser.param_groups:
                param_group['lr'] = self.config.LEARNING_RATE * 0.1
            self.config.LEARNING_RATE_DECAY = False
            return False
        if self.config.EARLY_STOP and self.val_step - self.best_val_step >= self.config.NON_IMPROVE_STEP:
            return True
        return False

    def send_stop_signal(self):
        self.global_client_model = self.best_client_model
        for client_index in range(self.config.CLIENT_NUM):
            self.comm.send(Message(
                title='MSG_SERVER_STOP_SIGNAL_TO_CLIENT_{}'.format(client_index),
                content=self.global_client_model if self.sync_final_model else None,
                msg_type="end"
            ), client_index)
        self.connect.send(-1)
