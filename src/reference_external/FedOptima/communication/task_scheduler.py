import gc
import logging
import pickle
import torch
import sys

from copy import deepcopy
from queue import Queue
from threading import Lock
from time import sleep
from communication.message import Message

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaskScheduler:
    def __init__(self, client_num, connection, comm, data_size, algorithm='counter', act_confirm=False, act_check=False,
                 act_buffer=1, timing=None, async_agg=True, agg_all_models=False, act_parallel_num=1):
        self.client_num = client_num
        self.connection = connection
        self.comm = comm
        self.data_size = data_size
        self.algorithm = algorithm
        self.act_confirm = act_confirm
        self.act_check = act_check
        self.counter = [0 for _ in range(client_num)]
        self.client_act_queues = [None for _ in range(client_num)]
        self.client_label_queues = [None for _ in range(client_num)]
        self.client_act_list = []
        self.activations_index = []
        self.act_avail = [True for _ in range(client_num)]
        self.act_buffer = act_buffer
        self.client_models = Queue()
        self.val_data = Queue()
        self.general_messages = [Queue() for _ in range(client_num)]
        self.logits = Queue()
        self.label_on_device = False
        self.timing = timing
        self.async_agg = async_agg
        self.agg_all_models = agg_all_models
        self.act_parallel_num = act_parallel_num
        self.block_train = False
        self.lock = Lock()

    def put(self):
        close_comm = 0
        while True:
            # gc.collect()
            message = self.comm.receive()
            if message.close_comm:
                close_comm += 1
                if close_comm == self.client_num:
                    break
            elif message.msg_type in ['train', 'val', 'agg', 'logit', 'general']:
                self.put_data(message.msg_type, message)
            else:
                raise Exception("Message type {} not recognised.".format(message.msg_type))
        logger.info("Task scheduler putter closed.")

    def get(self):
        while True:
            # gc.collect()
            batch_size = self.connection.recv()
            # Stop signal
            if batch_size == -1:
                break
            message_type = "wait"
            content = None
            while message_type == "wait":
                message_type, content = self.get_data(batch_size)
                if message_type == "wait":
                    sleep(0.001)
            self.connection.send(pickle.dumps([message_type, content]))
        logger.info("Task scheduler getter closed.")

    def put_data(self, message_type, message=None):
        self.lock.acquire()
        if message_type == 'agg':
            self.put_client_models(message)
        elif message_type == 'val':
            self.put_val_data(message)
        elif message_type == 'train':
            if self.act_confirm:
                self.comm.send(Message(
                    title="ACTIVATIONS_CONFIRM_TO_CLIENT_{}".format(message.origin)),
                    message.origin)
            self.put_activation(message)
        elif message_type == 'logit':
            self.logits.put(message)
        elif message_type == 'general':
            self.general_messages[message.origin].put(message)
        else:
            raise Exception("Message type {} not recognised.".format(message_type))
        self.lock.release()

    def get_data(self, batch_size):
        self.lock.acquire()
        if not self.client_models.empty():
            # get client models
            if not self.async_agg:
                message = self.get_activation(batch_size)
                if message != -1:
                    self.lock.release()
                    return 'train', message
            message_type = 'agg'
            message = self.get_client_models()
            # print("Get model")
        elif not self.logits.empty():
            self.block_train = True
            message_type = 'logit'
            message = self.logits.get()
            self.comm.act_parallel_num -= 1
            if self.comm.act_parallel_num == 0:
                self.block_train = False
        elif not self.val_data.empty():
            # get validation data
            if not self.async_agg:
                message = self.get_activation(batch_size)
                if message != -1:
                    self.lock.release()
                    return 'train', message
            message_type = 'val'
            message = self.get_val_data()
        else:
            # get training activations
            message_type = 'train'
            message = self.get_activation(batch_size)
            if message == -1:
                message_type = 'wait'
        self.lock.release()
        return message_type, message

    def get_activation(self, batch_size):
        if self.block_train or self.comm.act_parallel_num >= self.act_parallel_num:
            return -1
        if self.algorithm == 'fifo':
            activations = self.get_activation_fifo(batch_size)
            if activations != -1 and self.act_check:
                self.check_activation_fifo(batch_size)
        elif self.algorithm == 'counter':
            activations = self.get_activation_counter(batch_size)
            if activations != -1 and self.act_check:
                self.check_activation_counter(batch_size)
        elif self.algorithm == 'list':
            activations = self.get_activation_list()
            if activations != -1 and self.act_check:
                self.check_activation_list()
        else:
            raise "Task scheduler algorithm {} not implemented.".format(self.algorithm)
        return activations

    def put_activation(self, message):
        if self.timing:
            upload_timing = self.timing.break_down('upload', 0, 0)
            upload_timing.set_start_time(message.send_time)
            upload_timing.set_stop_time(message.receive_time)
        if self.algorithm == 'fifo':
            self.put_activation_fifo(message)
        elif self.algorithm == 'counter':
            self.put_activation_by_counter(message)
        elif self.algorithm == 'list':
            self.put_activation_list(message)
        else:
            raise "Task scheduler algorithm {} not implemented.".format(self.algorithm)

    def put_activation_fifo(self, message):
        activations, labels = message.content
        origin = message.origin
        self.activations_index.append([activations.size(dim=0), origin])
        if self.client_act_queues[0] is None:
            self.client_act_queues[0] = activations
        else:
            self.client_act_queues[0] = torch.cat([self.client_act_queues[0], activations])
        if self.client_label_queues[0] is None:
            self.client_label_queues[0] = labels
        else:
            self.client_label_queues[0] = torch.cat([self.client_label_queues[0], labels])

    def get_activation_fifo(self, batch_size):
        if self.client_act_queues[0] is None or self.client_label_queues[0] is None:
            return -1
        act_len = self.client_act_queues[0].size(dim=0)
        if batch_size > act_len:
            return -1
        activations_list = []
        labels_list = []
        activations_index = []
        for i in range(len(self.activations_index)):
            client_activation_len, client_index = self.activations_index[i]
            if client_activation_len > batch_size:
                activations_index.append([batch_size, client_index])
                self.activations_index[i][0] = client_activation_len - batch_size
                activations, self.client_act_queues[0] = torch.split(
                    self.client_act_queues[0],
                    [batch_size, self.client_act_queues[0].size(dim=0) - batch_size])
                activations_list.append(activations)
                labels, self.client_label_queues[0] = torch.split(
                    self.client_label_queues[0],
                    [batch_size, self.client_label_queues[0].size(dim=0) - batch_size])
                labels_list.append(labels)
                self.activations_index = self.activations_index[i:]
                break
            else:
                activations_index.append([client_activation_len, client_index])
                remaining_len = self.client_act_queues[0].size(dim=0) - client_activation_len
                if remaining_len > 0:
                    activations, self.client_act_queues[0] = torch.split(
                        self.client_act_queues[0],
                        [client_activation_len, remaining_len])
                    labels, self.client_label_queues[0] = torch.split(
                        self.client_label_queues[0],
                        [client_activation_len, remaining_len])
                else:
                    activations, self.client_act_queues[0] = self.client_act_queues[0], None
                    labels, self.client_label_queues[0] = self.client_label_queues[0], None
                activations_list.append(activations)
                labels_list.append(labels)
                batch_size -= client_activation_len
                if batch_size == 0:
                    self.activations_index = self.activations_index[i+1:]
                    break
        return torch.cat(activations_list), torch.cat(labels_list), activations_index

    def put_activation_by_counter(self, message):
        activations, labels = message.content
        if self.client_act_queues[message.origin] is None:
            self.client_act_queues[message.origin] = activations
        else:
            self.client_act_queues[message.origin] = torch.cat(
                [self.client_act_queues[message.origin], activations])
        if self.client_label_queues[message.origin] is None:
            self.client_label_queues[message.origin] = labels
        else:
            self.client_label_queues[message.origin] = torch.cat(
                [self.client_label_queues[message.origin], labels])
        self.counter[message.origin] += activations.size(dim=0)

    def get_activation_counter(self, batch_size):
        # if there are enough activations buffered
        act_len = sum([client_act.size(dim=0) for client_act in self.client_act_queues if client_act is not None])
        if batch_size > act_len:
            return -1
        activation_len = 0
        activations_list = []
        labels_list = []
        activations_index = []
        learning_rate_decay = None
        while activation_len < batch_size:
            client_index = self.get_client_index_by_counter()
            client_activation_len = min(batch_size - activation_len, self.client_act_queues[client_index].size(dim=0))
            remaining_len = self.client_act_queues[client_index].size(dim=0) - client_activation_len
            if remaining_len > 0:
                activations, self.client_act_queues[client_index] = torch.split(
                    self.client_act_queues[client_index], [client_activation_len, remaining_len])
                labels, self.client_label_queues[client_index] = torch.split(
                    self.client_label_queues[client_index], [client_activation_len, remaining_len])
            else:
                activations = self.client_act_queues[client_index]
                labels = self.client_label_queues[client_index]
                self.client_act_queues[client_index] = None
                self.client_label_queues[client_index] = None
            activation_len += client_activation_len
            activations_list.append(activations)
            labels_list.append(labels)
            activations_index.append([client_activation_len, client_index])
        return torch.cat(activations_list), torch.cat(labels_list), activations_index

    def get_client_index_by_counter(self):
        # what if self.client_act_queues[client_index] is None
        proxy_counter = deepcopy(self.counter)
        while True:
            client_index = proxy_counter.index(min(proxy_counter))
            if self.client_act_queues[client_index] is not None and self.client_label_queues[client_index] is not None:
                break
            proxy_counter[client_index] = sys.maxsize
        return client_index

    def put_activation_list(self, message):
        self.client_act_list.append(message)

    def get_activation_list(self):
        if not self.client_act_list:
            return -1
        message = self.client_act_list.pop(0)
        activations, labels = message.content
        origin = message.origin
        activation_index = [[activations.size(dim=0), origin]]
        return activations, labels, activation_index

    def check_activation_fifo(self, batch_size):
        if (self.client_act_queues[0] is not None
                and self.client_act_queues[0].size(dim=0) >= batch_size * self.act_buffer):
            return
        for client_index in range(self.client_num):
            self.comm.send(Message(
                msg_type='act_check',
                title="ACTIVATIONS_CONFIRM_TO_CLIENT_{}".format(client_index),
                content=True),
                client_index)

    def check_activation_counter(self, batch_size):
        for client_index in range(self.client_num):
            if (self.client_act_queues[client_index] is not None
                    and self.client_act_queues[client_index].size(dim=0) >= batch_size * self.act_buffer):
                continue
            self.comm.send(Message(
                msg_type='act_check',
                title="ACTIVATIONS_CONFIRM_TO_CLIENT_{}".format(client_index),
                content=True),
                client_index)

    def check_activation_list(self):
        if len(self.client_act_list) < self.act_buffer:
            for client_index in range(self.client_num):
                self.comm.send(Message(
                    msg_type='act_check',
                    title="ACTIVATIONS_CONFIRM_TO_CLIENT_{}".format(client_index),
                    content=True),
                    client_index)

    def put_client_models(self, message):
        self.client_models.put(message)

    def get_client_models(self):
        if self.agg_all_models:
            models = []
            while not self.client_models.empty():
                models.append(self.client_models.get())
            return models
        else:
            return self.client_models.get()

    def put_val_data(self, message):
        self.val_data.put(message)

    def get_val_data(self):
        return self.val_data.get()
