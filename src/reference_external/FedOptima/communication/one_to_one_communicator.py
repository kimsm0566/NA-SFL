import logging
import torch

from queue import Queue
from time import sleep
from threading import Lock
from communication.message import Message


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OneToOneCommunicator:
    def __init__(self, socket, time_difference=0., config=None, timing=None):
        self.socket = socket
        self.time_difference = time_difference
        self.config = config
        self.send_queue = Queue()
        self.receive_queue = Queue()
        self.grad = None
        self.loss = 0.
        self.model_queue = Queue(1)
        self.test_queue = Queue()
        self.target_queue = Queue()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.act_avail = True
        self.stop_signal = False
        self.final_client_model = None
        self.amount = 0.
        self.logit_amount = 0.
        self.timing = timing
        self.lock = Lock()

    def run_sender(self):
        while True:
            message = self.send_queue.get()
            amount = message.send_to(self.socket)
            self.amount += amount
            if message.msg_type == 'logit':
                self.logit_amount += amount
            if message.close_comm:
                logging.info("Output communication ends.")
                break

    def run_receiver(self):
        while True:
            message = Message(from_socket=self.socket, time_difference=self.time_difference, config=self.config)
            if message.msg_type == 'grad':
                self.put_grad(message)
            elif message.msg_type == 'agg':
                self.model_queue.put(message)
            elif message.msg_type == 'logit':
                self.compute_loss(message.content)
            elif message.msg_type == 'end':
                self.stop_signal = True
                self.final_client_model = message.content
            elif message.msg_type == 'test':
                self.test_queue.put(message)
            elif message.msg_type == 'act_check':
                # print('turn on act')
                self.act_avail = message.content
            else:
                self.receive_queue.put(message)
            if message.close_comm:
                logging.info("Input communication ends.")
                break

    def send(self, message):
        if self.config.LABEL_ON_DEVICE and message.msg_type == 'train':
            self.target_queue.put(message.content[1])
        self.send_queue.put(message)

    def receive(self, msg_type=None, batch_size=None, block=True):
        # block=False only applies on msg_type of aggregation
        if self.stop_signal:
            self.stop_signal = False
            return -1
        if msg_type == 'grad':
            grad = self.get_grad(batch_size)
            if grad is None:
                self.stop_signal = False
                return -1
            return grad, self.loss
        elif msg_type == 'agg':
            if not block and self.model_queue.empty():
                return None
            while self.model_queue.empty():
                if self.stop_signal:
                    self.stop_signal = False
                    return -1
                sleep(0.001)
            return self.model_queue.get()
        elif msg_type == 'end':
            if not block:
                if self.stop_signal:
                    return -1
                else:
                    return 1
            else:
                raise Exception('Message type "end" only work with non blocking.')
        elif msg_type == 'test':
            return self.test_queue.get()
        else:
            if not block and self.receive_queue.empty():
                return None
            while self.receive_queue.empty():
                if self.stop_signal:
                    self.stop_signal = False
                    return -1
                sleep(0.001)
            return self.receive_queue.get()

    def put_grad(self, message):
        self.lock.acquire()
        if self.timing:
            download_timing = self.timing.break_down('download', 0, 0)
            download_timing.set_start_time(message.send_time)
            download_timing.set_stop_time(message.receive_time)
        grad, self.loss = message.content
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = torch.cat([self.grad, grad])
        self.lock.release()

    def get_grad(self, batch_size):
        while self.grad is None or batch_size > self.grad.size(dim=0):
            if self.stop_signal:
                return None
            sleep(0.001)
        self.lock.acquire()
        if batch_size == self.grad.size(dim=0):
            grad, self.grad = self.grad, None
        else:
            grad, self.grad = torch.split(self.grad, [batch_size, self.grad.size(dim=0)-batch_size])
        self.lock.release()
        return grad

    def compute_loss(self, logits):
        targets = self.target_queue.get().to(self.device)
        logits = logits.to(self.device)
        logits.retain_grad()
        loss = self.criterion(logits, targets)
        loss.backward()
        self.send(Message(content=[logits.grad.cpu(), loss.item()], msg_type='logit', origin=self.config.CLIENT_INDEX))
