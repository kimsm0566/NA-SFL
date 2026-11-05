import logging

from threading import Thread
from torch.multiprocessing import Queue
from communication.message import Message


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OneToManyCommunicator:
    def __init__(self, sockets, time_differences=None):
        self.sockets = sockets
        self.time_differences = time_differences if time_differences else [0] * len(sockets)
        self.send_queue = Queue()
        self.receive_queue = Queue()
        self.client_num = len(sockets)
        self.amounts = [0.] * len(sockets)
        self.act_parallel_num = 0

    def run_receiver(self):
        def recv(client_index):
            while True:
                message = Message(from_socket=self.sockets[client_index],
                                  time_difference=self.time_differences[client_index])
                message.origin = client_index
                self.receive_queue.put(message)
                if message.close_comm:
                    logging.info("Input communication with Client {} ends.".format(client_index))
                    break

        receiver_threads = [Thread(target=recv, args=(client_index,)) for client_index in range(len(self.sockets))]
        for receiver_thread in receiver_threads:
            receiver_thread.start()
        for receiver_thread in receiver_threads:
            receiver_thread.join()
        logger.info("Receiver closed.")

    def run_sender(self):
        end_num = 0
        while True:
            message, client_index = self.send_queue.get()
            self.amounts[client_index] += message.send_to(self.sockets[client_index])
            if message.close_comm:
                end_num += 1
                if end_num == self.client_num:
                    break
        logger.info("Sender closed.")

    def send(self, message, client_index):
        if message.msg_type == 'logit':
            self.act_parallel_num += 1
        self.send_queue.put([message, client_index])

    def receive(self, expected_title=None):
        message = self.receive_queue.get()
        if expected_title is not None and message.title != expected_title:
            raise Exception("Expected " + expected_title + " but received " + message.title)
        return message
