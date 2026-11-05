import logging
import pickle
import struct
import time
import random

from communication.bandwidth_limiter import BandwidthLimiter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Message(BandwidthLimiter):
    def __init__(self,
                 title=None,
                 content=None,
                 msg_type="general",  # general, train, val, test, agg
                 origin=None,
                 end=False,
                 close_comm=False,
                 from_socket=None,
                 expected_title=None,
                 time_difference=0.,
                 config=None):
        if config and config.SIMULATE_BANDWIDTH:
            if config.UPLINK_BANDWIDTH == 0 or config.DYNAMIC_BANDWIDTH == 0:
                upload_speed_bytes_per_s, download_speed_bytes_per_s = None, None
                if not config.ASYNCHRONISATION or config.LAYER_NUM_ON_CLIENT == -1 or msg_type == 'train':
                    time.sleep(config.BANDWIDTH_CHANGE_INTERVAL)
            else:
                upload_speed_bytes_per_s = config.UPLINK_BANDWIDTH * 1024**2 / 8 if config.UPLINK_BANDWIDTH else None
                download_speed_bytes_per_s = config.DOWNLINK_BANDWIDTH * 1024**2 / 8 if config.DOWNLINK_BANDWIDTH else None
        else:
            upload_speed_bytes_per_s, download_speed_bytes_per_s = None, None
        super().__init__(upload_speed_bytes_per_s, download_speed_bytes_per_s)
        self.title = title
        self.content = content
        self.msg_type = msg_type
        self.origin = origin
        self.end = end
        self.close_comm = close_comm
        self.send_time = None
        self.receive_time = None
        self.elapsed_time = None
        self.loss_packet_rate = config.PACKET_LOSS_RATE if config else 0
        if from_socket:
            self.receive_from(from_socket, expected_title, time_difference)

    def send_to(self, sock):
        if self.loss_packet_rate > 0 and random.random() < self.loss_packet_rate:
            self.send_time = time.time()
            message = pickle.dumps([self.title, self.content, self.msg_type, self.origin, self.end,
                                    self.close_comm, self.send_time, False])
            self.sendall(sock, struct.pack(">I", len(message)))
            self.sendall(sock, message)
        self.send_time = time.time()
        message = pickle.dumps([self.title, self.content, self.msg_type, self.origin, self.end,
                                self.close_comm, self.send_time, True])
        self.sendall(sock, struct.pack(">I", len(message)))
        self.sendall(sock, message)
        logger.debug(str(self.title) + 'sent to' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))
        return len(message)

    def receive_from(self, sock, expected_title=None, time_difference=0.):
        message_len = struct.unpack(">I", self.recv(sock, 4))[0]
        self.title, self.content, self.msg_type, self.origin, self.end, \
            self.close_comm, self.send_time, real_packet = pickle.loads(self.recv(sock, message_len))
        if not real_packet:
            message_len = struct.unpack(">I", self.recv(sock, 4))[0]
            self.title, self.content, self.msg_type, self.origin, self.end, \
                self.close_comm, self.send_time, real_packet = pickle.loads(self.recv(sock, message_len))
        self.send_time -= time_difference
        self.receive_time = time.time()
        self.elapsed_time = self.receive_time - self.send_time
        logger.debug(str(self.title) + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))
        if expected_title is not None and self.title != expected_title:
            raise Exception("Expected " + expected_title + " but received " + self.title)

    def __str__(self):
        return "[Title] " + str(self.title) + " [Content] " + str(self.content) + " [End] " + str(self.end)
