import socket
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, config=None):
        self.config = config
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        # Deploy the neural network.
        self.net = None
        self.optimiser = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.communication_amount = {}
