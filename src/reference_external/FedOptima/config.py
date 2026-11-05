import os
import platform
import sys
import torch
from data_preprocessing.datasets import IMDB, SST2
from data_preprocessing.nlp_tools import build_vocab_from_dataset
from networks.model_config import ModelConfig


class Config:
    def __init__(self,
                 experiment_name="experiment",
                 task_type='image_classification',
                 server_address=None,
                 port=51000,
                 client_num=1,
                 asynchronisation=True,
                 max_val_step=100,
                 non_improve_step=10,
                 layer_num_on_client=-1,
                 label_on_device=True,
                 same_data_size_for_each_client=True,
                 client_index=-1,
                 simulated_client_num=1,
                 epoch_num=100000,
                 aggregation_frequency=-1,
                 val_frequency=1,
                 val_rate=1,
                 learning_rate=0.01,
                 global_learning_rate=1,
                 optim='sgd',
                 clip_grad_norm=None,
                 batch_size=100,
                 server_cpu_num=-1,
                 client_cpu_num=-1,
                 is_simulated=False,
                 simulate_bandwidth=True,
                 data_size=None,
                 test_data_size=None,
                 model_name='VGG5',
                 dataset_name='CIFAR-10',
                 dataset_split='balance_iid',
                 early_stop=True,
                 enable_val=True,
                 enable_test=True,
                 enable_profiler=False,
                 enable_optimiser=False,
                 pipe_batch_num=1,
                 async_comm=True,
                 duplicate_pipe_batch=False,
                 immediate_update=False,
                 profile_iter_num=-1,
                 profile_num_on_client=-1,
                 sort_algo='counter',
                 auxiliary='auxiliary_net',
                 act_buffer=1,
                 act_parallel_num=1,
                 buffer_agg=-1,
                 async_agg=True,
                 agg_all_models=False,
                 learning_rate_decay=False,
                 weight_decay=0,
                 uplink_bandwidth=None,
                 downlink_bandwidth=None,
                 packet_loss_rate=0,
                 dynamic_bandwidth=(),
                 bandwidth_change_interval=600,
                 device_drop_rate=0):
        # Framework info.
        self.EXPERIMENT_NAME = experiment_name
        self.TASK_TYPE = task_type
        self.SERVER_ADDRESS = server_address if server_address else '127.0.0.1'
        self.SERVER_PORT = port
        self.CLIENT_NUM = client_num
        self.ASYNCHRONISATION = asynchronisation
        self.MAX_VAL_STEP = max_val_step
        self.NON_IMPROVE_STEP = non_improve_step
        self.EARLY_STOP = early_stop
        self.ACT_BUFFER = act_buffer
        self.ACT_PARALLEL_NUM = act_parallel_num
        self.BUFFER_AGG = buffer_agg  # 1: FedAsync; >1: FedBuff
        self.ASYNC_AGG = async_agg  # True: Ours; False: CSE-FSL
        self.AGG_ALL_MODELS = agg_all_models  # True: Ours; False: CSE-FSL
        self.LEARNING_RATE_DECAY = learning_rate_decay
        self.WEIGHT_DECAY = weight_decay
        self.SAME_DATA_SIZE_FOR_EACH_CLIENT = same_data_size_for_each_client
        self.CLIENT_INDEX = client_index
        self.SIMULATED_CLIENT_NUM = simulated_client_num
        self.CONNECTION_TIME_LIMIT = 300
        self.GPU_NUM = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if server_cpu_num == -1 or server_cpu_num > os.cpu_count():
            server_cpu_num = os.cpu_count()
        self.SERVER_CPU_NUM = server_cpu_num
        self.CLIENT_CPU_NUM = client_cpu_num
        self.CLIENT_CPU_NUM_LIST = []
        self.IS_SIMULATED = is_simulated
        self.SIMULATE_BANDWIDTH = simulate_bandwidth
        self.ENABLE_OPTIMISER = enable_optimiser
        # Training info.
        self.DATA_LENGTH = data_size
        self.TEST_DATA_LENGTH = test_data_size
        self.BATCH_SIZE = batch_size
        self.DATASET_SPLIT = dataset_split
        self.EPOCH_NUM = epoch_num
        self.AGGREGATION_FREQ = aggregation_frequency
        self.VAL_FREQ = val_frequency
        self.LEARNING_RATE = learning_rate
        self.GLOBAL_LEARNING_RATE = global_learning_rate
        self.OPTIM = optim
        self.CLIP_GRAD_NORM = clip_grad_norm
        self.CLIENT_DATA_LENGTH = None
        self.PIPE_BATCH_NUM = pipe_batch_num if layer_num_on_client not in [0, -1] else 1
        if duplicate_pipe_batch:
            self.BATCH_SIZE *= self.PIPE_BATCH_NUM
        self.ASYNC_COMM = async_comm
        self.IMMEDIATE_UPDATE = immediate_update
        self.TIME_DIFFERENCE = 0.
        self.SORT_ALGO = sort_algo
        # Dataset info.
        self.DATASET_PATH = os.path.join(os.getcwd(), 'data')
        self.DATASET_NAME = dataset_name
        self.INPUT_CHANNEL_NUM = 0
        self.INPUT_IMAGE_SIZE = 0
        self.DATASET = None
        self.VOCAB = None
        self.NUM_CLASS = 0
        self.DATASET_MEAN = None
        self.DATASET_STD = None
        if self.DATASET_NAME == 'CIFAR-10':
            self.INPUT_IMAGE_SIZE = 32
            self.NUM_CLASS = 10
            self.INPUT_CHANNEL_NUM = 3
            self.DATASET_MEAN = (0.4914, 0.4822, 0.4465)
            self.DATASET_STD = (0.2023, 0.1994, 0.2010)
            if self.DATA_LENGTH is None:
                self.DATA_LENGTH = 50000 // self.CLIENT_NUM
            if self.TEST_DATA_LENGTH is None:
                self.TEST_DATA_LENGTH = 10000
        elif self.DATASET_NAME == 'CIFAR-100':
            self.INPUT_IMAGE_SIZE = 32
            self.NUM_CLASS = 100
            self.INPUT_CHANNEL_NUM = 3
            self.DATASET_MEAN = (0.4914, 0.4822, 0.4465)
            self.DATASET_STD = (0.2023, 0.1994, 0.2010)
            if self.DATA_LENGTH is None:
                self.DATA_LENGTH = 50000 // self.CLIENT_NUM
            if self.TEST_DATA_LENGTH is None:
                self.TEST_DATA_LENGTH = 10000
        elif self.DATASET_NAME == 'ImageNet':
            self.INPUT_IMAGE_SIZE = 224
            self.NUM_CLASS = 1000
            self.INPUT_CHANNEL_NUM = 3
            self.DATASET_MEAN = (0.485, 0.456, 0.406)
            self.DATASET_STD = (0.229, 0.224, 0.225)
        elif self.DATASET_NAME == 'MNIST':
            self.INPUT_IMAGE_SIZE = 28
            self.NUM_CLASS = 10
            self.INPUT_CHANNEL_NUM = 1
            self.DATASET_MEAN = (0.1307,)
            self.DATASET_STD = (0.1307,)
            if self.DATA_LENGTH is None:
                self.DATA_LENGTH = 60000 // self.CLIENT_NUM
            if self.TEST_DATA_LENGTH is None:
                self.TEST_DATA_LENGTH = 10000
        elif self.DATASET_NAME == 'SVHN':
            self.INPUT_IMAGE_SIZE = 32
            self.NUM_CLASS = 10
            self.INPUT_CHANNEL_NUM = 3
            self.DATASET_MEAN = (0.485, 0.456, 0.406)
            self.DATASET_STD = (0.229, 0.224, 0.225)
            if self.DATA_LENGTH is None:
                self.DATA_LENGTH = 73257 // self.CLIENT_NUM
            if self.TEST_DATA_LENGTH is None:
                self.TEST_DATA_LENGTH = 26032
        elif self.DATASET_NAME == 'COCO':
            self.INPUT_IMAGE_SIZE = 224
            self.NUM_CLASS = 80
            self.INPUT_CHANNEL_NUM = 3
            self.DATASET_MEAN = (0.471, 0.448, 0.408)
            self.DATASET_STD = (0.234, 0.239, 0.242)
        elif self.DATASET_NAME == 'tiny-imagenet-200':
            self.INPUT_IMAGE_SIZE = 64
            # self.INPUT_IMAGE_SIZE = 32
            self.NUM_CLASS = 200
            self.INPUT_CHANNEL_NUM = 3
            self.DATASET_MEAN = (0.485, 0.456, 0.406)
            self.DATASET_STD = (0.229, 0.224, 0.225)
            if self.DATA_LENGTH is None:
                self.DATA_LENGTH = 100000 // self.CLIENT_NUM
            if self.TEST_DATA_LENGTH is None:
                self.TEST_DATA_LENGTH = 10000
        elif self.DATASET_NAME == 'VOC':
            self.INPUT_IMAGE_SIZE = 224
            self.NUM_CLASS = 21
            self.INPUT_CHANNEL_NUM = 3
            self.DATASET_MEAN = (0.485, 0.456, 0.406)
            self.DATASET_STD = (0.229, 0.224, 0.225)
            if self.DATA_LENGTH is None:
                self.DATA_LENGTH = 1464 // self.CLIENT_NUM
            if self.TEST_DATA_LENGTH is None:
                self.TEST_DATA_LENGTH = 1449
        elif self.DATASET_NAME == 'Oxford-IIIT':
            self.INPUT_IMAGE_SIZE = 224
            self.NUM_CLASS = 2
            self.INPUT_CHANNEL_NUM = 3
            self.DATASET_MEAN = (0.485, 0.456, 0.406)
            self.DATASET_STD = (0.229, 0.224, 0.225)
            if self.DATA_LENGTH is None:
                self.DATA_LENGTH = 3680 // self.CLIENT_NUM
            if self.TEST_DATA_LENGTH is None:
                self.TEST_DATA_LENGTH = 3669
        elif self.DATASET_NAME == 'SST2':
            self.NUM_CLASS = 2
            if self.DATA_LENGTH is None:
                # self.DATA_LENGTH = 67349 // self.CLIENT_NUM
                self.DATA_LENGTH = 48000 // self.CLIENT_NUM
            if self.TEST_DATA_LENGTH is None:
                # self.TEST_DATA_LENGTH = 872
                self.TEST_DATA_LENGTH = 19349
        elif self.DATASET_NAME == 'IMDB':
            self.NUM_CLASS = 2
            if self.DATA_LENGTH is None:
                self.DATA_LENGTH = 25000 // self.CLIENT_NUM
                # self.DATA_LENGTH = 24832 // self.CLIENT_NUM
            if self.TEST_DATA_LENGTH is None:
                self.TEST_DATA_LENGTH = 25000
        else:
            raise Exception("Do not support dataset {}.".format(self.DATASET_NAME))
        # Model info.
        self.MODEL_NAME = model_name
        self.MODEL_DIR = os.path.join(os.getcwd(), 'models')
        self.AUXILIARY = auxiliary  # 'auxiliary_net' or 'auxiliary_classifier' or None
        self.MODEL_CONFIG = None
        self.NEURAL_NETWORK = None
        self.AUXILIARY_NETWORK = None
        self.LAYER_NUM = None
        if layer_num_on_client is None:
            self.LAYER_NUM_ON_CLIENT = {
                'VGG5': 1,
                'VGG8': 2,
                'VGG11': 1,
                'VGG16': 2,
                'VGG19': 2,
                'ResNet18': 1,
                'ResNet34': 1,
                'ResNet50': 1,
                'ResNet101': 1,
                'ResNet152': 1,
                'MobileNetSmall': 1,
                'MobileNetLarge': 1,
            }[self.MODEL_NAME]
        else:
            self.LAYER_NUM_ON_CLIENT = layer_num_on_client
        self.LABEL_ON_DEVICE = label_on_device
        # Result file info.
        self.RESULT_DIR = os.path.join(os.getcwd(), 'results')
        self.RESULT_SYNC_CSV_ADDRESS = os.path.join(self.RESULT_DIR, 'sync_results.csv')
        self.RESULT_ASYNC_CSV_ADDRESS = os.path.join(self.RESULT_DIR, 'results.csv')
        self.FIG_DIR = os.path.join(self.RESULT_DIR, 'figs')
        # Functions.
        self.ENABLE_VAL = enable_val
        self.ENABLE_TEST = enable_test and val_rate < 1
        self.VAL_RATE = val_rate
        self.ENABLE_PROFILER = enable_profiler
        self.PROFILE_DIR = os.path.join(self.RESULT_DIR, 'profiling')
        self.MODEL_SAVE = False
        # Profiler
        self.PROFILE_ITER_NUM = profile_iter_num
        self.PROFILE_NUM_ON_CLIENT = profile_num_on_client
        self.UPLINK_BANDWIDTH = uplink_bandwidth
        self.DOWNLINK_BANDWIDTH = downlink_bandwidth
        self.PACKET_LOSS_RATE = packet_loss_rate
        self.DYNAMIC_BANDWIDTH = dynamic_bandwidth
        self.BANDWIDTH_CHANGE_INTERVAL = bandwidth_change_interval
        self.DEVICE_DROP_RATE = device_drop_rate

    def get_model_info(self):
        self.MODEL_CONFIG = ModelConfig(self.INPUT_CHANNEL_NUM, self.INPUT_IMAGE_SIZE, self.VOCAB, self.NUM_CLASS)
        self.NEURAL_NETWORK = self.MODEL_CONFIG.get_model_config(self.MODEL_NAME)
        if self.AUXILIARY == 'auxiliary_net':
            self.AUXILIARY_NETWORK = self.MODEL_CONFIG.get_auxiliary_network(self.MODEL_NAME)
        elif self.AUXILIARY == 'auxiliary_classifier':
            self.AUXILIARY_NETWORK = self.MODEL_CONFIG.get_auxiliary_classifier(self.MODEL_NAME)
        else:
            self.AUXILIARY_NETWORK = None
        self.LAYER_NUM = len(self.NEURAL_NETWORK)

    def get_vocab(self):
        dataset_root = os.path.join(self.DATASET_PATH, self.DATASET_NAME)
        if self.DATASET_NAME == 'SST2':
            self.DATASET = SST2(root=os.path.join(self.DATASET_PATH, self.DATASET_NAME), split='train')
        elif self.DATASET_NAME == 'IMDB':
            self.DATASET = IMDB(root=os.path.join(self.DATASET_PATH, self.DATASET_NAME), split='train')
        if self.DATASET:
            vocab_file = os.path.join(dataset_root, self.DATASET_NAME + '_vocab')
            self.VOCAB = build_vocab_from_dataset(self.DATASET, vocab_file, max_len=50000)

    def server_init(self):
        if self.SAME_DATA_SIZE_FOR_EACH_CLIENT:
            self.DATA_LENGTH *= self.CLIENT_NUM
        self.CLIENT_DATA_LENGTH = self.DATA_LENGTH // (self.CLIENT_NUM * self.SIMULATED_CLIENT_NUM)
        if self.AGGREGATION_FREQ is None or self.AGGREGATION_FREQ == -1:
            # if AGGREGATION_FREQ is not set, aggregate once each epoch
            self.AGGREGATION_FREQ = self.CLIENT_DATA_LENGTH // self.BATCH_SIZE
        elif self.AGGREGATION_FREQ == 0:
            # if AGGREGATION_FREQ is 0, no aggregation happens.
            self.AGGREGATION_FREQ = sys.maxsize
        if self.ASYNCHRONISATION:
            self.ASYNC_COMM = True
            self.EARLY_STOP = True
            self.ENABLE_VAL = True
        if self.EARLY_STOP:
            self.ENABLE_VAL = True

    def client_init(self):
        if isinstance(self.CLIENT_CPU_NUM, list):
            self.CLIENT_CPU_NUM_LIST = self.CLIENT_CPU_NUM
            self.CLIENT_CPU_NUM = self.CLIENT_CPU_NUM[self.CLIENT_INDEX]
            self.CLIENT_CPU_NUM = min(os.cpu_count(), self.CLIENT_CPU_NUM)
        else:
            if self.CLIENT_CPU_NUM == -1 or self.CLIENT_CPU_NUM > os.cpu_count():
                self.CLIENT_CPU_NUM = os.cpu_count()
            self.CLIENT_CPU_NUM_LIST = [self.CLIENT_CPU_NUM for _ in range(self.CLIENT_NUM)]
        if isinstance(self.PIPE_BATCH_NUM, list):
            self.PIPE_BATCH_NUM = self.PIPE_BATCH_NUM[self.CLIENT_INDEX]

    def update(self, other, exceptions=None):
        if exceptions is None:
            exceptions = []
        for var_name, var_value in other.__dict__.items():
            if var_name in exceptions:
                continue
            if getattr(self, var_name) != var_value:
                setattr(self, var_name, var_value)
