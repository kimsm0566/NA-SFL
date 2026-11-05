import random
import numpy as np
import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import ImageFolder
from data_preprocessing.nlp_tools import build_vocab_from_dataset, nltk_tokenizer
from data_preprocessing.datasets import IMDB, SST2

MAX_SEQ_LENGTH = 256
RANDOM_SEED = 1994
random.seed(RANDOM_SEED)


def replace_tensor_value(tensor, replaced_values, replacing_value):
    for replaced_value in replaced_values:
        tensor[tensor == replaced_value] = replacing_value
    return tensor


def seed_worker():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


class DataReader:
    def __init__(self, config, set_name='train'):
        self.config = config
        self.dataset = None
        self.set_name = set_name
        self.torcht = False
        self._read()

    def _read(self):
        data_path = os.path.join(self.config.DATASET_PATH, self.config.DATASET_NAME, self.set_name)
        download = False
        if not os.path.exists(data_path) or not os.listdir(data_path):
            if self.set_name == 'train':
                if self.config.IS_SIMULATED:
                    download = True if self.config.CLIENT_INDEX == 0 else False
                else:
                    download = True
            else:
                download = True
        if self.set_name == 'train':
            if self.config.TASK_TYPE == 'semantic_segmentation':
                transform = transforms.Compose([
                    transforms.Resize((self.config.INPUT_IMAGE_SIZE, self.config.INPUT_IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.config.DATASET_MEAN, self.config.DATASET_STD)
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(self.config.INPUT_IMAGE_SIZE),
                    transforms.RandomCrop(self.config.INPUT_IMAGE_SIZE, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.config.DATASET_MEAN, self.config.DATASET_STD)
                ])
        elif self.set_name == 'test':
            if self.config.TASK_TYPE == 'semantic_segmentation':
                transform = transforms.Compose([
                    transforms.Resize((self.config.INPUT_IMAGE_SIZE, self.config.INPUT_IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.config.DATASET_MEAN, self.config.DATASET_STD)
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(self.config.INPUT_IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(self.config.DATASET_MEAN, self.config.DATASET_STD)
                ])
        else:
            raise Exception('Set name {} not recognised, which should be "train" or "test".'.format(self.set_name))
        if self.config.DATASET_NAME == "CIFAR-10":
            self.dataset = torchvision.datasets.CIFAR10(
                root=data_path, train=self.set_name == 'train', download=download, transform=transform)
        elif self.config.DATASET_NAME == "CIFAR-100":
            self.dataset = torchvision.datasets.CIFAR100(
                root=data_path, train=self.set_name == 'train', download=download, transform=transform)
        elif self.config.DATASET_NAME == "MNIST":
            self.dataset = torchvision.datasets.MNIST(
                root=data_path, train=self.set_name == 'train', download=download, transform=transform)
        elif self.config.DATASET_NAME == "SVHN":
            split = 'train' if self.set_name == 'train' else 'test'
            self.dataset = torchvision.datasets.SVHN(
                root=data_path, split=split, download=download, transform=transform)
        elif self.config.DATASET_NAME == "ImageNet":
            self.dataset = torchvision.datasets.ImageNet(
                root=data_path, train=self.set_name == 'train', download=None, transform=transform)
        elif self.config.DATASET_NAME == "tiny-imagenet-200":
            self.dataset = ImageFolder(root=data_path, transform=transform)
        elif self.config.DATASET_NAME == "VOC":
            mask_transform = transforms.Compose([
                transforms.Resize((self.config.INPUT_IMAGE_SIZE, self.config.INPUT_IMAGE_SIZE),
                                  interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Lambda(
                    lambda img: replace_tensor_value(
                        torch.tensor(np.array(img), dtype=torch.long), [255], 0))
            ])
            self.dataset = torchvision.datasets.VOCSegmentation(
                root=data_path, year='2012', image_set='train' if self.set_name == 'train' else 'val',
                download=download, transform=transform, target_transform=mask_transform)
        elif self.config.DATASET_NAME == "Oxford-IIIT":
            mask_transform = transforms.Compose([
                transforms.Resize((self.config.INPUT_IMAGE_SIZE, self.config.INPUT_IMAGE_SIZE),
                                  interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Lambda(
                    lambda img: replace_tensor_value(
                        torch.tensor(np.array(img), dtype=torch.long), [2, 3], 0))
            ])
            self.dataset = torchvision.datasets.OxfordIIITPet(
                root=data_path, target_types='segmentation', download=download,
                split='trainval' if self.set_name == 'train' else 'test',
                transform=transform, target_transform=mask_transform)
        elif self.config.DATASET_NAME == 'SST2':
            train_set = self.config.DATASET if self.config.DATASET else SST2(
                root=os.path.join(self.config.DATASET_PATH, self.config.DATASET_NAME), split='train')
            self.vocab = self.config.VOCAB if self.config.VOCAB else build_vocab_from_dataset(train_set, max_len=50000)
            if self.set_name == 'train':
                self.dataset = train_set[:-self.config.TEST_DATA_LENGTH]
            else:
                self.dataset = train_set[-self.config.TEST_DATA_LENGTH:]
        elif self.config.DATASET_NAME == 'IMDB':
            train_set = self.config.DATASET if self.config.DATASET else IMDB(
                root=os.path.join(self.config.DATASET_PATH, self.config.DATASET_NAME), split='train')
            self.vocab = self.config.VOCAB if self.config.VOCAB else build_vocab_from_dataset(train_set, max_len=50000)
            if self.set_name == 'train':
                self.dataset = train_set
            else:
                self.dataset = IMDB(
                    root=os.path.join(self.config.DATASET_PATH, self.config.DATASET_NAME), split='test')
        else:
            raise Exception("Data Source Not Found.")

    def get_data_loader(self, client_index=None):
        data_length = self.config.DATA_LENGTH if self.set_name == 'train' else self.config.TEST_DATA_LENGTH
        if isinstance(self.dataset, list):
            self.dataset = self.dataset[:data_length]
        elif isinstance(self.dataset, ImageFolder):
            self.dataset.imgs = self.dataset.imgs[:data_length]
        else:
            self.dataset.data, self.dataset.targets = self.dataset.data[:data_length], self.dataset.targets[:data_length]
        if self.set_name == 'train':
            if client_index is None:
                client_index = self.config.CLIENT_INDEX
            client_set = self.get_client_data(client_index)
            if self.config.MODEL_NAME.startswith('Transformer'):
                return self.get_transformer_dataloader(client_set)
            else:
                return self.get_cnn_dataloader(client_set)
        else:
            if self.config.MODEL_NAME.startswith('Transformer'):
                return self.get_transformer_dataloader(self.dataset)
            else:
                return self.get_cnn_dataloader(self.dataset)

    def get_cnn_dataloader(self, dataset):
        if self.set_name == 'train':
            return DataLoader(dataset,
                              batch_size=self.config.BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              worker_init_fn=seed_worker)
        else:
            if self.config.ENABLE_VAL:
                val_length = int(self.config.TEST_DATA_LENGTH * self.config.VAL_RATE)
                test_length = self.config.TEST_DATA_LENGTH - val_length
                val_set, test_set = random_split(dataset, [val_length, test_length])
                return (DataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False, drop_last=True,
                                   worker_init_fn=seed_worker),
                        DataLoader(test_set, batch_size=self.config.BATCH_SIZE, shuffle=False, drop_last=True,
                                   worker_init_fn=seed_worker))
            else:
                return None, DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, drop_last=True,
                                        worker_init_fn=seed_worker)

    def get_transformer_dataloader(self, dataset):
        def collate_batch(batch):
            label_list, text_list = [], []
            for (text, label) in batch:
                label_list.append(label)
                processed_text = torch.tensor(self.vocab(nltk_tokenizer(text)), dtype=torch.int64)
                if len(processed_text) > MAX_SEQ_LENGTH:
                    processed_text = processed_text[:MAX_SEQ_LENGTH]
                text_list.append(processed_text)
            return (pad_sequence(text_list, padding_value=self.vocab("<pad>"), batch_first=True),
                    torch.tensor(label_list, dtype=torch.int64))

        if self.set_name == 'train':
            return DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, collate_fn=collate_batch,
                              worker_init_fn=seed_worker)
        else:
            if self.config.ENABLE_VAL:
                val_length = int(self.config.TEST_DATA_LENGTH * self.config.VAL_RATE)
                test_length = self.config.TEST_DATA_LENGTH - val_length
                val_set, test_set = random_split(dataset, [val_length, test_length])
                return (DataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False, collate_fn=collate_batch,
                                   drop_last=True, worker_init_fn=seed_worker),
                        DataLoader(test_set, batch_size=self.config.BATCH_SIZE, shuffle=False, collate_fn=collate_batch,
                                   drop_last=True, worker_init_fn=seed_worker))
            return None, DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, collate_fn=collate_batch,
                                    drop_last=True, worker_init_fn=seed_worker)

    def get_client_data(self, client_index):
        if self.config.DATASET_SPLIT == 'balance_iid':
            return random_split(
                self.dataset, [self.config.CLIENT_DATA_LENGTH
                          for _ in range(self.config.CLIENT_NUM * self.config.SIMULATED_CLIENT_NUM)])[client_index]
        else:
            partition = self.dirichlet_partition()
            client_indices = partition[client_index]
            return Subset(self.dataset, client_indices)

    def dirichlet_partition(self, partition_file=True):
        file_path = os.path.join(
            self.config.DATASET_PATH, self.config.DATASET_NAME,
            "{}_partition_{}.pickle".format(self.config.DATASET_SPLIT, self.config.CLIENT_NUM))
        if partition_file and os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        if isinstance(self.dataset, list):
            targets = [data[1] for data in self.dataset]
        elif isinstance(self.dataset, ImageFolder):
            targets = [data[1] for data in self.dataset.imgs]
        else:
            targets = self.dataset.targets
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)
        if self.config.DATASET_SPLIT == 'unbalance_non_iid':
            partition = self.dirichlet_unbalance_partition(targets, 0.5)
        elif self.config.DATASET_SPLIT == 'balance_non_iid':
            partition = self.dirichlet_balance_partition(targets, 0.5)
        else:
            raise Exception("Unrecognized dataset split: {}".format(self.config.DATASET_SPLIT))
        if partition_file and not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                pickle.dump(partition, f)
        return partition

    def dirichlet_unbalance_partition(self, targets, dir_alpha, min_require_size=None):
        num_clients = self.config.CLIENT_NUM * self.config.SIMULATED_CLIENT_NUM
        if min_require_size is None:
            min_require_size = self.config.NUM_CLASS
        num_samples = targets.shape[0]
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(self.config.NUM_CLASS)]
            # for each class in the dataset
            for k in range(self.config.NUM_CLASS):
                idx_k = np.where(targets == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(dir_alpha, num_clients))
                # Balance
                proportions = np.array(
                    [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                     zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        client_dict = dict()
        for cid in range(num_clients):
            np.random.shuffle(idx_batch[cid])
            client_dict[cid] = np.array(idx_batch[cid])
        return client_dict

    def dirichlet_balance_partition(self, targets, dir_alpha, verbose=False):
        num_clients = self.config.CLIENT_NUM * self.config.SIMULATED_CLIENT_NUM
        client_sample_nums = [self.config.CLIENT_DATA_LENGTH for _ in range(self.config.CLIENT_NUM)]
        class_priors = np.random.dirichlet(alpha=[dir_alpha] * self.config.NUM_CLASS,
                                           size=num_clients)
        prior_cumsum = np.cumsum(class_priors, axis=1)
        idx_list = [np.where(targets == i)[0] for i in range(self.config.NUM_CLASS)]
        class_amount = [len(idx_list[i]) for i in range(self.config.NUM_CLASS)]
        client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                          range(num_clients)]
        while np.sum(client_sample_nums) != 0:
            curr_cid = np.random.randint(num_clients)
            # If current node is full resample a client
            if verbose:
                print('Remaining Data: %d' % np.sum(client_sample_nums))
            if client_sample_nums[curr_cid] <= 0:
                continue
            client_sample_nums[curr_cid] -= 1
            curr_prior = prior_cumsum[curr_cid]
            while True:
                curr_class = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if no rest in current class samples
                if class_amount[curr_class] <= 0:
                    # Exception handling: If the current class has no samples left, randomly select a non-zero class
                    while True:
                        new_class = np.random.randint(self.config.NUM_CLASS)
                        if class_amount[new_class] > 0:
                            curr_class = new_class
                            break
                class_amount[curr_class] -= 1
                client_indices[curr_cid][client_sample_nums[curr_cid]] = idx_list[curr_class][class_amount[curr_class]]
                break
        client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
        return client_dict
