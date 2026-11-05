import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
from networks.vgg import VGG
from networks.resnet import ResNet
from networks.mobilenet import MobileNet
from networks.fcn import FCN
from networks.deeplab import DeepLab
from networks.transformer import Transformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_net(config, model_type="server"):
    """
    Build a neural network
    :param config: network config
    :param model_type: "complete": complete model; "server": server-side model; "client": client-side model;
        "auxiliary_net": auxiliary networks; "auxiliary_classifier": auxiliary classifier.
    """
    if config.MODEL_NAME.startswith("VGG"):
        net = VGG(config, model_type)
    elif config.MODEL_NAME.startswith("ResNet"):
        net = ResNet(config, model_type)
    elif config.MODEL_NAME.startswith("MobileNet"):
        net = MobileNet(config, model_type)
    elif config.MODEL_NAME.startswith("FCN"):
        net = FCN(config, model_type)
    elif config.MODEL_NAME.startswith("DeepLab"):
        net = DeepLab(config, model_type)
    elif config.MODEL_NAME.startswith("Transformer"):
        net = Transformer(config, model_type)
    else:
        raise Exception("Undefined Neural Network.")
    logger.debug(str(net))
    return net


def get_average_model(models, weights=None):
    if weights is None or weights == []:
        weights = [1 for _ in models]
    assert len(weights) == len(models), \
        "The number of weights {} is not consistent with the number of models {}.".format(len(weights), len(models))
    return {
        weight_name: sum([models[i][weight_name] * weights[i] for i in range(len(models))]) / sum(weights)
        for weight_name in models[0].keys()
    }


def merge_models(client_model, server_model, complete_model):
    client_keys = list(client_model)
    server_keys = list(server_model)
    complete_keys = list(complete_model)
    for i in range(len(client_keys)):
        complete_model[complete_keys[i]] = client_model[client_keys[i]]
    for i in range(len(server_keys)):
        complete_model[complete_keys[i+len(client_keys)]] = server_model[server_keys[i]]


def split_models(client_model, server_model, complete_model):
    client_keys = list(client_model)
    server_keys = list(server_model)
    complete_keys = list(complete_model)
    for i in range(len(client_keys)):
        client_model[client_keys[i]] = complete_model[complete_keys[i]]
    for i in range(len(server_keys)):
        server_model[server_keys[i]] = complete_model[complete_keys[i+len(client_keys)]]


def split_nets(client_net, server_net, complete_net):
    client_model = client_net.state_dict()
    server_model = server_net.state_dict()
    complete_model = complete_net.state_dict()
    split_models(client_model, server_model, complete_model)
    client_net.load_state_dict(client_model)
    server_net.load_state_dict(server_model)


def get_model_diff(model1, model2):
    return {
        weight_name: model1[weight_name] - model2[weight_name]
        for weight_name in model1.keys()
    }


def update_model(model, model_diff, lr):
    return {
        weight_name: model[weight_name] - model_diff[weight_name] * lr
        for weight_name in model.keys()
    }


def get_optimiser(net, layer_num_on_client, learning_rate, algorithm='sgd', weight_decay=0):
    if (layer_num_on_client == -1 and net.model_type == 'server') or (
            layer_num_on_client == 0 and net.model_type == 'client'):
        return None
    if algorithm == 'adam':
        optimiser = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=learning_rate,
                                     weight_decay=weight_decay)
    elif algorithm == 'sgd':
        optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise Exception("Optimisation algorithm {} not found.".format(algorithm))
    return optimiser


def get_state_dict_size(state_dict):
    total_size = 0
    for params in state_dict.values():
        param_size = 0
        for dim_size in params.size():
            if param_size == 0:
                param_size = dim_size
            else:
                param_size *= dim_size
        total_size += param_size
    return total_size


def print_log(config, test_accuracy, training_time, communication_amount, split_points=0, parallel_batch_nums=0):
    logger.info("Experiment configuration:")
    logger.info("Client number: {}".format(config.CLIENT_NUM))
    logger.info("Layer number on clients: {}".format(split_points))
    logger.info("Parallel batch number: {}".format(parallel_batch_nums))
    logger.info("Whether same data size for each client: {}".format(config.SAME_DATA_SIZE_FOR_EACH_CLIENT))
    logger.info("Data size: {}".format(config.DATA_LENGTH))
    logger.info("Batch size: {}".format(config.BATCH_SIZE))
    logger.info("Aggregation frequency: {}".format(config.AGGREGATION_FREQ))
    logger.info("Epoch number: {}".format(config.EPOCH_NUM))
    logger.info("Server thread number: {}".format(config.SERVER_CPU_NUM))
    logger.info("Client thread number: {}".format(config.CLIENT_CPU_NUM))
    logger.info("Results:")
    logger.info('Test Accuracy: {:.2f}'.format(test_accuracy))
    logger.info('Training time: {:.2f}'.format(training_time))
    logger.info('Total communication amount: {:.4e}'.format(communication_amount))


def save_result(config, file, test_accuracy, training_time, communication_amount, split_points=0, parallel_batch_nums=0):
    file.write("Experiment configuration:\n")
    file.write("Client number: {}\n".format(config.CLIENT_NUM))
    file.write("Layer number on clients: {}\n".format(split_points))
    file.write("Parallel batch number: {}\n".format(parallel_batch_nums))
    file.write("Whether same data size for each client: {}\n".format(config.SAME_DATA_SIZE_FOR_EACH_CLIENT))
    file.write("Data size: {}\n".format(config.DATA_LENGTH))
    file.write("Batch size: {}\n".format(config.BATCH_SIZE))
    file.write("Aggregation frequency: {}\n".format(config.AGGREGATION_FREQ))
    file.write("Epoch number: {}\n".format(config.EPOCH_NUM))
    file.write("Server thread number: {}\n".format(config.SERVER_CPU_NUM))
    file.write("Client thread number: {}\n".format(config.CLIENT_CPU_NUM))
    file.write("Results:\n")
    file.write("Test accuracy: {:.2f}\n".format(test_accuracy))
    file.write("Training time: {:.2f}\n".format(training_time))
    file.write("Training communication amount: {:.4e}\n".format(communication_amount.get("TRAINING", 0)))
    file.write("Aggregation communication amount: {:.4e}\n".format(communication_amount.get("AGGREGATION", 0)))
    file.write("Testing communication amount: {:.4e}\n".format(communication_amount.get("TESTING", 0)))
    file.write("Total communication amount: {:.4e}\n".format(communication_amount["TOTAL"]))
    file.write("\n")


def split_batch(inputs, targets, partition_num=1, keep_last=True):
    chunk_size = inputs.size(dim=0) // partition_num
    remaining = inputs.size(dim=0) % chunk_size
    micro_inputs = inputs.split(chunk_size, 0)
    micro_targets = targets.split(chunk_size, 0)
    if remaining == 1 or (not keep_last and remaining > 0):
        return zip(micro_inputs[:-1], micro_targets[:-1])
    return zip(micro_inputs, micro_targets)


def draw_timeline(start_iter, end_iter, server_timing, client_timing,
                  time_difference=0., save_address=None, show=False):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(16, 4))
    timings = {
        'server': server_timing,
        'client': client_timing,
    }
    y_labels = ['client_comp', 'download', 'upload', 'server_comp']
    y_data = {
        'server_comp': [],
        'upload': [],
        'download': [],
        'client_comp': [],
    }
    total_start_time = 0
    total_end_time = 100
    for node_name, timing in timings.items():
        for i in range(start_iter, end_iter):
            for name, sub_timings in timing.break_downs[0][i].items():
                if name == 'forward':
                    color = 'green'
                    y_label = node_name + '_comp'
                elif name == 'backward':
                    color = 'blue'
                    y_label = node_name + '_comp'
                elif name == 'upload':
                    color = 'yellow'
                    y_label = name
                elif name == 'download':
                    color = 'red'
                    y_label = name
                elif name == 'update':
                    color = 'black'
                    y_label = node_name + '_comp'
                elif name == 'idle':
                    color = 'white'
                    y_label = node_name + '_comp'
                elif name == 'load':
                    color = 'orange'
                    y_label = node_name + '_comp'
                elif name == 'iter' and node_name == 'client':
                    if i == start_iter:
                        total_start_time = sub_timings[0].start_time - time_difference
                    elif i == end_iter - 1:
                        total_end_time = sub_timings[0].stop_time - time_difference
                    continue
                else:
                    continue
                y_data[y_label].append([node_name, sub_timings, color])
    for y_label in y_labels:
        for node_name, sub_timings, color in y_data[y_label]:
            for sub_timing in sub_timings:
                start_time = sub_timing.start_time - total_start_time
                if node_name == "client":
                    start_time -= time_difference
                ax.barh(y_label, sub_timing.elapsed_time, left=start_time,
                        align='center', color=color, edgecolor='black')
    ax.set_xlabel('Timeline')
    ax.set_title('Performance Analysis')
    ax.set_xlim((0, total_end_time - total_start_time))
    if save_address:
        plt.savefig(save_address)
    if show:
        plt.show()
    plt.close(fig)


def save_stats_csv(stat, file_name):
    result_dict = {
        "experiment_name": stat["experiment_name"],
        "client_index": stat["client_index"],
        "server_forward": stat["training_time"]["server"]["forward"],
        "server_backward": stat["training_time"]["server"]["backward"],
        "server_update": stat["training_time"]["server"]["update"],
        "server_idle": stat["training_time"]["server"]["idle"],
        "client_data_load": stat["training_time"]["client"]["data_load"],
        "client_forward": stat["training_time"]["client"]["forward"],
        "client_backward": stat["training_time"]["client"]["backward"],
        "client_update": stat["training_time"]["client"]["update"],
        "client_idle": stat["training_time"]["client"]["idle"],
        "client_straggler": stat["training_time"]["client"]["straggler"],
        "upload": stat["training_time"]["upload"],
        "download": stat["training_time"]["download"],
        "total": stat["training_time"]["server"]["total"],
        "converge": stat["training_time"]["server"]["converge"],
        "best step": stat["training_time"]["server"]["best step"],
        "comm amount": (stat["server_communication_amount"] + stat["client_communication_amount"]) / 1024 / 1024,
        "test_accuracy": stat["test_accuracy"],
        "train_loss": str(stat["train_loss"]),
        "train_acc": str(stat["train_acc"]),
        "val_loss": str(stat["val_loss"]),
        "val_acc": str(stat["val_acc"]),
        "aggregate": stat["training_time"]["server"]["aggregate"],
        "time_line": str(stat["time_line"])
    }
    result_df = pd.DataFrame(result_dict, index=[0])
    if os.path.exists(file_name):
        result_df.to_csv(file_name, mode='a', header=False)
    else:
        result_df.to_csv(file_name)


def save_multi_stats_csv(stat, file_name):
    result_dict = {
        "experiment_name": stat["experiment_name"],
        "training_time": str(stat["training_time"]["total"]),
        "server_total": str(stat["training_time"]["server"]["total"]),
        "server_forward": str(stat["training_time"]["server"]["forward"]),
        "server_backward": str(stat["training_time"]["server"]["backward"]),
        "server_update": str(stat["training_time"]["server"]["update"]),
        "server_idle": str(stat["training_time"]["server"]["idle"]),
        "server_aggregate": str(stat["training_time"]["server"]["aggregate"]),
        "client_total": str([client_timing["total"] for client_timing in stat["training_time"]["client"]]),
        "client_data_load": str([client_timing["data_load"] for client_timing in stat["training_time"]["client"]]),
        "client_forward": str([client_timing["forward"] for client_timing in stat["training_time"]["client"]]),
        "client_backward": str([client_timing["backward"] for client_timing in stat["training_time"]["client"]]),
        "client_update": str([client_timing["update"] for client_timing in stat["training_time"]["client"]]),
        "client_idle": str([client_timing["idle"] for client_timing in stat["training_time"]["client"]]),
        "client_straggler": str([client_timing["straggler"] for client_timing in stat["training_time"]["client"]]),
        "upload": str(stat["training_time"]["server"]["upload"]),
        "download": str(sum([client_timing["download"] for client_timing in stat["training_time"]["client"]])),
        "total": str(stat["training_time"]["server"]["total"]),
        "converge": str(stat["training_time"]["server"]["converge"]),
        "best step": str(stat["training_time"]["server"]["best step"]),
        "comm amount": (sum(stat["server_communication_amount"])
                        + sum(stat["client_communication_amount"])) / 1024 / 1024,
        "test_accuracy": stat["test_accuracy"],
        "train_loss": str(stat["train_loss"]),
        "train_acc": str(stat["train_acc"]),
        "val_loss": str(stat["val_loss"]),
        "val_acc": str(stat["val_acc"]),
        "time_line": str(stat["time_line"])
    }
    result_df = pd.DataFrame(result_dict, index=[0])
    if os.path.exists(file_name):
        result_df.to_csv(file_name, mode='a', header=False)
    else:
        result_df.to_csv(file_name)


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
