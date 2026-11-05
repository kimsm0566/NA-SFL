import argparse
import platform

import config
from memory_profiler import memory_usage
from node.async_server import AsyncServer
from node.server import Server

if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def run(server_config=None):
    if server_config is None:
        parser = argparse.ArgumentParser(description='Server-side arguments.')
        parser.add_argument('--port', '-p', type=int, default=51000)
        parser.add_argument('--client_num', '-c', type=int, nargs='+', default=[1],
                            help='the number of clients, default [1]')
        parser.add_argument('--layer_num_on_client', '-l', type=int, default=-1,
                            help='the number of layer on clients, default -1: no partition')
        parser.add_argument('--epoch_num', '-e', type=int, default=100,
                            help='the number of training epochs, default 100')
        parser.add_argument('--same_data_size_for_each_client', '-s',
                            help='the input data size for each client is the same, default False', action='store_true')
        parser.add_argument('--aggregation_frequency', '-a', type=int, default=None,
                            help='the number of iterations after which aggregation happens')
        parser.add_argument('--batch_size', '-d', type=int, default=100, help='the size of mini-batch of data')
        parser.add_argument('--server_cpu_num', '-t', type=int, default=0, help='the number of cpus on server')
        parser.add_argument('--client_cpu_num', '-t', type=int, default=0, help='the number of cpus on clients')
        parser.add_argument('--data_size', '-x', type=int, default=10000, help='the size of training data')
        parser.add_argument('--test_data_size', '-y', type=int, default=10000, help='the size of test data')
        parser.add_argument('--enable_profiler', '-z',
                            help='enable profiler, default False', action='store_true')
        args = parser.parse_args()
        server_config = config.Config(
            port=args.port,
            client_num=args.client_num,
            layer_num_on_client=args.layer_num_on_client,
            epoch_num=args.epoch_num,
            same_data_size_for_each_client=args.same_data_size_for_each_client,
            aggregation_frequency=args.aggregation_frequency,
            batch_size=args.batch_size,
            server_cpu_num=args.server_cpu_num,
            client_cpu_num=args.client_cpu_num,
            data_size=args.data_size,
            test_data_size=args.test_data_size,
            enable_profiler=args.enable_profiler
        )
    server_config.server_init()
    if server_config.ASYNCHRONISATION:
        server = AsyncServer(server_config)
    else:
        server = Server(server_config)
    mem = max(memory_usage(proc=server.run))
    # server.run()
    print("Peak Memory Usage: {} MiB".format(mem))
    return


if __name__ == '__main__':
    run()
