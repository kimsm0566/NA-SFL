from torch.multiprocessing import Manager, Queue, Lock
from util.util import get_average_model


class AsyncAggregator:
    def __init__(self):
        self.global_model_with_version = Manager().Queue(1)
        self.lock = Lock()

    def init_global_model(self, global_model):
        if self.global_model_with_version.empty():
            self.global_model_with_version.put((global_model, 0))

    def aggregate(self, client_model, client_version, timing=None):
        self.lock.acquire()
        global_model, global_model_version = self.global_model_with_version.get()
        if timing:
            timing.start()
        # FedAsync
        alpha = 0.9
        a = 10
        b = 4
        difference = global_model_version - client_version
        staleness = 1 if difference <= b else 1 / (a * (difference - b) + 1)
        alpha *= staleness
        global_model = get_average_model(
            [global_model, client_model],
            [1 - alpha, alpha]
        )
        global_model_version += 1
        self.global_model_with_version.put((global_model, global_model_version))
        if timing:
            timing.stop()
        self.lock.release()
        return global_model, global_model_version
