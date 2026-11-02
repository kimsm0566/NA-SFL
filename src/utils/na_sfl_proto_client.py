"""v3/v4 retained SD/Prototype graphs; every Client waits for its Server reply."""
from dataclasses import replace
from utils.sflv2 import SFLV2Client, _timed


class PrototypeClient(SFLV2Client):
    def __init__(self, client_id, model, config, *, prototype_mode=False):
        super().__init__(client_id, model, config)
        self.prototype_mode = prototype_mode
        self.gradient_mode = 'server'
        self.auxiliary_head = self.auxiliary_optimizer = None

    def prepare_upload(self, job_id, begun, labels):
        if self.pending is None or self.pending.job_id != job_id or self.pending.version != self.version:
            raise RuntimeError('prototype upload requires the matching retained graph')
        if not self.prototype_mode:
            return dict(job_kind='SD_JOB', activation=begun['activation'], labels=labels,
                        prototype_compute_s=0.)
        from utils.na_sfl_prototype import make_local_prototypes
        (protos, classes, counts), elapsed = _timed(self.device,
            lambda: make_local_prototypes(self.pending.activation, labels))
        # p retains the original z graph. SFLV2Client.complete applies grad_p to p,
        # so autograd distributes the class mean's 1/count factor exactly once.
        self.pending = replace(self.pending, activation=protos)
        return dict(job_kind='PROTO_JOB', activation=protos.detach().clone(), labels=classes,
                    counts=counts, prototype_compute_s=elapsed)
