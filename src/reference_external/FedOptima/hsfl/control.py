"""Thread-safe visit accounting and the matched experiment's stop policy.

Call ``start`` immediately before the first model broadcast, count a batch only
after its local optimizer step succeeds, and use one evaluator. The runtime must
call ``poll_stop`` independently of incoming jobs (its watchdog runs every .2s).
Initial evaluation belongs outside this controller. No background thread, model,
network operation, fixed round limit, or total visit limit is implemented here.
"""
import copy
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import math
from numbers import Integral
import threading
import time


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def _decimal(value, name):
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a finite number')
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not result.is_finite():
        raise ValueError(f'{name} must be a finite number')
    return result


def _percentage(value):
    # A local context makes output independent of the evaluator thread's context.
    # Retain an exact rational for the strict comparison, including recurring
    # fractions whose finite Decimal representation could round at the boundary.
    with localcontext() as context:
        context.prec = 60
        return str(Decimal(value.numerator) / Decimal(value.denominator))


class ExperimentControl:
    """All public methods serialize policy state through one reentrant lock.

    Return values: start -> None; record_visits -> new integer total;
    evaluation_due/poll_stop/request_stop -> bool; record_evaluation -> detached
    row dict; snapshot -> detached dict. request_stop returns True only for the
    first stop request, whose reason wins. poll_stop never waits.

    snapshot's ``latest10mean`` is a Decimal percentage string (None until the
    configured window is full); the name stays stable for the default ten rows.
    ``elapsed_s`` freezes at the first stop request; later successful in-flight
    batches still count. A row records its sequential round, trigger_visits,
    actual_visits at completion, elapsed_s, correct/total, accuracy and metadata.
    Metadata is nested so callers cannot overwrite authoritative accounting.
    """

    def __init__(self, max_wall_s=18000, window=10, threshold=70,
                 samples_per_round=30000, clock=time.monotonic):
        wall = _decimal(max_wall_s, 'max_wall_s')
        if wall <= 0:
            raise ValueError('max_wall_s must be positive')
        self.max_wall_s = float(wall)
        if not math.isfinite(self.max_wall_s):
            raise ValueError('max_wall_s exceeds the clock range')
        self.window = _integer(window, 'window', 1)
        self.samples_per_round = _integer(samples_per_round, 'samples_per_round', 1)
        self.threshold = _decimal(threshold, 'threshold')
        if not 0 <= self.threshold <= 100:
            raise ValueError('threshold must be between 0 and 100')
        if not callable(clock):
            raise TypeError('clock must be callable')
        self._clock = clock
        self._lock = threading.RLock()
        self.stop_event = threading.Event()
        self._started_at = None
        self._stopped_elapsed = None
        self._reason = None
        self._total_visits = 0
        self._client_visits = {}
        self._evaluations = []
        self._latest_mean = None

    def _now(self):
        value = float(self._clock())
        if not math.isfinite(value):
            raise ValueError('clock must return finite time')
        return value

    def _elapsed(self):
        if self._stopped_elapsed is not None:
            return self._stopped_elapsed
        if self._started_at is None:
            return 0.0
        elapsed = self._now() - self._started_at
        if elapsed < 0:
            raise ValueError('clock moved before the experiment start')
        return elapsed

    def _require_started(self):
        if self._started_at is None:
            raise RuntimeError('start must precede training accounting')

    def _stop(self, reason, elapsed):
        if self._reason is not None:
            return False
        self._reason = reason
        self._stopped_elapsed = elapsed
        self.stop_event.set()
        return True

    def _poll(self, elapsed):
        if self._started_at is not None and elapsed >= self.max_wall_s:
            self._stop('wall_time_limit', elapsed)
        return self.stop_event.is_set()

    def start(self):
        with self._lock:
            if self._started_at is not None or self.stop_event.is_set():
                raise RuntimeError('experiment cannot be started twice or after stop')
            self._started_at = self._now()

    def record_visits(self, client_id, samples):
        client_id = _integer(client_id, 'client_id')
        samples = _integer(samples, 'samples', 1)
        with self._lock:
            self._require_started()
            self._total_visits += samples
            self._client_visits[client_id] = self._client_visits.get(client_id, 0) + samples
            return self._total_visits

    def evaluation_due(self):
        with self._lock:
            return (self._started_at is not None and not self.stop_event.is_set()
                    and self._total_visits >=
                    (len(self._evaluations) + 1) * self.samples_per_round)

    def record_evaluation(self, correct, total, **metadata):
        correct = _integer(correct, 'correct')
        total = _integer(total, 'total', 1)
        if correct > total:
            raise ValueError('correct must not exceed total')
        metadata = copy.deepcopy(metadata)
        with self._lock:
            self._require_started()
            round_id = len(self._evaluations) + 1
            trigger = round_id * self.samples_per_round
            if self._total_visits < trigger:
                raise RuntimeError('evaluation has not reached its visit threshold')
            elapsed = self._elapsed()
            self._poll(elapsed)
            row = {
                'round': round_id, 'trigger_visits': trigger,
                'actual_visits': self._total_visits, 'elapsed_s': elapsed,
                'correct': correct, 'total': total,
                'accuracy': _percentage(Fraction(100 * correct, total)),
                'metadata': metadata,
            }
            self._evaluations.append(row)
            if len(self._evaluations) >= self.window:
                recent = self._evaluations[-self.window:]
                mean = sum((Fraction(100 * r['correct'], r['total']) for r in recent),
                           Fraction(0)) / self.window
                self._latest_mean = _percentage(mean)
                if mean > Fraction(self.threshold):
                    self._stop('rolling_accuracy', elapsed)
            return copy.deepcopy(row)

    def poll_stop(self):
        with self._lock:
            return self._poll(self._elapsed())

    def request_stop(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('reason must be a nonempty string')
        with self._lock:
            return self._stop(reason, self._elapsed())

    def snapshot(self):
        with self._lock:
            elapsed = self._elapsed()
            self._poll(elapsed)
            return copy.deepcopy({
                'started': self._started_at is not None,
                'elapsed_s': elapsed, 'total_visits': self._total_visits,
                'client_visits': self._client_visits,
                'evaluations': self._evaluations,
                'latest10mean': self._latest_mean, 'reason': self._reason,
                'stopped': self.stop_event.is_set(),
                'max_wall_s': self.max_wall_s, 'window': self.window,
                'threshold': str(self.threshold),
                'samples_per_round': self.samples_per_round,
            })
