"""Pure accounting/stop policy: no model, torch, GPU, network, or sleeping."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
import importlib.util
from pathlib import Path

import pytest


def api():
    path = Path(__file__).parents[1] / 'reference_external/FedOptima/hsfl/control.py'
    assert path.exists(), 'FedOptima stopping policy is not implemented'
    spec = importlib.util.spec_from_file_location('fedoptima_control', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExperimentControl


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def ready(**kwargs):
    clock = Clock()
    control = api()(clock=clock, **kwargs)
    control.start()
    return control, clock


def evaluate(control, correct=7000, total=10000, **metadata):
    control.record_visits(0, 30000)
    return control.record_evaluation(correct, total, **metadata)


def test_start_excludes_preparation_and_rejects_duplicate_start():
    clock = Clock()
    control = api()(clock=clock)
    clock.now += 50000
    assert control.snapshot()['elapsed_s'] == 0
    assert not control.poll_stop()
    control.start()
    assert not control.poll_stop()
    with pytest.raises(RuntimeError):
        control.start()
    clock.now += 2
    assert control.snapshot()['elapsed_s'] == 2


def test_timeout_without_any_job_at_exact_five_hours():
    control, clock = ready()
    clock.now += 17999.999
    assert not control.poll_stop()
    clock.now += 0.001
    assert control.poll_stop()
    assert control.stop_event.is_set()
    assert control.snapshot()['reason'] == 'wall_time_limit'
    assert control.snapshot()['total_visits'] == 0


def test_exact_seventy_does_not_stop_but_strictly_greater_does():
    control, _ = ready()
    for _ in range(9):
        evaluate(control, 10000)
    assert not control.poll_stop()
    assert control.snapshot()['latest10mean'] is None
    evaluate(control, 0)
    assert Decimal(control.snapshot()['latest10mean']) == 90
    assert control.poll_stop()
    exact, _ = ready()
    for _ in range(10):
        evaluate(exact)
    assert Decimal(exact.snapshot()['latest10mean']) == 70
    assert not exact.poll_stop()
    evaluate(exact, 7001)
    assert Decimal(exact.snapshot()['latest10mean']) == Decimal('70.001')
    assert exact.snapshot()['reason'] == 'rolling_accuracy'


def test_window_replaces_old_rows_instead_of_averaging_all_history():
    control, _ = ready()
    evaluate(control, 0)
    for _ in range(9):
        evaluate(control)
    assert Decimal(control.snapshot()['latest10mean']) == 63
    evaluate(control, 7001)
    assert Decimal(control.snapshot()['latest10mean']) == Decimal('70.001')
    assert control.poll_stop()


def test_rounds_are_sequential_and_trigger_visits_differ_from_actual_visits():
    control, _ = ready()
    control.record_visits(2, 29999)
    assert not control.evaluation_due()
    with pytest.raises(RuntimeError):
        control.record_evaluation(0, 10000)
    control.record_visits(5, 60058)
    for expected_round in (1, 2, 3):
        assert control.evaluation_due()
        row = control.record_evaluation(0, 10000, model_version=19)
        assert row['round'] == expected_round
        assert row['trigger_visits'] == expected_round * 30000
        assert row['actual_visits'] == 90057
        assert row['metadata'] == {'model_version': 19}
    assert not control.evaluation_due()
    assert control.snapshot()['client_visits'] == {2: 29999, 5: 60058}


def test_no_fixed_hundred_round_or_three_million_visit_cap():
    control, _ = ready()
    for _ in range(111):
        evaluate(control, 0)
    assert control.snapshot()['evaluations'][-1]['round'] == 111
    assert control.snapshot()['total_visits'] == 3330000
    assert not control.poll_stop()


def test_evaluation_mean_uses_each_integer_count_not_pooled_counts():
    control, _ = ready(window=2)
    evaluate(control, 1, 2)
    evaluate(control, 9, 10)
    assert Decimal(control.snapshot()['latest10mean']) == 70
    assert not control.poll_stop()


def test_decimal_context_does_not_change_strict_boundary():
    control, _ = ready(window=3)
    with localcontext() as context:
        context.prec = 2
        evaluate(control, 1, 3)
        evaluate(control, 2, 3)
        evaluate(control, 1, 1)
    assert not control.poll_stop()  # (100/3+200/3+100)/3 < 70


def test_recurring_fractions_at_exact_seventy_do_not_trigger_stop():
    control, _ = ready(window=4)
    for correct, total in [(1, 3), (2, 3), (9, 10), (9, 10)]:
        evaluate(control, correct, total)
    assert Decimal(control.snapshot()['latest10mean']) == 70
    assert not control.poll_stop()


def test_timeout_has_priority_when_evaluation_finishes_after_deadline():
    control, clock = ready()
    for _ in range(9):
        evaluate(control, 10000)
    clock.now += 18000
    evaluate(control, 10000)
    assert control.snapshot()['reason'] == 'wall_time_limit'


def test_concurrent_accounting_cannot_drop_visits_and_snapshots_are_detached():
    control, _ = ready()
    def worker(client):
        for _ in range(500):
            control.record_visits(client, 56)
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(worker, range(10)))
    snapshot = control.snapshot()
    assert snapshot['total_visits'] == 280000
    assert snapshot['client_visits'] == dict.fromkeys(range(10), 28000)
    snapshot['client_visits'][0] = -1
    assert control.snapshot()['client_visits'][0] == 28000
    row = control.record_evaluation(1, 10000, details={'ids': [1]})
    row['metadata']['details']['ids'].append(2)
    assert control.snapshot()['evaluations'][0]['metadata']['details']['ids'] == [1]


def test_first_stop_reason_wins_and_inflight_success_still_counts():
    control, clock = ready()
    assert control.request_stop('worker_failure')
    assert not control.request_stop('other')
    clock.now += 18000
    assert control.poll_stop()
    control.record_visits(0, 56)
    assert control.snapshot()['reason'] == 'worker_failure'
    assert control.snapshot()['total_visits'] == 56
    assert control.snapshot()['elapsed_s'] == 0


def test_simultaneous_evaluation_completions_have_unique_sequential_rounds():
    control, _ = ready()
    control.record_visits(0, 300000)
    with ThreadPoolExecutor(max_workers=10) as executor:
        rows = list(executor.map(lambda _: control.record_evaluation(0, 10000), range(10)))
    assert sorted(row['round'] for row in rows) == list(range(1, 11))
    assert not control.evaluation_due()
    with pytest.raises(RuntimeError):
        control.record_evaluation(0, 10000)


def test_simultaneous_stop_requests_preserve_one_winning_reason():
    control, _ = ready()
    def stop(index):
        return index, control.request_stop(f'failure_{index}')
    with ThreadPoolExecutor(max_workers=10) as executor:
        rows = list(executor.map(stop, range(10)))
    winners = [index for index, won in rows if won]
    assert len(winners) == 1
    assert control.snapshot()['reason'] == f'failure_{winners[0]}'


def test_cancellation_before_start_cannot_restart_the_experiment():
    control = api()()
    assert control.request_stop('preparation_failed')
    with pytest.raises(RuntimeError):
        control.start()
    assert control.snapshot()['elapsed_s'] == 0


@pytest.mark.parametrize('kwargs', [
    {'max_wall_s': 0}, {'max_wall_s': float('inf')}, {'max_wall_s': float('nan')},
    {'max_wall_s': True}, {'window': 0}, {'window': 1.5}, {'window': True},
    {'threshold': -1}, {'threshold': 101}, {'threshold': 'NaN'},
    {'samples_per_round': 0}, {'samples_per_round': 3.5}, {'clock': None},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        api()(**kwargs)


@pytest.mark.parametrize('correct,total', [
    (-1, 100), (101, 100), (0, 0), (1, -1), (1.5, 2), (True, 2),
    (float('nan'), 10), (1, float('inf')),
])
def test_invalid_evaluation_counts_do_not_consume_round(correct, total):
    control, _ = ready()
    control.record_visits(0, 30000)
    with pytest.raises((ValueError, TypeError)):
        control.record_evaluation(correct, total)
    assert control.evaluation_due()
    assert control.snapshot()['evaluations'] == []


@pytest.mark.parametrize('client,samples', [
    (-1, 56), (True, 56), (0, 0), (0, -1), (0, 1.5), (0, True),
    (0, float('nan')), (0, float('inf')),
])
def test_invalid_visits_do_not_mutate_totals(client, samples):
    control, _ = ready()
    with pytest.raises((ValueError, TypeError)):
        control.record_visits(client, samples)
    assert control.snapshot()['total_visits'] == 0


def test_accounting_requires_start_and_stop_reason_must_be_nonempty():
    control = api()()
    with pytest.raises(RuntimeError):
        control.record_visits(0, 128)
    with pytest.raises(RuntimeError):
        control.record_evaluation(0, 10000)
    assert not control.evaluation_due()
    with pytest.raises(ValueError):
        control.request_stop(' ')


def test_invalid_clock_cannot_create_a_nonfinite_elapsed_time():
    control, clock = ready()
    clock.now = float('nan')
    with pytest.raises(ValueError):
        control.poll_stop()


def test_backwards_clock_is_rejected():
    control, clock = ready()
    clock.now -= 1
    with pytest.raises(ValueError):
        control.poll_stop()
