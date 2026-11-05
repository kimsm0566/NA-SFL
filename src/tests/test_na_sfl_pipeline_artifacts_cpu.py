"""Hand-calculated event metrics; no model, GPU, network, or baseline mutations."""
import copy
import csv
import json

import pytest


def writer():
    from utils.na_sfl_pipeline_artifacts import write_pipeline_artifacts
    return write_pipeline_artifacts


def job(index, **fields):
    return {'job_id': [1, 0, index], 'round': 1, 'client_id': 0,
        'local_batch_index': index, 'samples': 128, 'pipeline_group_id': index // 2,
        'group_size': 2, 'client_optimizer_step_performed': index % 2 == 1,
        'client_optimizer_step_id': index // 2, 'job_kind': 'SD_JOB', **fields}


def write(tmp_path, jobs, round_row=None, **options):
    return writer()(tmp_path, [round_row or {'round': 1,
        'training_wall_s_debug_only': 20., 'worker_drain_elapsed_s': 20.,
        'server_updates': len(jobs)}], jobs,
        config={'n_clients': 1, 'slow_client_ids': [0]}, **options)['pipeline']


def test_overlap_is_union_of_different_jobs_and_pacing_is_separate(tmp_path):
    # Three intersections overlap each other: their union is [3,9], not 3+4.
    jobs = [job(0, client_forward_start_s=0., forward_compute_end_s=2.,
                activation_ready_s=3., sd_send_start_s=2., sd_receive_complete_s=8.),
            job(1, client_forward_start_s=3., forward_compute_end_s=6.,
                activation_ready_s=7., sd_send_start_s=6., sd_receive_complete_s=10.),
            job(2, client_forward_start_s=5., forward_compute_end_s=9.,
                activation_ready_s=10., sd_send_start_s=9., sd_receive_complete_s=12.)]
    result = write(tmp_path, jobs)
    assert result['client_forward_uplink_overlap_s'] == 6.
    assert result['client_forward_uplink_overlap_paced_s'] == 7.


def test_overlap_never_pairs_different_clients_or_same_job(tmp_path):
    jobs = [job(0, client_forward_start_s=0., forward_compute_end_s=3.,
                activation_ready_s=4., sd_send_start_s=2., sd_receive_complete_s=5.),
            job(1, client_id=1, job_id=[1, 1, 1], client_forward_start_s=2.,
                forward_compute_end_s=4., activation_ready_s=4.,
                sd_send_start_s=4., sd_receive_complete_s=7.)]
    assert write(tmp_path, jobs)['client_forward_uplink_overlap_s'] == 0.


def test_receive_boundary_touch_is_zero_overlap(tmp_path):
    jobs = [job(0, client_forward_start_s=0., forward_compute_end_s=1.,
                sd_send_start_s=1., sd_receive_complete_s=3.),
            job(1, client_forward_start_s=3., forward_compute_end_s=4.,
                sd_send_start_s=4., sd_receive_complete_s=6.)]
    result = write(tmp_path, jobs)
    assert result['client_forward_uplink_overlap_s'] == 0.
    assert result['client_forward_uplink_overlap_paced_s'] is None


def test_aligned_cuda_forward_is_preferred_and_uncertainty_shrinks_lower_bound(tmp_path):
    # Host F2=[2,9] would claim seven seconds. CUDA F2=[4,6] gives only two;
    # +/-0.25 clock alignment guarantees [4.25,5.75], or 1.5 seconds.
    jobs = [job(0, client_forward_start_s=0., forward_compute_end_s=1.,
                activation_ready_s=1., sd_send_start_s=1., sd_receive_complete_s=10.,
                cuda_client_forward_start_s=0., cuda_client_forward_end_s=1.,
                cuda_event_origin_host_offset_s=0., cuda_event_origin_host_uncertainty_s=.25),
            job(1, client_forward_start_s=2., forward_compute_end_s=9.,
                activation_ready_s=9., sd_send_start_s=10., sd_receive_complete_s=12.,
                cuda_client_forward_start_s=3., cuda_client_forward_end_s=5.,
                cuda_event_origin_host_offset_s=1., cuda_event_origin_host_uncertainty_s=.25)]
    result = write(tmp_path, jobs)
    assert result['client_forward_uplink_overlap_s'] == 2.
    assert result['client_forward_uplink_overlap_lower_bound_s'] == 1.5
    assert result['client_forward_uplink_overlap_host_phase_s'] == 7.
    assert result['client_forward_uplink_overlap_clock'] == 'aligned_cuda_events'


def test_unaligned_cuda_events_do_not_claim_cuda_overlap(tmp_path):
    jobs = [job(0, client_forward_start_s=0., forward_compute_end_s=1.,
                sd_send_start_s=1., sd_receive_complete_s=4.,
                cuda_client_forward_start_s=0., cuda_client_forward_end_s=1.),
            job(1, client_forward_start_s=2., forward_compute_end_s=3.,
                sd_send_start_s=4., sd_receive_complete_s=5.,
                cuda_client_forward_start_s=2., cuda_client_forward_end_s=3.)]
    result = write(tmp_path, jobs)
    assert result['client_forward_uplink_overlap_s'] == 1.
    assert result['client_forward_uplink_overlap_clock'] == 'host_forward_phase'
    assert result['client_forward_uplink_overlap_lower_bound_s'] is None


def test_queue_area_max_depth_and_service_wall_use_actual_arrival(tmp_path):
    jobs = [job(0, sd_receive_complete_s=0., server_start_s=3., server_end_s=5.,
                enqueue_s=.5, dequeue_s=2.5, server_model_compute_s=.2),
            job(1, sd_receive_complete_s=1., server_start_s=5., server_end_s=6.,
                enqueue_s=2., dequeue_s=4.5, server_model_compute_s=.3),
            job(2, sd_receive_complete_s=4., server_start_s=6., server_end_s=8.,
                enqueue_s=4.2, dequeue_s=5.5, server_model_compute_s=.4)]
    result = write(tmp_path, jobs, {'round': 1, 'training_wall_s_debug_only': 10.,
                                  'worker_drain_elapsed_s': 10., 'server_updates': 3})
    assert result['server_queue_wait_mean_s'] == 3.
    assert result['server_queue_wait_p95_s'] == pytest.approx(3.9)
    assert result['server_queue_wait_max_s'] == 4.
    assert result['server_queue_area_job_s'] == 9.
    assert result['server_queue_depth_mean'] == .9
    assert result['server_queue_depth_max'] == 2
    assert result['server_busy_wall_s'] == 5.
    assert result['server_idle_wall_s'] == 5.
    assert result['server_cuda_model_busy_s'] == pytest.approx(.9)
    assert result['measured_gpu_utilization_percent'] is None
    row = json.loads((tmp_path / 'pipeline_jobs.jsonl').read_text().splitlines()[0])
    assert row['server_queue_wait_s'] == 3.
    assert row['server_internal_queue_wait_s'] == 2.


def test_touching_queue_intervals_do_not_create_phantom_depth(tmp_path):
    jobs = [job(0, sd_receive_complete_s=0., server_start_s=1., server_end_s=2.),
            job(1, sd_receive_complete_s=1., server_start_s=2., server_end_s=3.)]
    assert write(tmp_path, jobs)['server_queue_depth_max'] == 1


def test_duplicate_service_intervals_are_counted_once_for_busy_wall(tmp_path):
    jobs = [job(0, sd_receive_complete_s=0., server_start_s=2., server_end_s=5.),
            job(1, sd_receive_complete_s=1., server_start_s=2., server_end_s=5.)]
    assert write(tmp_path, jobs)['server_busy_wall_s'] == 3.


def test_group_idle_optimizer_steps_and_partial_odd_group_are_not_doubled(tmp_path):
    jobs = [job(0, client_idle_wait_s=2., group_complete_s=13.,
                group_optimizer_step_start_s=12., group_optimizer_step_end_s=13.),
            job(1, samples=56, client_idle_wait_s=2., group_complete_s=13.,
                group_optimizer_step_start_s=12., group_optimizer_step_end_s=13.),
            job(2, group_size=1, client_optimizer_step_performed=True,
                client_idle_wait_s=3., group_complete_s=21.,
                group_optimizer_step_start_s=20., group_optimizer_step_end_s=21.)]
    result = write(tmp_path, jobs, {'round': 1, 'training_wall_s_debug_only': 30.,
                                  'worker_drain_elapsed_s': 29., 'server_updates': 3})
    assert result['jobs'] == 3 and result['client_visits'] == 312
    assert result['client_optimizer_steps'] == 2 and result['server_updates'] == 3
    assert result['client_idle_wait_s'] == 5.
    assert result['client_optimizer_wall_s'] == 2.
    assert result['round_end_wait_s'] == 8.
    clients = list(csv.DictReader((tmp_path / 'pipeline_events_by_client.csv').open()))
    assert len(clients) == 1 and clients[0]['client_optimizer_steps'] == '2'


def test_240_jobs_are_120_group_steps_and_keep_visit_counts(tmp_path):
    jobs = [job(index, job_id=[1, cid, index], client_id=cid,
        samples=56 if index == 23 else 128, inflight_on_forward=1 + index % 2,
        pending_graph_bytes=1024 * (1 + index % 2),
        pending_graph_count_after_backward=1 - index % 2)
        for cid in range(10) for index in range(24)]
    result = writer()(tmp_path, [{'round': 1, 'training_wall_s_debug_only': 100.,
        'server_updates': 240}], jobs, config={'n_clients': 10, 'slow_client_ids': [0, 5]})['pipeline']
    assert result['jobs'] == 240 and result['client_visits'] == 30000
    assert result['client_optimizer_steps'] == 120 and result['server_updates'] == 240
    assert result['max_inflight'] == result['max_pending_graphs'] == 2
    assert result['max_pending_graph_boundary_bytes'] == 2048
    clients = list(csv.DictReader((tmp_path / 'pipeline_events_by_client.csv').open()))
    assert len(clients) == 10
    assert all(row['jobs'] == '24' and row['client_optimizer_steps'] == '12' for row in clients)


def test_missing_measurements_are_null_not_zero_and_legacy_files_survive(tmp_path):
    original = [job(0)]
    saved = copy.deepcopy(original)
    legacy = {name: b'original\n' for name in ('per_job.jsonl', 'per_round.csv', 'job_timings.csv')}
    for name, content in legacy.items():
        (tmp_path / name).write_bytes(content)
    result = write(tmp_path, original)
    for key in ('server_busy_wall_s', 'server_cuda_model_busy_s', 'server_queue_depth_mean',
                'client_forward_uplink_overlap_s', 'client_idle_wait_s', 'gradient_pending_s',
                'round_end_wait_s', 'sender_queue_wait_s', 'max_sendqueue_bytes', 'upload_bytes'):
        assert result[key] is None, key
    assert original == saved
    assert all((tmp_path / name).read_bytes() == content for name, content in legacy.items())
    assert not list(tmp_path.glob('*.png')) and not list(tmp_path.glob('*.pdf'))
    json.dumps(result, allow_nan=False)


def test_gradient_pending_and_sender_wait_remain_separate_from_idle(tmp_path):
    jobs = [job(0, forward_compute_end_s=2., gradient_available_s=8.,
        sendqueue_enqueue_s=2.5, sender_dequeue_s=3., client_idle_wait_s=0.),
        job(1, forward_compute_end_s=5., gradient_available_s=9.,
        sendqueue_enqueue_s=5.5, sender_dequeue_s=7., client_idle_wait_s=1.25)]
    result = write(tmp_path, jobs)
    assert result['gradient_pending_s'] == 10.
    assert result['sender_queue_wait_s'] == 2.
    assert result['client_idle_wait_s'] == 1.25


def test_round_local_clocks_are_never_merged_between_rounds(tmp_path):
    first = job(0, group_size=1, client_optimizer_step_performed=True,
        sd_receive_complete_s=0., server_start_s=1., server_end_s=3.)
    second = {**first, 'round': 2, 'job_id': [2, 0, 0]}
    result = writer()(tmp_path, [{'round': i, 'training_wall_s_debug_only': 10.,
        'server_updates': 1} for i in (1, 2)], [first, second], config={})['pipeline']
    assert result['server_busy_wall_s'] == 4.
    assert result['server_queue_area_job_s'] == 2.
    assert result['server_queue_depth_mean'] == .1
    assert result['client_optimizer_steps'] == 2


def test_unknown_round_duration_does_not_invent_timeweighted_depth(tmp_path):
    result = write(tmp_path, [job(0, sd_receive_complete_s=0., server_start_s=2.,
        server_end_s=4.)], {'round': 1, 'server_updates': 1})
    assert result['training_wall_s'] is None
    assert result['server_queue_wait_mean_s'] == 2.
    assert result['server_queue_area_job_s'] == 2.
    assert result['server_queue_depth_mean'] is None
    assert result['server_idle_wall_s'] is None


def test_missing_last_group_completion_does_not_invent_round_end_wait(tmp_path):
    jobs = [job(0, group_size=1, client_optimizer_step_performed=True,
                group_complete_s=5.),
            job(2, group_size=1, client_optimizer_step_performed=True)]
    assert write(tmp_path, jobs)['round_end_wait_s'] is None


def test_actual_gpu_memory_and_cpu_queue_peaks_are_reported_separately(tmp_path):
    jobs = [job(0, max_sender_queue_depth=1, max_queued_cpu_bytes=300,
                pending_graph_bytes=200),
            job(1, sendqueue_depth_on_enqueue=2, sendqueue_cpu_bytes_on_enqueue=400,
                pending_graph_bytes=350)]
    result = write(tmp_path, jobs, {'round': 1, 'server_updates': 2,
        'gpu_peak_allocated_bytes': 1000, 'gpu_peak_reserved_bytes': 2000})
    assert result['gpu_peak_allocated_bytes'] == 1000
    assert result['gpu_peak_reserved_bytes'] == 2000
    assert result['max_sendqueue_depth'] == 2 and result['max_sendqueue_bytes'] == 400
    assert result['max_pending_graph_boundary_bytes'] == 350


def test_invalid_reversed_event_interval_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write(tmp_path, [job(0, sd_receive_complete_s=4., server_start_s=3.)])


def test_render_is_opt_in_and_creates_png_pdf_from_cpu_events(tmp_path):
    jobs = [job(0, group_size=1, client_optimizer_step_performed=True,
        client_forward_start_s=0., forward_compute_end_s=1., activation_ready_s=1.,
        sd_send_start_s=1., sd_receive_complete_s=2., server_start_s=2., server_end_s=3.)]
    write(tmp_path, jobs, {'round': 1, 'training_wall_s_debug_only': 10.,
        'test_accuracy': 30., 'elapsed_wall_s': 12., 'server_updates': 1}, render=True)
    for stem in ('pipeline_accuracy_by_round', 'pipeline_accuracy_by_elapsed',
                 'pipeline_queue_idle', 'pipeline_overlap_timeline'):
        for suffix in ('.png', '.pdf'):
            assert (tmp_path / (stem + suffix)).stat().st_size > 100
