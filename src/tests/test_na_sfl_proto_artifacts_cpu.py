"""Pure reporting contracts; representative records, no model or CUDA execution."""
import csv
import importlib
import json

import pytest
import yaml


def writer():
    module = importlib.util.find_spec('utils.na_sfl_proto_artifacts')
    assert module is not None, 'prototype artifact writer has not been implemented'
    return importlib.import_module('utils.na_sfl_proto_artifacts').write_proto_artifacts


def config(method='na_sfl_v3_proto'):
    return {'method': method, 'num_clients': 2, 'slow_client_ids': [0],
            'prototype': {'alignment_weight_beta': .1},
            'v4': {'enabled': method == 'na_sfl_v4_residual', 'pseudo_gradient_to_client': True},
            'metrics': {'target_accuracies': [70., 65., 90.]}}


def job(cid, n, loss, **extra):
    return {'job_id': (1, cid, 0), 'round': 1, 'client_id': cid, 'local_batch_index': 0,
        'job_kind': 'PROTO_JOB' if cid == 0 else 'SD_JOB', 'samples': n,
        'labels': [2] * n, 'proto_class_ids': [2], 'proto_counts': [n],
        'sd_payload_bytes': n * 10, 'sd_header_bytes': 3, 'sd_wire_bytes': n * 10 + 3,
        'gradient_payload_bytes': n * 8, 'gradient_header_bytes': 4,
        'gradient_wire_bytes': n * 8 + 4,
        'sd_send_start_s': 1., 'sd_send_return_s': 1.5, 'sd_receive_complete_s': 3.,
        'enqueue_s': 3., 'server_start_s': 4., 'server_end_s': 7.,
        'server_model_compute_s': 1., 'server_prototype_compute_s': .2,
        'client_forward_compute_s': .1, 'prototype_compute_s': .05,
        'gradient_send_start_s': 7., 'gradient_receive_complete_s': 8.,
        'gradient_wait_s': 100., 'client_gradient_wait_s': 200.,
        'client_backward_start_s': 8., 'client_compute_done_s': 9.,
        'client_complete_s': 9. + cid, 'L_SD': loss if cid else 0.,
        'L_ProtoCE': loss, 'L_Align': .2, 'L_total': loss,
        'server_optimizer_step_id': cid + 1, **extra}


def rounds():
    return [{'round': 1, 'test_accuracy': 66., 'test_ce_loss': 1.2,
        'elapsed_wall_s': 15., 'training_wall_s_debug_only': 12.,
        'round_wall_s_debug_only': 14., 'server_updates': 2,
        'global_prototype_stats': {'round': 1, 'version': 1,
            'valid_class_mask': [False, False, True], 'sample_counts_by_class': [0, 0, 8],
            'prototype_norm': [0., 0., 2.]}}]


def read_csv(path):
    return list(csv.DictReader(path.open()))


def test_wire_totals_partial_batch_losses_idle_and_optimizer_steps(tmp_path):
    (tmp_path / 'partition.json').write_text(json.dumps({'labels_by_client': [[2, 2], [2] * 6]}))
    jobs = [job(0, 2, 2.), job(1, 6, 4.)]
    summary = writer()(tmp_path, rounds(), jobs, [{'round': 1, 'fedavg_s': .7}], config=config())
    assert summary['upload_bytes'] == 86 and summary['download_bytes'] == 72
    assert summary['upload_payload_bytes'] == 80 and summary['upload_header_bytes'] == 6
    assert summary['download_payload_bytes'] == 64 and summary['download_header_bytes'] == 8
    assert summary['total_sd_jobs'] == summary['total_proto_jobs'] == 1
    assert summary['server_optimizer_steps'] == 2
    row = read_csv(tmp_path / 'per_round.csv')[0]
    assert float(row['L_total']) == 3.5
    assert float(row['L_SD']) == 3.
    assert float(row['round_train_s']) == 12.
    assert float(row['round_time_s']) == 14.
    assert float(row['server_model_compute_s']) == 2.
    assert float(row['server_idle_s']) == 10.
    assert float(row['server_idle_pct']) == pytest.approx(100 * 10 / 12)
    assert json.loads(row['aggregation_record']) == {'round': 1, 'fedavg_s': .7}
    assert json.loads((tmp_path / 'data_partition_summary.json').read_text())['labels_by_client'][0] == [2, 2]
    assert yaml.safe_load((tmp_path / 'config_resolved.yaml').read_text()) == config()
    assert not (tmp_path / 'residual_bank_stats.jsonl').exists()


def test_client_wait_excludes_upload_and_round_tail_uses_local_completion(tmp_path):
    writer()(tmp_path, rounds(), [job(0, 2, 2.), job(1, 6, 4.)], [], config=config())
    clients = read_csv(tmp_path / 'per_client_round.csv')
    assert [r['client_group'] for r in clients] == ['slow', 'fast']
    assert [float(r['client_round_end_wait_s']) for r in clients] == [3., 2.]
    for row in clients:
        assert float(row['gradient_wait_s']) == 5.
        assert float(row['upload_wire_time_s']) == 2.
        assert float(row['upload_wait_or_block_s']) == .5
        assert float(row['gradient_download_time_s']) == 1.
        assert float(row['server_queue_wait_s']) == 1.
        assert float(row['client_backward_update_s']) == 1.
    saved = [json.loads(line) for line in (tmp_path / 'per_job.jsonl').read_text().splitlines()]
    assert saved[0]['gradient_wait_s'] == 5.
    assert saved[0]['client_gradient_wait_s'] == 200.
    assert saved[0]['sd_count'] == 0 and saved[1]['sd_count'] == 6
    assert saved[0]['upload_payload_bytes'] == 20
    assert saved[0]['job_id'] == [1, 0, 0]
    assert saved[0]['upload_done_ts'] == 3.


def test_queue_wait_excludes_server_staging_after_dequeue(tmp_path):
    writer()(tmp_path, rounds(), [job(0, 2, 2., dequeue_s=3.25)], [], config=config())
    saved = json.loads((tmp_path / 'per_job.jsonl').read_text().splitlines()[0])
    assert saved['server_queue_wait_s'] == .25


def test_round_tail_uses_worker_origin_instead_of_earlier_trainer_start(tmp_path):
    raw = rounds()
    raw[0]['worker_drain_elapsed_s'] = 11.
    writer()(tmp_path, raw, [job(0, 2, 2.), job(1, 6, 4.)], [], config=config())
    clients = read_csv(tmp_path / 'per_client_round.csv')
    assert [float(r['client_round_end_wait_s']) for r in clients] == [2., 1.]


def test_targets_include_initial_crossing_and_unreached(tmp_path):
    summary = writer()(tmp_path, rounds(), [], [], config=config(),
                       initial_evaluation={'test_accuracy': 71., 'test_ce_loss': 2., 'elapsed_wall_s': 1.})
    assert summary['targets'][0] == {'accuracy': 70., 'status': 'reached', 'round': 0, 'elapsed_wall_s': 1.}
    assert summary['targets'][2] == {'accuracy': 90., 'status': 'not_reached', 'round': None, 'elapsed_wall_s': None}
    assert summary['best_round'] == 0 and summary['best_accuracy'] == 71.
    assert summary['final_round'] == 1 and summary['final_accuracy'] == 66.


def test_v4_first_round_fallback_and_raw_donors_survive(tmp_path):
    cfg = config('na_sfl_v4_residual')
    row = rounds()[0]
    row['residual_bank_stats'] = {'bank_version': 1, 'residual_bank_size_by_class': [0, 0, 2],
                                  'class_donor_coverage': [False, False, True], 'memory_bytes': 16}
    jobs = [job(0, 2, 2., L_Pseudo=0., pseudo_samples_generated=0,
                pseudo_fallback_classes=[2], pseudo_donors=[], bank_version=0,
                pseudo_fallback_reason='empty_bank', pseudo_gradient_to_client=True)]
    result = writer()(tmp_path, [row], jobs, [], config=cfg)
    assert result['pseudo_samples_generated'] == 0
    assert result['pseudo_fallback_classes'] == 1
    assert result['pseudo_loss'] == 0.
    bank = json.loads((tmp_path / 'residual_bank_stats.jsonl').read_text())
    assert bank['bank_version'] == 1 and bank['memory_bytes'] == 16
    saved = json.loads((tmp_path / 'per_job.jsonl').read_text())
    assert saved['pseudo_fallback_reason'] == 'empty_bank' and saved['pseudo_donors'] == []
    assert result['pseudo_gradient_to_client'] is True


def test_empty_artifacts_and_six_plot_outputs(tmp_path):
    result = writer()(tmp_path, [], [], [], config=config('na_sfl_v4_residual'), render=True)
    assert result['best_round'] is None and result['final_round'] is None
    assert result['server_optimizer_steps'] == 0 and result['avg_classes_per_slow_proto'] is None
    assert len(read_csv(tmp_path / 'per_round.csv')) == 0
    assert len(read_csv(tmp_path / 'per_client_round.csv')) == 0
    assert (tmp_path / 'per_job.jsonl').read_text() == ''
    for name in ('accuracy_vs_round', 'accuracy_vs_elapsed', 'loss_vs_round', 'per_round_time',
                 'server_idle', 'communication_bytes'):
        assert (tmp_path / 'plots' / (name + '.png')).read_bytes().startswith(b'\x89PNG')


def test_repeated_forward_metadata_does_not_duplicate_optimizer_steps_and_render_rows(tmp_path):
    row = rounds()[0]
    row.pop('server_updates')
    jobs = [job(0, 2, 2., server_forward_calls=3, server_optimizer_step_id=17),
            job(1, 6, 4., server_forward_calls=2, server_optimizer_step_id=18)]
    result = writer()(tmp_path, [row], jobs, [], config=config(), render=True)
    assert result['server_optimizer_steps'] == 2
    assert result['targets'][1]['round'] == 1 and result['targets'][1]['elapsed_wall_s'] == 15.
    assert len(list((tmp_path / 'plots').glob('*.png'))) == 6
    assert json.loads((tmp_path / 'global_prototype_stats.jsonl').read_text())['sample_counts_by_class'] == [0, 0, 8]


def test_v4_missing_pseudo_fields_and_weighted_pseudo_losses(tmp_path):
    cfg = config('na_sfl_v4_residual')
    first = writer()(tmp_path, rounds(), [job(0, 2, 2.)], [], config=cfg)
    assert first['pseudo_samples_generated'] == first['pseudo_fallback_classes'] == 0
    assert first['pseudo_loss'] is None
    result = writer()(tmp_path, rounds(), [job(0, 2, 2., L_Pseudo=0.),
        job(1, 6, 4., L_Pseudo=2., pseudo_samples_generated=4,
            pseudo_donors=[{'class_id': 2, 'indices': [1, 3, 4, 8]}])], [], config=cfg)
    assert result['pseudo_loss'] == 1.5 and result['pseudo_samples_generated'] == 4
    saved = [json.loads(line) for line in (tmp_path / 'per_job.jsonl').read_text().splitlines()]
    assert saved[-1]['pseudo_donors'] == [{'class_id': 2, 'indices': [1, 3, 4, 8]}]
