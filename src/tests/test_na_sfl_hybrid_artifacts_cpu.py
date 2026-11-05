"""Hybrid reporting and baseline-identity checks without model execution."""
import csv
import json
from copy import deepcopy

import pytest


def test_hybrid_reporting_counts_selected_sd_and_float_bytes_separately(tmp_path):
    from utils.na_sfl_proto_artifacts import write_proto_artifacts
    cfg={'method':'na_sfl_proto_random_sd','n_clients':1,'slow_client_ids':[0],
        'random_sd':{'sample_ratio':.1,'sample_seed':42},'metrics':{'target_accuracies':[70.]}}
    job={'job_id':(1,0,0),'job_kind':'HYBRID_JOB','samples':56,'client_id':0,
        'selected_sd_count':6,'sample_ratio':.1,'proto_class_ids':[0,2],'proto_counts':[40,16],
        'activation_bytes':(6+2)*8*4,'gradient_bytes':(6+2)*8*4,
        'sd_payload_bytes':300,'sd_wire_bytes':400,'gradient_payload_bytes':280,
        'gradient_wire_bytes':380,'client_complete_s':9.}
    rounds=[{'round':1,'test_accuracy':55.,'elapsed_wall_s':12.,'training_wall_s_debug_only':10.,
        'round_wall_s_debug_only':11.,'worker_drain_elapsed_s':10.,'server_updates':1}]
    out=write_proto_artifacts(tmp_path,rounds,[job],[],config=cfg)
    saved=json.loads((tmp_path/'per_job.jsonl').read_text())
    assert saved['sd_count']==6
    assert out['actual_sd_count']==6 and out['actual_proto_count']==2
    assert out['total_hybrid_jobs']==1
    assert out['upload_float_bytes']==out['download_float_bytes']==256
    assert out['upload_bytes']==400 and out['download_bytes']==380
    local=list(csv.DictReader((tmp_path/'per_client_round.csv').open()))[0]
    assert local['samples']=='56' and local['sample_ratio']=='0.1'
    assert local['actual_sd_count']=='6' and local['actual_proto_count']=='2'


def test_matched_hybrid_config_only_changes_method_sampling_and_output():
    from utils.na_sfl_config import load_config, DEFAULTS
    from utils.na_sfl_hybrid_experiment import validate_matched_config
    base=load_config(overrides={'method':'na_sfl_v3_proto','rounds':100,
        'bn_policy':'dual_running_stats_shared_affine',
        'prototype':{'fast_sd_weight':.8,'fast_proto_weight':.2}})
    hybrid=load_config(overrides={**{k:base[k] for k in DEFAULTS},
        'method':'na_sfl_proto_random_sd','random_sd':{'sample_ratio':.1,'sample_seed':42}})
    validate_matched_config(base,hybrid)
    for field in ('seed','lr','batch_size','samples_per_client','bn_policy'):
        changed=deepcopy(hybrid)
        changed[field]={'seed':43,'lr':.01,'batch_size':64,'samples_per_client':2500,
            'bn_policy':'shared_original'}[field]
        with pytest.raises(ValueError):validate_matched_config(base,changed)


def test_matching_completion_uses_exact_last10_and_null_tta(tmp_path):
    from utils.na_sfl_hybrid_experiment import accuracy_record
    rows=[{'round':r,'test_accuracy':62.453,'elapsed_wall_s':r*100.} for r in range(1,101)]
    summary={'initial_evaluation':{'test_accuracy':10.,'elapsed_wall_s':1.},
        'launch_wall_s_debug_only':10020.,'upload_bytes':120,'download_bytes':100,
        'upload_float_bytes':110,'download_float_bytes':90,'server_updates':24000}
    record=accuracy_record('hybrid',tmp_path,summary,rows)
    assert record['last10_mean_exact']=='62.453'
    assert record['tta_70'] is None and record['first_ge_70'] is None
    assert record['total_wall_time']==10020.
    rows[-1]['test_accuracy']=70.
    record=accuracy_record('hybrid',tmp_path,summary,rows)
    assert record['first_ge_70']==100 and record['tta_70']==10000.


def test_model_traffic_is_included_once_and_separate_from_job_bytes(tmp_path):
    from utils.na_sfl_hybrid_experiment import model_traffic
    (tmp_path/'network').mkdir()
    (tmp_path/'network/network_metadata.json').write_text(json.dumps({'model_traffic':[
        {'direction':'upload','wire_bytes':100,'payload_bytes':80},
        {'direction':'download','wire_bytes':150,'payload_bytes':120},
        {'direction':'download','wire_bytes':150,'payload_bytes':120}]}))
    out=model_traffic(tmp_path)
    assert out=={'model_upload_bytes':100,'model_download_bytes':300,
        'model_upload_payload_bytes':80,'model_download_payload_bytes':240}


def test_network_identity_ignores_observed_apply_time_and_horizon_hash():
    from utils.na_sfl_hybrid_experiment import validate_rate_rows
    base={'round':1,'clients':[{'client_id':0,'uplink_mbps':20.1,'downlink_mbps':49.}],
        'kernel_rates':[{'client_id':0,'uplink_mbps':20.1,'downlink_mbps':49.}],
        'apply_wall_s':.06,'rate_scenario_sha256':'100-round-hash'}
    smoke={**base,'apply_wall_s':.13,'rate_scenario_sha256':'two-round-hash'}
    validate_rate_rows([{'network_rates':smoke}],[{'network_rates':base}])
    for key in ('round','clients','kernel_rates'):
        changed=deepcopy(smoke)
        if key=='round':changed[key]=2
        else:changed[key][0]['uplink_mbps']=18.
        with pytest.raises(ValueError):validate_rate_rows([{'network_rates':changed}],[{'network_rates':base}])
        del changed[key]
        with pytest.raises(ValueError):validate_rate_rows([{'network_rates':changed}],[{'network_rates':base}])


def test_smoke_recovery_rejects_changed_computation_or_any_main_launch(tmp_path):
    from utils.na_sfl_hybrid_experiment import validate_recovery_contract
    old={'utils/na_sfl_prototype.py':'model-old','utils/na_sfl_hybrid_experiment.py':'gate-old',
        'tests/test_na_sfl_hybrid_artifacts_cpu.py':'tests-old'}
    now={**old,'utils/na_sfl_hybrid_experiment.py':'gate-new',
        'tests/test_na_sfl_hybrid_artifacts_cpu.py':'tests-new'}
    state={'status':'failed','active_condition':'smoke_test','active_child_pid':None,
        'error':'actual Hybrid per-round rates differ from baseline'}
    validate_recovery_contract(tmp_path,state,old,now)
    changed={**now,'utils/na_sfl_prototype.py':'different-loss'}
    with pytest.raises(ValueError):validate_recovery_contract(tmp_path,state,old,changed)
    (tmp_path/'run_100round').mkdir()
    with pytest.raises(ValueError):validate_recovery_contract(tmp_path,state,old,now)
