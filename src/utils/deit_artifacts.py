"""Shared NA-SFL checkpoint and artifact helpers retained from the existing runner."""
import csv
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch


def capture_rng(scheduler):
    state=np.random.get_state()
    device=torch.cuda.current_device()
    return {'python':random.getstate(),'numpy':[state[0],state[1].tolist(),*state[2:]],
            'torch_cpu':torch.get_rng_state(),'torch_cuda':{'device':device,'state':torch.cuda.get_rng_state(device)},
            'scheduler':scheduler.getstate()}


def restore_rng(state,scheduler):
    random.setstate(state['python'])
    n=state['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]))
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state(state['torch_cuda']['state'],device=state['torch_cuda']['device'])
    scheduler.setstate(state['scheduler'])


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value


def finite_tree(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point():
        if not torch.isfinite(value).all().item(): raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value,dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value,(tuple,list)):
        for v in value: finite_tree(v)


def atomic_checkpoint(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    finite_tree(value)
    try:
        with temporary.open('wb') as stream:
            torch.save(value,stream); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def json_file(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def csv_file(path,rows):
    rows=list(rows)
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    path=Path(path); temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    temporary.replace(path)


def gpu_snapshot(device=0):
    physical = str(device)
    runtime_visible = os.environ.get('NVIDIA_VISIBLE_DEVICES', '').strip()
    single_runtime_uuid = runtime_visible.startswith('GPU-') and ',' not in runtime_visible
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible = [item.strip() for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if item.strip()]
        if (type(device) is not int or not 0 <= device < len(visible)
                or not (visible[device].isdigit() or visible[device].startswith('GPU-'))):
            raise RuntimeError('requested logical GPU is unavailable in the visible device list (CUDA_VISIBLE_DEVICES); use a full GPU UUID')
        physical = visible[device]
        if single_runtime_uuid and physical.isdigit():
            if physical != '0':
                raise RuntimeError('requested GPU is outside the single visible NVIDIA runtime GPU')
            physical = runtime_visible
    elif single_runtime_uuid:
        if device != 0:
            raise RuntimeError('requested logical GPU is outside the single visible NVIDIA runtime GPU')
        physical = runtime_visible
    result=subprocess.run(['nvidia-smi','--id='+physical,
        '--query-gpu=uuid,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    v=next(csv.reader(result.stdout.splitlines(),skipinitialspace=True))
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    return {'uuid':v[0],'name':v[1],'total_bytes':int(v[2])*1024**2,'used_bytes':int(v[3])*1024**2,
        'free_bytes':int(v[4])*1024**2,'temperature_c':float(v[5]),'utilization_percent':float(v[6]),
        'compute_processes':[line for line in apps.stdout.strip().splitlines() if line.startswith(v[0]+',')]}
