# 실행 방법과 실험 설정

작성일: 2026-11-02. 모든 Python 명령은 컨테이너의 `/workspace/HSFL/src`에서 실행한다. 설치와 CIFAR10 준비는 [환경 설정](../02-portable-setup/README.md)을 따른다. 이 문서의 명령을 저장소를 clone했다는 이유만으로 자동 실행하지 않는다.

## 공통 학습 조건

CIFAR10 train50000/test10000 중 고정 seed42로 선택한30000장을 Client10명×3000장으로 분할한다. capacity-balanced class-wise Dirichlet alpha0.1이며 자연 quantity skew 대신 동일 용량을 강제한다. CIFAR-stem ResNet18/cut2, B128(마지막56), local epoch1, Adam lr0.001/wd0.0005, FP32·TF32 off, random initialization이다. Client와 Aux optimizer는 FedAvg 뒤 reset하고 공유 Server optimizer는 유지한다.

실행은 **단일 process**, Client별 thread/CUDA stream10개와 공유 Server thread/stream1개다. MPI를 여러 rank로 실행하면 거부한다. DataLoader num_workers0과 batch별 고정 증강 seed를 사용하지만 실제 FCFS 도착 순서는 OS/CUDA scheduling 때문에 seed만으로 완전히 같지 않다.

| Method | Client gradient | Aux | 서버 처리 |
|---|---|---|---|
| `splitfedv2` | 전원 Server gradient | 없음 | individual 또는 concat 비교 |
| `na_sfl_v1` | Slow0·5는 Aux, 나머지는 Server | 전원 학습·FedAvg | individual |
| `na_sfl_v2` | v1과 같음 | 전원 학습·FedAvg | 현재 FIFO buffer의 SD를 기다림 없이 최대512samples concat |
| `na_sfl_v3_proto` | Fast는 grad_SD, Slow0·5는 grad_Prototype | 없음 | individual FCFS |
| `na_sfl_v4_residual` | v3 + Slow PseudoCE 경로 | 없음 | individual FCFS |

v1 기본 YAML의 Aux gradient 수는0이며 논문의 모든 condition이 항상2명인 것은 아니다. Mininet 비교 YAML은 k2를 선택한다. v1/v2의 `--aux-client-count k`에서 k는 보조 gradient 사용자 수이며 Aux head 학습 개수가 아니다.

## 네트워크와 연산 목표

Mininet 설정은 Client별 독립 링크10개다. Slow는 Client0·5이며 UL20/DL50Mbps, Fast8명은 UL/DL100Mbps, RTT10ms, 추가 packet loss/jitter0이다. Server 공통 링크에는 rate cap을 걸지 않는다. 실제 TCP로 activation/prototype·gradient와 초기/FedAvg 모델도 전송한다.

`na_sfl_network_variable.yaml`, `na_sfl_v3_variable.yaml`, `na_sfl_v4_variable.yaml`은 각 client UL/DL에 독립 `U(-.1,.1)` 변동을 round별로 적용하고 round 안에서는 고정한다. Fast90–110, Slow UL18–22/DL45–55Mbps다. 네트워크 전용 seed42로 같은100round를 요청하면 같은 속도표/hash를 사용한다. actual HTB readback 및 scenario hash가 각 결과에 남는다. 고정 프리셋도 보존한다.

가정 Client compute target은 F0.3초와 B+Adam0.6초다. v1/v2는 Aux0.1초도 포함한다. partial batch는 표본 비율로 보정한다. 모델 계산은 실제 GPU에서 수행하고 target보다 일찍 끝난 phase만 추가 대기한다. 이는 독립 Jetson 장치의 실측 속도가 아니며 v3/v4에는 가짜 Aux delay를 추가하지 않는다.

## 먼저 사용할 작은 학습 명령

```bash
python run_exp_main.py --help
# 네트워크 없는 기존 v1 단일 조건을1round까지만 실행
python run_exp_main.py --aux-client-count 2 --stop-after-rounds 1
# 실제 Mininet + 전체30k장 Smoke
python run_exp_main.py --config configs/na_sfl_v3.yaml --smoke-test --rounds 1 --device 0
python run_exp_main.py --config configs/na_sfl_v4.yaml --smoke-test --rounds 2 --device 0
```

새 GPU 모델이면 `--allow-unprofiled-gpu`를 추가한다. GPU UUID 선택은 [설치 안내](../02-portable-setup/README.md#6-다른-nvidia-gpu)를 따른다. `--smoke-test`는 최대2round이며 자기 checkpoint를 finite/hash 확인 후 지워 기존 보관 모델을 유지한다. 실제 학습을 CPU에서 실행하지 않는다.

## ±10% 환경에서 v3 → v4 → v2 → v1을 같은 GPU에서 실행

아래는 사용자가 본 실험을 명시적으로 시작할 때의 명령이다. GitHub 게시 작업에서는 실행하지 않았다. 하나가 성공적으로 종료된 뒤 다음 명령을 실행한다. 각 명령은 새로운 초기값/seed42/분할에서 독립 시작하며 `--resume`을 사용하지 않는다.

```bash
python run_exp_main.py --config configs/na_sfl_v3_variable.yaml --rounds 100 --device 0
python run_exp_main.py --config configs/na_sfl_v4_variable.yaml --rounds 100 --device 0
python run_exp_main.py --config configs/na_sfl_network_variable.yaml --method na_sfl_v2 \
  --server-policy concat --rounds 100 --device 0
python run_exp_main.py --config configs/na_sfl_network_variable.yaml --method na_sfl_v1 \
  --server-policy individual --rounds 100 --device 0
```

v3/v4 variable 프리셋 기본은2round이며100round는 위와 같이 명시한다. v1/v2 network 프리셋 기본은100round이므로 작은 확인에는 `--rounds 1` 또는 `--smoke-test`를 추가한다. `--target-accuracy 70`은 TTA 집계 기준이고 해당 정확도에서 조기 종료하는 기능이 아니다. 모든 비교에는 같은 GPU/분할/초기화/학습·네트워크 seed를 사용한다.

기존 `--comparison`은 SplitFedV2±Concat와 v1/v2의 네 조건용이다. v3/v4를 자동 포함하지 않는다. `--sweep`은 기존 v1 Aux gradient 수0…10 실험용이며 v3/v4 실행에는 사용하지 않는다.

## Prototype/Pseudo ablation

```bash
python run_exp_main.py --config configs/na_sfl_v3.yaml --smoke-test --alignment-weight-beta 0
python run_exp_main.py --config configs/na_sfl_v4.yaml --smoke-test --pseudo-loss-weight-gamma 0
python run_exp_main.py --config configs/na_sfl_v4.yaml --smoke-test --no-pseudo-gradient-to-client
python run_exp_main.py --config configs/na_sfl_v4.yaml --smoke-test --no-residual-recenter
```

초기값 β=.1(MSE),γ=.25, residual64/class·FP32, pseudo4/class/min2, scale1, recenter=true다. 최적 수렴값을 확정한 설정이 아니다. γ=0은 extra pseudo forward도 생략하므로 동일 고정 Job 순서에서 v3와 gradient/BN/parameter가 같다.

## 결과·checkpoint·재개

각 condition의 `result_path` 아래 고유 run ID 폴더에 config.json/yaml, environment.json, partition.json, summary.json, round_metrics, job_timings, execution_boundaries, server_batches, source_snapshot/manifest 및 그래프가 남는다. v3/v4는 per_round.csv, per_client_round.csv, per_job.jsonl, global_prototype_stats.jsonl, v4 residual_bank_stats.jsonl, config_resolved.yaml,6개 plot도 제공한다. 완료 여부는 프로세스 존재만 보지 않고 status.json/summary.completed를 확인한다.

정식 실행은 최종 Client/Server/Aux/optimizer와 필요한 상태를 `checkpoints/final.pt` 하나로 묶고 프로젝트 전체 최신5개만 유지한다. 오래된 초과 checkpoint만 정리하며 로그/지표/설정/snapshot은 보존한다. Smoke 모델은 보관하지 않는다. 원본 연구 서버의 데이터/과거 결과/최신5개 모델은 이 저장소에 포함되지 않는다.

재개는 완전히 drain된 round 경계만 지원한다. 학습 종료/`--stop-after-rounds`로 저장된 같은 source/config의 final.pt를 사용한다. source/config·partition·optimizer reset·round/version·feature memory/RNG 계약을 검사하므로 코드 변경 뒤 과거 checkpoint를 임의 재개하지 않는다. 논리 Client batch 순서는 seed로 다음 round에 재구성하며 batch 중간 저장은 지원하지 않는다.

```bash
python run_exp_main.py --config configs/na_sfl_v4.yaml --rounds 100 --stop-after-rounds 2
# 위 결과의 실제 final.pt와 동일 config/source를 사용:
python run_exp_main.py --config configs/na_sfl_v4.yaml --rounds 100 \
  --resume /실제/run/checkpoints/final.pt
```

재개된 실행의 초기 모델 재전송도 wall에 포함될 수 있다. Smoke와 정식 실행의 config contract는 다르므로 `--smoke-test` 결과를 일반100round에 이어 붙이지 않는다.
