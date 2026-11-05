# 실행 방법과 실험 설정

갱신일: 2026-11-05. 모든 Python 명령은 컨테이너의 `/workspace/HSFL/src`에서 실행한다. 설치와 CIFAR10 준비는 [환경 설정](../02-portable-setup/README.md)을 따른다. 저장소를 clone했다는 이유만으로 학습을 자동 실행하지 않는다.

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
| `na_sfl_proto_random_sd` (v5) | Fast 전체/Slow Prototype+SD10% | 없음 | individual / DualBN |
| `na_sfl_random_sd_only` | Prototype 없이 Fast 전체/Slow SD10%, CE1.0 | 없음 | individual |
| `na_sfl_trainer_sender_pipeline` (C) | v5 그대로 + 2batch 평균 누적 | 없음 | Server Job당1 Step, Client2 Jobs당1 Step |

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

실험 묶음은 **`results/YYYYMMDD_설명/`** 형식이다. 날짜는 최초 시작일(Asia/Seoul), 설명은 영문 `snake_case`다. 예: `--result-path results/20261105_na_sfl_v5_run01`. 준비·검증·분석에도 같으며 같은 날의 추가 구분자는 설명 뒤에 붙인다.

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

## 최신 v5 / SD-only / C

v5는 Slow0·5의 전체 batch class mean Prototype과 Random SD10%(B128→13/B56→6)를
하나의 HYBRID_JOB으로 전송한다. Fast8은 전체 SD다. 전원 SD0.8+Proto0.2/MSEβ0.1/DualBN이며
Slow의 두 gradient는 같은 Forward graph에 적용한다. 추론은 전원 원본 SD다.
SD-only는 Prototype을 제거하고 전원 SD CE1.0을 쓴다.

```bash
python run_exp_main.py --config configs/na_sfl_proto_random_sd.yaml --smoke-test --rounds 2
python run_exp_main.py --config configs/na_sfl_random_sd_only.yaml --smoke-test --rounds 2
python run_exp_main.py --config configs/na_sfl_pipeline_c.yaml --smoke-test --rounds 2
# 단독 fresh100round:
python run_exp_main.py --config configs/na_sfl_proto_random_sd.yaml --rounds 100 \
  --result-path results/20261105_v5_reference
python run_exp_main.py --config configs/na_sfl_pipeline_c.yaml --rounds 100 \
  --result-path results/20261105_c_standalone
```

기준선 일치와 실제 중첩/통신량까지 감사하는 관리 경로는 아래와 같다.
`/path/to/...`는 **같은 GPU에서 완료한 로컬 run 디렉터리**로 바꾼다.
새 서버에는 과거 원시 기준선이 없으므로 로컬 기준선을 먼저 학습해야 한다. 공개 CSV는 이를 대체하지 않는다.

```bash
python run_exp_main.py --config configs/na_sfl_proto_random_sd.yaml --hybrid-experiment \
  --baseline-reference /path/to/completed_v3_dual_bn_run --result-path results/20261105_v5_comparison
python run_exp_main.py --config configs/na_sfl_random_sd_only.yaml --sd-only-experiment \
  --baseline-reference /path/to/completed_v5_run --result-path results/20261105_sd_only_comparison
python run_exp_main.py --config configs/na_sfl_pipeline_c.yaml --pipeline-experiment \
  --baseline-reference /path/to/completed_v5_run --result-path results/20261105_c_comparison
```

각 관리자는 실제2roundSmoke 검증 후 fresh100round를 시작한다.
완료100round/config/manifest/source snapshot/초기값/분할/GPU UUID/속도표를 검증한다.
Smoke가 실패하면 main을 시작하지 않는다. **B는 구현·실험하지 않는다.**

C는 Client당 단일 UL Sender·별도 DL Receiver를 유지하고 Trainer만 CUDA/autograd/optimizer를 소유한다.
같은 Client 파라미터로2 Forward를 만든 뒤 각각 기존 공유 Server FCFS의 별도 Job/Step으로 처리한다.
두 gradient를 각1/2로 누적한 뒤 Client Adam1회다. B128/B56도 batch동일가중,
홀수 마지막 singleton은1/1이다. Client당24 Jobs/12 Steps, 전체240 Server Steps/120 Client Steps/round다.
두 Job은 서로 다른 Server version의 gradient일 수 있다. graph/in-flight/Send/Completion은 최대2개다.

실제 model Forward CUDA 이벤트의 보수적 UL 중첩 하한>0과 입력/그룹/RNG/BN/Prototype/bytes/drain/
finite checkpoint를 확인한다. host phase/pacing 중첩과 구분하며 Server receive-to-service queue와 내부 queue,
gradient pending과 Trainer idle도 구분한다. Server wall idle은 GPU utilization이 아니다.
**A=v5와 C 차이는 pipeline과 accumulation/update빈도의 합성 효과**이며 B가 없어 인과 분리하지 않는다.

## v3 Loss·BN 조건부 실험

```bash
# 같은 loss의 Dual BN 조건 직접 실행:
python run_exp_main.py --config configs/na_sfl_v3_variable.yaml --rounds 100 \
  --fast-sd-weight 0.8 --fast-proto-weight 0.2 --bn-policy dual_running_stats_shared_affine
# 기존 v3 0.5:0.5/shared100round 기준 자동 분기:
python run_exp_main.py --config configs/na_sfl_v3_variable.yaml --rounds 100 \
  --v3-bn-ablation --baseline-reference /path/to/completed_original_v3_run
```

첫0.8:0.2의 마지막10round 평균이 기준선보다 엄격히 높으면 BN Freeze/Dual도0.8:0.2,
그렇지 않으면0.5:0.5로 실행한다. Freeze는 Proto BN 통계만 eval이고 gradient는 유지한다.
Dual BN은 affine/optimizer 공유, running statistics만 SD/Proto로 분리하며 평가 domain은 SD다.
