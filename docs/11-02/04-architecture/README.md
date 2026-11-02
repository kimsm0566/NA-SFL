# 코드 구조와 gradient 흐름

작성일: 2026-11-02. 현재 활성 구현만 설명한다. 과거 MicroSFL/DeiT/Jetson/WiFi 코드를 복원하지 않는다.

## 실행·모델·데이터

`src/run_exp_main.py`가 유일한 학습 진입점이다. `utils/option.py`와 `na_sfl_config.py`가 설정을 검증하고, `na_sfl_run.py`가 입력/환경/결과/최종 모델을 관리한다. `NASFLTrainer`는 native split node·loader·평가·FedAvg·checkpoint를 재사용한다. Client는10개 독립 모델, Server는 단일 공유 모델이며 Client별 Server 복제나 Server FedAvg는 없다.

`models/resnet18.py`는 torchvision BasicBlock/ResNet18을 CIFAR stem3×3/stride1/no-maxpool/head10으로 구성한다. cut1/2/3은 layer1/2/3 뒤다. cut2의 sample activation은 `[128,16,16]` FP32다. Client와 Server를 합친 원래 모델의 실제 SD 경로만 global test evaluation에 사용한다.

`data/na_sfl_data.py`는 seed 고정 subset과 class-wise Dirichlet preference를 IPF로 동일3000장 quota에 맞춘다. 마지막56장 batch를 버리지 않는다. 증강/loader 순서는 Client/round/epoch/batch key로 고정하지만 live FCFS는 bitwise 재현되지 않는다.

## v1/v2와 비동기 worker

`utils/deit_async.py`의 Client thread/stream과 Server thread/stream이 실제 readiness queue/Future를 사용한다. Server gradient Client는 반환 gradient 적용 후 다음 batch로 진행한다. Aux gradient Client는 유한 TCP 송신 credit와 queue backpressure 안에서 local update/다음 batch를 진행할 수 있다. 소유 graph/Future/transport drain 뒤 Client/Aux FedAvg와 local optimizer reset을 수행한다.

v1은 individual SD 처리다. v2는 이미 수신된 FIFO prefix를 최대512samples까지 기다림 없이 concat하고 한 번 backward/optimizer step한다. Server gradient를 받는 각 SD에는 전체 loss의 평균 배율을 local mean으로 보정하여 반환한다. Concat에서는 SD 수와 Server optimizer step 수가 다르므로 수렴 비교 시 분리한다.

## v3: Prototype gradient

Fast8명은 SD `z`와 labels만 보낸다. 서버가 batch 내 class mean `p_c=mean(z_j:y_j=c)`를 미분 가능하게 계산한다. 같은 Server classifier가 SD→Prototype 순서로 forward하며 loss는 다음과 같다.

```text
Fast: 0.5 L_SD + 0.5 (L_ProtoCE + beta L_Align)
Slow: 1.0 (L_ProtoCE + beta L_Align)
L_ProtoCE = sum_c n_c CE(server(p_c), c) / sum_c n_c
```

Slow0·5는 로컬에서 class mean graph를 유지하고 Prototype/classes/counts만 보낸다. 반환 grad_p를 그 graph에 적용하면 각 같은 class SD에는 grad_p/n_c가 분배된다. Fast의 반환 grad_z에는 원본 SD CE와 서버 class mean 경로가 합쳐진다. Aux head/학습/FedAvg와 Concat은 없다.

Alignment는 이전 round Global Prototype의 valid class만 대상으로 class count 가중 MSE를 쓴다. 선택적으로 squaredL2를 사용할 수 있다. Target은 detach되고 round 내 고정되며 Alignment만으로 Server parameter는 갱신되지 않는다. 모든 Fast/Slow job의 class count로 global memory를 누적한 뒤 drain·Client FedAvg 후 한 번 갱신한다. 없는 class의 이전 값/valid는 유지한다. 첫 round alignment는0이다.

## v4: 이전 round Residual + Pseudo-SD

Fast batch의 class count>=2일 때 `r_j=z_j-own_batch_class_mean`을 detach해 CPU FP32로 보관한다. 현재 round builder와 이전 완료 round snapshot을 분리한다. Slow job은 snapshot의 같은 class donor만 복원 없이 최대4개·최소2개 선택한다. 선택 residual subset을 recenter하여 `pseudo=p_slow+scale*(r-mean(r))`를 생성하므로 class별 pseudo 평균은 p_slow다.

```text
Slow v4: L_ProtoCE + beta L_Align + gamma L_Pseudo
```

PseudoCE는 eligible class별 pseudo CE 평균을 해당 Slow class count로 가중하고 eligible counts 합으로 정규화한다. donor 부족은 다른 class로 대체하지 않고 Proto-only fallback한다. 첫 round bank는 비어 있다. 기본 pseudo gradient는 Client까지 전달하며 false ablation은 해당 pseudo base만 detach한다. γ=0은 생성과 추가 classifier/BN forward를 모두 생략한다.

Residual은 class별64개, cut2/10class snapshot 최대80MiB와 builder 별도 최대80MiB다. seed/round/class/job/sample의 SHA256 keyed priority 중 최소64개를 유지하는 bounded uniform reservoir를 사용한다. model/data RNG를 소비하지 않고 후보 도착 순서에 독립적이다. donor ID/priority/version/coverage/bytes를 기록한다.

## TCP/Mininet과 측정

`na_sfl_network.py`가 소유 namespace·LinuxBridge·독립 TCLink10개를 만들고 `na_sfl_transport.py`가 길이-prefix/JSON descriptor/SHA256 tensor framing을 처리한다. Fast typed upload는 SD/labels, Slow는 protos/classes/counts다. 다운로드는 대응 gradient와 Slow classes다. dtype/shape/finite/count/identity 및 immutable origin byte를 확인하고 실제 decoded tensor를 계산에 사용한다.

최종 payload read 시각, 검증 완료, queue admission/dequeue, Server 연산, gradient 수신, Client update 완료를 구분한다. full-frame 수신 ticket 순서로 admission하므로 검증 시간이 다른 Job이 먼저 도착한 Job을 추월하지 않는다. 순수 kernel 도착 시각이나 wire-only PHY 지연을 측정했다고 주장하지 않는다.

UL/DL bytes는 application payload+header+8byte prefix이며 TCP/IP/L2 overhead는 제외한다. 초기/FedAvg model 전송은 별도 network metadata에 남긴다. `server_idle=round_train-sum(server_model_compute)`는 통신/큐/관리 시간을 포함하는 모델 비연산 시간이며 물리 GPU 유휴율과 다르다. 전체 compute는 공유 GPU에서 실행하므로 독립 실제10대 Client 성능으로 해석하지 않는다.

## 모듈 위치

| 파일 | 역할 |
|---|---|
| `utils/na_sfl.py`, `sflv2.py` | 기존 Client/Server node·Aux update·평가 |
| `utils/na_sfl_train.py`, `deit_async.py` | round 조율·실제 worker·FedAvg/drain |
| `utils/na_sfl_batch.py` | no-wait concat·phase pacing·membership 검증 |
| `utils/na_sfl_proto_client.py`, `na_sfl_prototype.py` | retained class mean graph·prototype memory/loss/Server |
| `utils/na_sfl_residual.py` | residual snapshot/builder·pseudo·checkpoint 검증 |
| `utils/na_sfl_proto_protocol.py`, `na_sfl_transport.py` | typed payload·실제 비동기 TCP |
| `utils/na_sfl_network.py` | Mininet·속도표·HTB readback·model transfer·소유 자원 정리 |
| `utils/na_sfl_artifacts.py`, `na_sfl_proto_artifacts.py` | 지표·그래프·TTA·checkpoint 보관 |
| `utils/deit_artifacts.py`, `sflv2_metrics.py` | checkpoint/RNG/CSV/JSON·source snapshot/manifest |
| `checks/check_environment.py`, `tests/` | 읽기 전용 환경 확인·반복 CPU 검사 |
