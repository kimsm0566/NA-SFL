# NA-SFL — Network-Aware Split Federated Learning

CIFAR10 / CIFAR-stem ResNet18에서 SplitFedV2와 **NA-SFLv1–v5**를 비교하는 연구 코드다. 최신 배포에는 **Prototype 제거 조건과 C 2 in-flight pipeline**도 포함한다. 단일 GPU의 Client별 thread/CUDA stream10개와 공유 Server1개, 실제 Mininet/TCP의 독립 Client 링크와 round별 전송률 변동을 사용한다.

이 저장소는 **최신 코드·환경 재현 자료·확정 결과 요약/그래프**를 제공한다. 데이터, 모델 가중치, 전체 원시 결과·개인 경로·과거 연구 PDF/압축 파일은 포함하지 않는다. 학습은 CUDA 필수이며 GPU 회귀 suite는 실행하지 않는다. [프로젝트 현황과 그래프](docs/11-05/07-project-status/README.md).

## 처음 실행할 때

새 Linux x86_64 서버에서 [상세 환경 설정](docs/11-02/02-portable-setup/README.md)을 순서대로 따른다. 호스트 NVIDIA 드라이버, Docker Engine/Compose, NVIDIA Container Toolkit이 먼저 필요하다.

```bash
git clone https://github.com/kimsm0566/NA-SFL.git
cd NA-SFL
cp .env.example .env
# .env의 NA_SFL_GPU를 사용할 GPU의 전체 UUID로 설정(nvidia-smi -L).
sudo docker compose build
sudo docker compose up -d
sudo docker exec na-sfl nvidia-smi -L
sudo docker exec -w /workspace/HSFL/src -e CUDA_VISIBLE_DEVICES= na-sfl \
  python checks/check_environment.py --network
sudo docker exec -w /workspace/HSFL/src -e CUDA_VISIBLE_DEVICES= na-sfl \
  python -m pytest -q -p no:cacheprovider tests --ignore=tests/test_fedoptima_model_data_cpu.py
```

CIFAR10을 `src/datasets/cifar10/cifar-10-batches-py/`에 배치한 뒤 `check_environment.py --network --data-path datasets`로 확인한다. 학습 중 자동 다운로드는 하지 않는다. 새 서버에서 공식 데이터를 받는 명령은 설치 문서4절에 있다.

데이터 준비 후에는 위 ignore 없이 전체 CPU 검사를 실행한다. 해당 파일은 실제 CIFAR10 분할·data identity를 검사한다.

실제 GPU 학습을 사용자가 시작할 때:

```bash
sudo docker exec -w /workspace/HSFL/src na-sfl \
  python run_exp_main.py --config configs/na_sfl_v3.yaml --smoke-test --rounds 1 --device 0
sudo docker exec -w /workspace/HSFL/src na-sfl \
  python run_exp_main.py --config configs/na_sfl_v4.yaml --smoke-test --rounds 2 --device 0
```

A5000/Quadro RTX5000 이외 NVIDIA GPU는 `--allow-unprofiled-gpu`가 필요하다. 다른 compute process가 있거나 VRAM 여유가80% 미만이면 계속 거부한다. GPU UUID를 명시하면 CUDA logical device와 `nvidia-smi` 점검 대상을 일치시킬 수 있다. 새 GPU의 아키텍처 지원·메모리·성능까지 검증한 옵션은 아니다.

## 문서

- [OS·GPU·Docker·Mininet 설치, 데이터, 문제 해결, exact image 이전](docs/11-02/02-portable-setup/README.md)
- [모든 method·Smoke·±10% 네트워크·100round·ablation·checkpoint 실행](docs/11-02/03-experiments/README.md)
- [코드 구조, v3/v4 Loss·gradient·Prototype/Residual·TCP 흐름](docs/11-02/04-architecture/README.md)
- [실제로 확인한 검사, 재현 한계, source/의존성 추적](docs/11-02/05-reproducibility/README.md)
- [v1~v5·Loss/BN·Aux 수·확정 결과와 곡선](docs/11-05/07-project-status/README.md)
- [FedOptima 외부 참조·adapter·지원 범위](docs/11-05/09-fedoptima-reference/README.md)
- [Python lock](requirements.txt), [Dockerfile](Dockerfile), [Compose](compose.yaml), [관측 환경](environment/reference.json)
- [의존 라이브러리와 데이터 출처](THIRD_PARTY_NOTICES.md)

## 기본 실험 조건

| 항목 | 설정 |
|---|---|
| 모델 / cut | CIFAR-stem ResNet18 / layer2 뒤 cut2 |
| 데이터 | CIFAR10, Client10×3000장, capacity-balanced Dirichlet α0.1 |
| seed / batch / local epochs | 42 / 128(마지막56) / 1 |
| optimizer | Adam lr0.001, weight_decay0.0005 |
| Client / Aux optimizer | round FedAvg 후 reset; Server Adam은 지속 |
| 통신 | Mininet Client별 독립 링크, Fast100/100Mbps·Slow0/5 UL20/DL50Mbps·RTT10ms |
| 변동 옵션 | round별 독립 ±10%, UL/DL·Client별 독립, round 안에서는 고정 |
| 정밀도 | FP32, TF32 off |

v1/v2는 모든 Client가 Aux를 학습하고 선택된 Client만 Aux gradient를 쓴다. v2는 받은 SD를 no-wait concat한다. **v3/v4는 Aux 없이** Fast가 SD를, Slow가 Prototype/classes/counts만 업로드한다. v4는 이전 완료 round의 Fast Residual로 Slow Pseudo-SD를 생성한다. 공유 Server는 Job당 한 번 업데이트한다.

**v5**(`na_sfl_proto_random_sd`)는 Fast 전체 SD, Slow0·5 전체 batch Prototype+Random SD10%
(B128→13/B56→6)를 한 HYBRID_JOB으로 전송한다. 전원 SD0.8+Proto0.2/MSEβ0.1/DualBN이며
Slow의 두 gradient는 같은 Client Forward graph에 적용한다. 평가/추론은 전원 원본 SD다.

`na_sfl_random_sd_only`는 Prototype을 제거하고 Fast 전체/Slow SD10%와 전원 SD CE1.0을 쓴다.
**C**(`na_sfl_trainer_sender_pipeline`)는 v5를 유지하고 같은 Client 가중치로2 Forward→
각1/2 gradient→1 Adam Step을 수행한다. 전체240 Server Steps/120 Client Steps/round다.
**B는 구현·실험하지 않았다.** A=v5와 C 차이에는 pipeline과 accumulation/update 빈도가 함께 포함된다.

```bash
# 사용자가 실제 GPU 학습을 시작할 때만 실행한다.
sudo docker exec -w /workspace/HSFL/src na-sfl python run_exp_main.py \
  --config configs/na_sfl_proto_random_sd.yaml --smoke-test --rounds 2 --device 0
sudo docker exec -w /workspace/HSFL/src na-sfl python run_exp_main.py \
  --config configs/na_sfl_pipeline_c.yaml --smoke-test --rounds 2 --device 0
```

## 저장소 구성

```text
Dockerfile, compose.yaml        새 서버용 환경 recipe
requirements.txt              활성 Python dependency closure 고정
environment/                  OS/Python/apt/base image 관측 기록
src/run_exp_main.py            유일한 학습 진입점
src/{models,data,utils}/       모델·분할·학습·TCP·Mininet·지표
src/configs/                  fixed/variable 및 v1~v5/SD-only/C 설정
src/reference_external/       MIT FedOptima 참조와 별도 adapter
src/checks/check_environment.py 읽기 전용 환경/데이터 확인
src/tests/                    GPU 없는 반복 CPU 검사
docs/11-02/                   설치·실험·구조·검증 문서
docs/11-05/, reports/          최신 방법·완료 결과 요약·곡선·그래프
```

새 환경을 만드는 recipe와 기존 환경을 binary 그대로 옮기는 방법을 구분한다. 이 게시 작업에서는 원래 컨테이너를 보존하며 CPU/정적 검증만 수행했다. 다른 서버의 새 이미지 build·새 GPU Smoke까지 이미 검증했다는 뜻은 아니다. 실제 도착 순서와 하드웨어 차이 때문에 동일 seed가 bitwise 학습/동일 wall을 보장하지 않는다.
