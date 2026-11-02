# 실제 검증과 재현 범위

작성일: 2026-11-02.

## 이번 게시에서 실행한 검증

기존 `hsfl-keepalive` 컨테이너의 Python3.10.12/PyTorch2.4.0+cu121 환경에서 CUDA를 숨긴 전체 CPU 검사 **193개가 통과**했다. 기존186개에 GPU 이동성 검사7개를 추가했다. matplotlib/pyparsing deprecation warning14개는 기존 경고다.

공개할66개 파일만 따로 복사한 clean export에서도 데이터·과거 결과 없이193개가 통과했다(7.30초). 이 검사는 원래 연구 디렉터리의 숨은 파일/과거 코드에 의존하지 않는지 확인한 것이며 새 Docker image build 검증과 구분한다.

추가 검사는 실측하지 않은 GPU가 기본 거부되고 명시적 opt-in에서만 허용되는지, opt-in에서도 다른 process/메모리 보호를 유지하는지, CLI/config boolean 계약, CUDA_VISIBLE_DEVICES의 physical selector 대응, 존재하지 않는 logical index 거부, NVIDIA runtime이 UUID 한 개만 노출한 경우의 mask 유무를 확인한다. 실제 GPU나 CUDA context를 만들지 않는 검사다.

읽기 전용 `checks/check_environment.py`는 고정 package·Python·torch CUDA build·torchvision import 및 Mininet/tool/root를 확인하고, 선택적으로 이미 존재하는 CIFAR10을 `download=False`로 검사한다. 이 명령은 Mininet topology나 GPU model을 실행하지 않는다. 최종 실행 결과는 [environment/validation.json](../../../environment/validation.json)에 보관한다.

## 직전 구현 단계의 실제 Smoke

2026-11-01의 v3/v4 구현 단계에서 기존 GPU0 A5000과 실제 fixed Mininet 링크로 v3 1round와 v4 2round를 완료했다. Client10×3000장/round,240jobs/round, 마지막56장 batch를 처리했다. v4 round2의 Slow48jobs에서 Pseudo-SD992개를 생성했고 donor의 이전 round/동일 class·single in-flight·drain/FedAvg/memory swap을 감사했다.

v3 Test13.55%/110.350초, v4 Test11.45→21.47%/214.827초는 기능 Smoke이며 장기 수렴·TTA 개선을 뜻하지 않는다. 이 Smoke는 이동성 옵션을 추가하기 전 source snapshot의 기록이다. GPU 이동성 변경 뒤 새 GPU Smoke나100round를 실행하지 않았다. 원시 데이터/실험 artifact는 원래 연구 서버에 보존하고 이 공개 저장소에 재배포하지 않는다.

## 재현에 필요한 고정 요소

- `requirements.txt`, Docker base digest, Mininet commit, critical apt versions.
- 모델 구조/cut, 데이터 canonical 파일과 selected subset/partition, 모든 learning/augmentation seed.
- Client/Server/Aux optimizer 종류와 reset/persist 정책, BatchNorm buffer의 FedAvg 처리.
- 네트워크 전용 seed와 round 수, rate schedule hash, 실제 HTB readback.
- GPU 종류/driver, Python/CUDA/cuDNN, thread 제한, Client phase target.
- 같은 source snapshot/config 및 완전히 drain된 round 경계의 checkpoint/RNG/feature memory state.

동일 seed라도 live FCFS 도착 순서는 OS/CUDA scheduling의 영향을 받는다. 따라서 같은 초기값/분할을 검증하는 것과 bitwise 동일 trajectory를 주장하는 것을 구분한다. 모델 수학의 v3↔γ0v4 동일성은 고정 Job 순서의 CPU 수치검사로 확인했다.

## 포함하지 않은 것

데이터와 `.pt/.pth/.ckpt/.safetensors`, 전체 실험 결과·로그·과거 source snapshot, 개인 서버 작업 지침, 연구 PDF·외부 코드 archive와 인증 정보는 업로드하지 않는다. 기존 학습 모델 보관 정책은 실험별 final1개·프로젝트 전체 최신5개이며 로그/지표는 보존한다.

2026-11-01 요청했다가 취소한 v3→v4→v2→v1 대기열은 취소 상태다. 이 공개 저장소 게시로 다시 실행하지 않았다. GitHub에는 실행 예만 제공하며 import/clone/환경 검사만으로 학습을 시작하는 hook는 없다.

## 아직 직접 검증하지 않은 범위

새 Docker image의 전체 build, 새 컨테이너에서의 Mininet topology와 GPU 학습, 다른 GPU 아키텍처에서의 수렴/VRAM,100round 신규 비교 및 GPU checkpoint 재개는 이번 게시 작업에서 실행하지 않았다. 원래 컨테이너를 재사용하는 운영 규칙을 지키면서 recipe/의존성/import/CPU/CLI/출판 파일과 remote Git object를 검증한 범위를 명확히 보고한다.

`environment/observed-python-packages.json`, `observed-apt-packages.tsv`는 당시 전체 inventory이며 필수 설치 목록과 구분한다. binary 수준으로 기존 컨테이너를 보존하려면 [exact image 이전](../02-portable-setup/README.md#8-binary-환경-자체를-그대로-옮겨야-할-때) 절차가 별도로 필요하다.
