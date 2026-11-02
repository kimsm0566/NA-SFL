# 새 서버에서 환경 재현하기

작성일: 2026-11-02. 이 문서는 새 Linux 서버용이다. 원래 연구 서버의 `hsfl-keepalive` 컨테이너를 지우거나 교체하지 않는다.

## 1. 재현 기준과 필요한 자원

| 항목 | 기준 |
|---|---|
| 플랫폼 | Linux x86_64, Ubuntu 22.04 계열 |
| Python | 3.10.12 |
| PyTorch / torchvision | 2.4.0+cu121 / 0.19.0+cu121 |
| CUDA runtime / torch cuDNN wheel | 12.1 / 9.1.0.70 |
| NumPy / Pillow / PyYAML | 1.26.4 / 10.4.0 / 6.0.2 |
| matplotlib / mpi4py / pytest | 3.9.4 / 4.0.3 / 8.3.5 |
| Open MPI | 4.1.2 (단일 실행기의 import 호환성용; MPI 분산 학습 아님) |
| Mininet Python | 2.3.1b4, commit `6eb8973c0bfd13c25c244a3871130c5e36b5fbd7` |
| OS Mininet 도구 | Ubuntu mininet2.3.0 패키지의 `mnexec` 등 + 위 Python 소스 버전 |
| 링크 도구 | iproute2 5.15.0-1ubuntu2.2, bridge-utils 1.7-1ubuntu3 |
| 기준 GPU / driver | RTX A5000 24GB 또는 Quadro RTX5000 16GB / 535.247.01 |
| Docker shm / CPU worker thread 제한 | 8GiB / OMP·MKL 각2 |
| 여유 디스크 | 이미지·wheel·데이터·결과를 위해 최소25GiB 이상 권장 |

전체 관측 패키지는 [environment](../../../environment/)에 기록했다. `requirements.txt`는 현재 사용 의존성과 그 하위 의존성의 설치 버전을 고정한다. 전체 관측 목록에는 삭제한 과거 방법의 패키지도 있으므로 전부 설치할 필요는 없다. 원래 로컬 이미지 `hsfl-research-wifi:20261024`는 Docker Hub에서 받을 수 있는 공개 이미지가 아니다.

Dockerfile은 공개 CUDA base를 digest로 고정하고, Python dependency와 Mininet commit을 명시한다. 핵심 apt 패키지도 기존 버전으로 고정한다. Ubuntu 저장소가 해당 버전을 제공하지 않으면 빌드가 실패하며 다른 버전으로 조용히 바꾸지 않는다. 이 경우 8절의 이미지 이동 또는 패키지 archive 확보가 필요하다. base에 포함된 나머지 OS 패키지나 host kernel/driver까지 binary 동일하다고 보장하지 않는다.

## 2. 호스트 준비: Docker와 NVIDIA runtime

이미 관리자가 Docker와 NVIDIA runtime을 설치했다면 이 절의 설치 명령을 반복하지 않고 `nvidia-smi`, `docker version`, `docker compose version`으로 확인한다. 호스트 CUDA toolkit 설치는 필수가 아니며, 호환되는 NVIDIA 드라이버는 필요하다. 기준 서버에서는535.247.01을 사용했다.

아래는 **새 Ubuntu22.04 서버**의 Docker 설치 예다. 이미 설치된 Docker/Snap Docker를 강제로 삭제하는 절차는 포함하지 않는다.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/na-sfl-docker.asc
sudo install -m 0644 /tmp/na-sfl-docker.asc /etc/apt/keyrings/docker.asc
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu jammy stable' \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

NVIDIA Container Toolkit 설치 및 Docker runtime 등록:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/na-sfl-nvidia.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/na-sfl-nvidia.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi
sudo docker info
sudo docker compose version
```

Docker 재시작은 그 서버의 기존 컨테이너에 영향을 줄 수 있으므로 관리자가 허용한 시점에 실행한다. 드라이버가 없는 새 서버는 먼저 서버 GPU에 맞는 NVIDIA 드라이버를 설치하고 필요하면 재부팅한다. 근거와 최신 플랫폼별 차이는 [Docker 공식 Ubuntu 안내](https://docs.docker.com/engine/install/ubuntu/)와 [NVIDIA Toolkit 공식 안내](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)를 따른다.

## 3. 저장소 clone과 이미지 build

```bash
git clone https://github.com/kimsm0566/NA-SFL.git
cd NA-SFL
cp .env.example .env
nvidia-smi -L
```

`.env`의 `NA_SFL_GPU=0`을 사용하려는 물리 GPU의 **전체 UUID**로 바꾸는 것을 권장한다. 예: `NA_SFL_GPU=GPU-...`. 다른 GPU는 runtime에 노출하지 않는다. 숫자 GPU 순서는 호스트/runtime 설정에 따라 달라질 수 있다.

```bash
sudo docker compose build
sudo docker compose up -d
sudo docker exec na-sfl nvidia-smi -L
sudo docker exec na-sfl python --version
```

`compose.yaml`은 `runtime: nvidia`, `privileged`, host network, seccomp/AppArmor unconfined,8GiB shm을 사용한다. Mininet의 namespace·veth·bridge·HTB/netem 작업에 root/capability가 필요하다. `/var/run/netns`와 checkout을 bind mount한다. 현재 NA는 독립 namespace 안의 LinuxBridge를 사용하며 물리 NIC·호스트 route·NetworkManager를 설정하지 않는다. 과거 WiFi용 `/sys`·hwsim·OVS/wmediumd는 현재 NA에 필요하지 않아 recipe에 복원하지 않았다. `mn -c`나 hwsim unload를 정상 종료 절차로 쓰지 않는다.

코드는 이미지 안에 복사하지 않고 `/workspace/HSFL`에 checkout을 연결한다. 따라서 컨테이너를 종료/재시작해도 데이터와 결과는 clone한 디렉터리에 남는다. 다른 checkout/컨테이너를 병행할 경우 container name과 프로젝트 경로를 구분한다.

Compose가 없는 서버의 동등한 runtime 예(먼저 위 Dockerfile을 build):

```bash
sudo docker build -t na-sfl:torch2.4.0-cu121 .
sudo docker run -d --name na-sfl --init --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NA_SFL_CONTAINER_NAME=na-sfl \
  --privileged --network host --shm-size=8g \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -v "$PWD:/workspace/HSFL" -v /var/run/netns:/var/run/netns \
  -w /workspace/HSFL/src na-sfl:torch2.4.0-cu121
```

GPU0 대신 UUID를 쓸 수 있다. 이 컨테이너의 Python 실행 환경은 host의 conda/venv와 분리한다. 현재 연구 서버에서는 이 생성 명령을 실행하지 않는다.

## 4. CIFAR10 배치

데이터를 GitHub나 Docker image에 포함하지 않는다. 학습은 `download=False`이며 다음 파일을 요구한다.

```text
src/datasets/cifar10/cifar-10-batches-py/
  batches.meta
  data_batch_1 ... data_batch_5
  test_batch
```

이미 보유한 공식 CIFAR10 Python 파일을 이 경로로 복사한다. **새 서버에서 새로 받을 경우에만** 다음 명령을 명시적으로 실행한다.

```bash
sudo docker exec -w /workspace/HSFL/src na-sfl python -c \
  "from torchvision.datasets import CIFAR10; CIFAR10('datasets/cifar10', train=True, download=True); CIFAR10('datasets/cifar10', train=False, download=True)"
```

torchvision은 archive와 각 파일의 checksum을 검사한다. 원본: [CIFAR 공식 페이지](https://www.cs.toronto.edu/~kriz/cifar.html), Python archive MD5 `c58f30108f718f92721af3b95e74349a`. CIFAR100은 현재 학습에 필요하지 않다. torchvision pretrained weight 다운로드도 하지 않는다.

## 5. 설치·데이터·CPU 검사

다음은 GPU 연산을 하지 않고 패키지/import·도구·데이터를 확인한다. `--network`는 도구/root/import만 확인하며 topology를 만들지 않는다.

```bash
sudo docker exec -w /workspace/HSFL/src -e CUDA_VISIBLE_DEVICES= na-sfl \
  python checks/check_environment.py --network --data-path datasets
sudo docker exec na-sfl python -m pip check
sudo docker exec -w /workspace/HSFL/src -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 na-sfl \
  python -m pytest -q -p no:cacheprovider tests
sudo docker exec -w /workspace/HSFL/src -e CUDA_VISIBLE_DEVICES= na-sfl \
  python run_exp_main.py --help
```

환경 검사에 버전 차이가 나오면 package inventory 및 lock와 비교한다. GPU 회귀 suite는 없으며 실행하지 않는다. 실제 모델 학습 Smoke는 아래 실험 문서의 명령을 사용자가 명시적으로 실행할 때만 시작한다.

## 6. 다른 NVIDIA GPU

기본 실행은 실측한 A5000/Quadro RTX5000만 허용한다. 다른 NVIDIA GPU는 `--allow-unprofiled-gpu`로 명시적으로 선택한다. CUDA 필수·다른 compute process 없음·VRAM80% 이상 여유 조건은 계속 적용한다. 이 옵션은 새 GPU의 메모리 용량이나 PyTorch2.4의 해당 아키텍처 지원을 보장하지 않는다. GPU를 바꾸면 시간 비교의 하드웨어 조건도 바뀐다.

정확한 GPU 선택을 위해 전체 UUID를 사용한다. 예:

```bash
sudo docker exec -w /workspace/HSFL/src \
  -e CUDA_VISIBLE_DEVICES=GPU-여기에_실제_UUID -e CUDA_DEVICE_ORDER=PCI_BUS_ID na-sfl \
  python run_exp_main.py --config configs/na_sfl_v3.yaml --smoke-test --rounds 1 \
  --device 0 --allow-unprofiled-gpu
```

`--device 0`은 **보이는 GPU 목록의 logical index0**이다. 해당 UUID는 `nvidia-smi -L`의 실제 값으로 바꾼다. 코드의 busy/메모리 점검도 CUDA_VISIBLE_DEVICES에 해당하는 물리 GPU를 조회한다. MIG는 이번 recipe/선택 검증 범위 밖이다.

## 7. 흔한 문제

| 증상 | 확인/해결 |
|---|---|
| `unsupported, busy or insufficient free memory` | 기본 GPU 모델 제한, 다른 compute process,80% 여유를 확인. 새 모델만 opt-in으로 허용하며 다른 사람의 프로세스를 종료하지 않는다. |
| `CUDA is required` | runtime 설치, NVIDIA 드라이버, Compose GPU UUID, 컨테이너 `nvidia-smi` 확인. CPU 학습 fallback 없음. |
| `Operation not permitted`, namespace 생성 실패 | root/privileged/seccomp/AppArmor 및 `/var/run/netns` 권한 확인. 호스트 전체 NetworkManager를 끄거나 기존 namespace를 지우지 않는다. |
| `mnexec`/`brctl`/`ethtool` 없음 | apt-runtime.lock와 Docker build 로그 확인. Python mininet만 pip 설치하면 실행 파일이 부족할 수 있다. |
| Mininet version2.3.0으로 import | OS apt 모듈보다 `/usr/local`의 pinned Python 소스2.3.1b4가 우선인지 확인. |
| HTB plaintext/경고 | 현재 parser는 iproute2의 JSON/plaintext 차이를 처리한다. round 기록의 kernel rate/ceil을 확인한다. |
| CIFAR10 not found/corrupted | 디렉터리 중첩과 canonical batch 파일/체크섬 확인. 학습 중 자동 download하지 않는다. |
| exact apt version not found | 새 버전으로 무작정 바꾸지 않고 reference inventory와 비교; 아래 full image 이전 방식이나 해당 Ubuntu package archive를 사용한다. |
| `resume ... contract mismatch` | 같은 source snapshot/config/partition인지 확인. 과거 결과를 최신 소스로 임의 재개하지 않는다. |

## 8. binary 환경 자체를 그대로 옮겨야 할 때

GitHub recipe는 기록된 의존성/소스를 재구성한다. 기존 컨테이너의 모든 OS 파일까지 동일하게 옮기려면 원본 이미지의 별도 전송이 필요하며 수십GB 이미지는 GitHub에 넣지 않는다.

원래 서버의 이미지에는 나중에 컨테이너에 추가한 패키지가 빠질 수 있다. 관리자가 허용하면 **기존 컨테이너를 교체하지 않고** 새 이름으로 상태를 commit한 뒤 save한다. 새 서버에서는 load하고, 위 runtime 설정의 이미지명만 그 태그로 바꾼다.

```bash
# 관리자 승인 하에 수동 실행하는 예. 이번 게시 작업에서는 실행하지 않았다.
docker commit hsfl-keepalive na-sfl-exact:20261102
docker save na-sfl-exact:20261102 | gzip > na-sfl-exact-20261102.tar.gz
sha256sum na-sfl-exact-20261102.tar.gz
# 파일을 새 서버로 별도 전송한 뒤:
gunzip -c na-sfl-exact-20261102.tar.gz | docker load
```

bind-mounted dataset/source/results는 image commit/save에 포함되지 않는다. `git clone` 및 데이터 복사가 별도로 필요하다. host kernel/driver도 image에 포함되지 않는다. 실험 결과의 source/config/hash 및 실제 wall을 함께 비교해야 한다.

## 검증 범위

본 게시 작업에서는 **기존 컨테이너**에서 CPU 검사·환경 검사·CLI·파일목록을 검증한다. 원래 환경을 보존하기 위해 새 이미지 build/새 컨테이너 기동/새 서버 GPU Smoke는 실행하지 않았다. 따라서 Dockerfile이 다른 서버에서 이미 끝까지 build·학습 검증됐다고 주장하지 않는다. 실행한 최종 검증은 [재현성과 검증 기록](../05-reproducibility/README.md)에 남긴다.
