# NA-SFL 활성 소스

실행 경로는 `run_exp_main.py` 하나이며 기존 live_async Client별 thread/CUDA stream과 공유 Server를 재사용한다. 현재 모델은 CIFAR10용 CIFAR-stem ResNet18이고 cut1/2/3을 지원한다. SplitFedV2 및 NA-SFLv1/v2/v3/v4를 선택할 수 있다.

새 서버에서는 [루트 README](../README.md)와 [설치 문서](../docs/11-02/02-portable-setup/README.md)를 먼저 따른다. **모든 명령의 작업 디렉터리는 컨테이너의 `/workspace/HSFL/src`다.** 데이터/결과는 GitHub에 포함하지 않는다.

```bash
python run_exp_main.py --help
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
CUDA_VISIBLE_DEVICES= python checks/check_environment.py --network --data-path datasets
```

학습을 명시적으로 시작할 때만 다음을 실행한다.

```bash
python run_exp_main.py --config configs/na_sfl_v3.yaml --smoke-test --rounds 1 --device 0
python run_exp_main.py --config configs/na_sfl_v4.yaml --smoke-test --rounds 2 --device 0
```

v1/v2는 Aux gradient 사용자 수와 Aux 학습 수를 구분한다. v3/v4는 Aux 없이 Fast SD/Slow Prototype을 업로드하고 전원 Server gradient를 쓴다. variable 설정은 Client별 UL/DL 독립 ±10%·round 내 고정·네트워크 seed42다. 새 GPU는 `--allow-unprofiled-gpu`를 명시하며 busy/free-memory 보호와 CUDA 필수 정책은 유지한다.

[실험·CLI·설정·보관/재개](../docs/11-02/03-experiments/README.md), [모듈·수학·통신 구조](../docs/11-02/04-architecture/README.md), [검증·재현 범위](../docs/11-02/05-reproducibility/README.md)에 상세 내용을 모았다. GPU 회귀 suite는 실행하지 않는다. 기존 연구 서버의 과거 결과/문서/snapshot은 로컬에 보존되며 이 공개 배포에서 자동 복원하거나 학습을 재개하지 않는다.
