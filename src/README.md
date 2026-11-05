# 현재 NA-SFL 소스

학습 진입점은 `run_exp_main.py` 하나다. 명령은 컨테이너의 `/workspace/HSFL/src`에서 실행한다.
[루트 README](../README.md)와 [새 서버 설치](../docs/11-02/02-portable-setup/README.md)를 먼저 따른다.

```bash
python run_exp_main.py --help
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests \
  --ignore=tests/test_fedoptima_model_data_cpu.py
# CIFAR10 준비 후에는 ignore 없이 전체 tests를 실행한다.
```

SplitFedV2, NA-SFLv1/v2, `na_sfl_v3_proto`, `na_sfl_v4_residual`,
v5=`na_sfl_proto_random_sd`, `na_sfl_random_sd_only`, C=`na_sfl_trainer_sender_pipeline`를 지원한다.
C는 window2/accumulation2로 고정되어 있고 B는 제공하지 않는다.

```bash
python run_exp_main.py --config configs/na_sfl_proto_random_sd.yaml --smoke-test --rounds 2
python run_exp_main.py --config configs/na_sfl_pipeline_c.yaml --smoke-test --rounds 2
```

실제 모델 실행은 CUDA 필수다. 다른 GPU에는 `--allow-unprofiled-gpu`를 명시한다.
busy/VRAM 보호·모델1개/실험·전역최신5개 정책을 유지한다. 결과는 `results/YYYYMMDD_snake_case/`로 지정한다.
설치/import/CPU 검사만으로 학습을 자동 시작하지 않는다.

`--hybrid-experiment`, `--sd-only-experiment`, `--pipeline-experiment`는 완료된 로컬 기준선
run을 `--baseline-reference`로 받아 실제2roundSmoke 후 fresh100round를 실행한다.
config/초기값/데이터/분할/같은 GPU UUID/속도표/원시 artifact가 필요하다.
보고용 CSV만으로 기준선 검증이나 checkpoint 재개를 할 수 없다.

[실행·설정](../docs/11-02/03-experiments/README.md) · [구조](../docs/11-02/04-architecture/README.md) ·
[연구 결과](../docs/11-05/07-project-status/README.md) · [검증](../docs/11-02/05-reproducibility/README.md).
`reference_external/FedOptima`는 MIT 외부 참조이며 NA 실행기가 자동 학습시키지 않는다.
