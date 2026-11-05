# 검증 결과와 재현 범위

갱신일: 2026-11-05.

## 공개 사본에서 실행한 검사

실행 중인 소스를 바꾸지 않고 검토 완료 C 사본을 export했다. runtime/config는 동결 소스와 동일하다.
배포 사본의 Protocol CPU 테스트만 다른 테스트의 sys.path 부작용에 의존하지 않도록 import를 보완했다.

- 데이터·과거 결과 없는 사본: **490 CPU tests passed**, 기존 경고14개,22.92초.
- 기존 공식 CIFAR10을 잠시 연결한 전체 사본: **505 passed**, 기존 경고14개,27.13초.
- 고정53개 Python package와 Mininet/tool/root/CIFAR import·읽기 점검 통과.
- CLI help와 pip check 통과. 새 GPU model/Mininet topology를 게시 검증에서 실행하지 않았다.
- 데이터 연결은 검사 후 삭제했다. export에 데이터·모델·과거 run·개인 agent 파일은 없다.

데이터 없는 검사는 test_fedoptima_model_data_cpu.py를 제외한다. 해당15개는 CIFAR10과 초기 모델/분할/data identity를 확인하며 모델 Forward/Backward를 실행하지 않는다.
[설치·검사](../02-portable-setup/README.md) · [구조화한 검증](../../../environment/validation.json).

## 별도 승인된 실제 학습과 구분

v1~v5·SplitFed·Loss/BN·SD-only100round는 별도 승인 실행이며 게시 작업이 새로 학습/재평가한 결과가 아니다.
공개 reports는 완료된 원시 곡선/요약을 CPU로 정리하고 입력 hash를 보존했다.
[완료 결과](../../11-05/07-project-status/README.md).

C는 별도 승인 GPU1 queue가 SD-only 종료 후 시작했다. 실제2roundSmoke에서480Jobs/240ClientSteps/480ServerSteps,
60,000visits, bounded2, 실제CUDA Forward(B2)/UL(B1)의 보수적 중첩>0, bytes/BN/Prototype/drain을 검증했다.
이는 기능/정합성 확인이며100round 수렴·TTA 또는 장기 메모리 안정성 검증이 아니다.
[시각이 기록된 상태](../../../reports/20261105_research_summary/status_snapshot.json)를 따른다.
B는 구현·실험하지 않았고 새 clone이 원래 서버의 대기열을 실행하는 hook는 없다.

## 재현 조건과 해석

Python/PyTorch/torchvision/Mininet·OS 도구/driver, GPU/VRAM, seed/초기값/정밀도,
공식 데이터/30k 분할, Client pacing, optimizer reset·Server persist, BN,
HTB readback/network seed/horizon, source/config/완전 drain된 checkpoint를 함께 맞춘다.
live FCFS 순서는 OS/CUDA scheduling의 영향을 받아 같은 seed만으로 bitwise trajectory를 보장하지 않는다.

SplitFed/v1/v2 고정 링크와 v3~v5의±10% 링크, SplitFed A5000/NA RTX5000 조건 차이가 있다.
A=v5/C 차이에는 pipeline과 accumulation/update빈도가 함께 바뀐다. SD-only/v5는 Proto 제거와 SD loss계수0.8→1.0도 함께 바뀐다.
TTA70은 최초 평가 crossing이며 지속70% 유지 기준이 아니다. Server wall idle과 실제 GPU utilization,
host phase와 CUDA 모델 이벤트 overlap을 구분한다.

## 포함·제외와 미검증 범위

소스/config/반복 CPU 검사/환경 recipe/실행 문서, MIT FedOptima 참조와 adapter,
개인 경로를 제거한 결과 요약·곡선·PNG를 포함한다. 데이터/checkpoint/전체 Job 로그/source snapshot/
개인 AGENTS/PDF/인증은 제외한다. 모델 보관은 실험당 final1개/전역최신5개며 원시 로그·지표는 보존한다.

새 Docker build/새 컨테이너 GPU·Mininet 실행, 다른 GPU의 메모리/수렴, 과거 checkpoint 재개는
이번 게시 검증에서 실행하지 않았다. 원래 컨테이너/전역 패키지도 바꾸지 않았다.
NA는 새 GPU opt-in을 제공하지만 FedOptima 환경 일치 adapter에는 원래 GPU UUID guard가 남아 있다.
[외부 코드 범위](../../11-05/09-fedoptima-reference/README.md)를 확인한다.
