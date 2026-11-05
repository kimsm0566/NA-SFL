# FedOptima 외부 참조와 환경 일치 adapter

고정 원본은 [blessonvar/FedOptima](https://github.com/blessonvar/FedOptima)의
commit `3eba341c7f4ca4fafc450ec2714a05484a14fdb8`이다.
[MIT 고지](../../../src/reference_external/FedOptima/LICENSE)와 원본 README/config/그림을 보존한다.
`hsfl/`와 `run.py --hsfl-experiment` opt-in은 현재 연구 환경에 추가한 코드다.
NA 단일 실행기가 이를 자동 학습시키지 않는다.

## 확인한 범위

원본 CIFAR10/VGG5/Client2 실행에서 Validation10회·마지막45.00%를 확인했다.
기본 val_rate1은 별도 Test를 끄므로45%를 독립 Test 정확도로 보고하지 않는다.
원본 loopback TCP/sleep bandwidth 실행은 현재 NA의 Mininet 실험과 다른 조건이다.

환경 일치 실험은 CIFAR-stem ResNet18/cut2,10×3000/seed42/동일 Dirichlet 분할,
실제 Mininet Fast100/100·Slow0/5 20/50Mbps·RTT10/±10%, Client F.3/Aux+B+Adam.7초를 연결했다.
원본 비동기 집계/counter scheduler/Aux를 재사용하며 단일 GPU의 Client thread/stream을 쓴다.
송신 종료를 위해 유한 SD credit1과 전송 허용 race 처리를 보완했으므로 논문 원본 그대로의 재현은 아니다.

종료는 최근10개30k-visit 평가의 평균 Test가 엄격히70%를 초과하거나 학습 wall18000초다.
고정100round가 아니다. 완료된5시간 실행에서489개 평가, 마지막59.56%·최고66.65%·
마지막10평균59.075%를 기록했고70% 조건에 미달해 시간 제한으로 종료했다.
Local14,696,936 visits, Server6,499,584 visits/50,778 updates다.

## 다른 서버의 지원 경계

root [requirements](../../../requirements.txt)를 설치하며 외부 폴더 원본 requirements는 설치하지 않는다.
NLTK corpus나 pretrained weights는 현재 CIFAR10 실험에 필요하지 않다.

주 NA 실행 경로는 새 GPU UUID와 `--allow-unprofiled-gpu`를 지원한다.
이 **FedOptima 환경 일치 adapter는 원래 연구 GPU UUID를 guard로 고정**한 실험 코드다.
다른 서버에서 `--hsfl-experiment`를 그대로 실행하면 거부한다. 이 배포는 동결 소스를 보존하며
guard를 해제한 새 GPU 실행까지 검증하지 않았다. 이식 시 GPU 선택/metadata/동등한 보호를
함께 조정하고 실제 Smoke를 새로 검증해야 한다. NA의 portable 실행과 이 외부 adapter를 구분한다.

관련 CPU 검사는 Protocol/Control/Network/모델·데이터 계약을 확인한다.
`test_fedoptima_model_data_cpu.py`는 공식 CIFAR10 로컬 파일이 필요하며 모델 Forward/Backward는 실행하지 않는다.
다른 CPU 검사는 데이터 없는 설치에서도 실행할 수 있다.
