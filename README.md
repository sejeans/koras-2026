# 제17회 대학생 신뢰성 프로젝트 경진대회 작업 폴더

| 파일 | 내용 |
|---|---|
| 수상전략_보고서.md | 대회 조사, 역대 수상작 분석, 발표 구성안, 예상 질문 |
| 발표수치_정리.md | 대본 숫자 대조, 새로 계산한 신뢰성 지표, 하지 못한 실험, 대본 수정안 |
| scripts/ | 이번에 계산에 쓴 스크립트. 모두 재학습 없이 기존 모델과 예측 파일만 쓴다 |
| results/ | 계산 결과 원본(JSON, parquet) |
| cache/ | 임시 파일. 지워도 된다 |

재실행 순서 (시스템 파이썬 `C:\Program Files\Python311\python.exe`, scripts 폴더에서 실행)

1. `reliability_metrics_v3.py` : v3 시험 데이터 지표, 수 분
2. `eval_87ch_on_confirmatory_validation.py` : 87채널을 확증 검증 데이터에 평가, 약 10분, 디스크 임시 공간 약 3.5GB
3. `confirmatory_metrics.py` : 확증 분할 지표
4. `bench_preprocess_pc.py` : PC 전처리 시간, D:\TS 원본 PNG 필요
5. `print_summary.py` : 1번 결과 요약 출력
6. `confirmatory_test_paired.py` : 확증 시험 데이터 짝비교. AI모델/EV_Model_work/confirmatory_20261016/test/predictions.csv 필요(GPU 평가 약 2.5시간)


세 가지 모두 확인했고 파일은 수정하지 않았습니다. 두 체크포인트는 구조는 같지만 가중치 값은 다릅니다. 신뢰구간 상한은 **1.58**이 맞습니다.

## 1. all.pt와 deadline_v3/train/best.pt

`.venv-yolo`(torch 2.2.2) 환경에서 열었습니다. `all.pt`는 state_dict가 아니라 `Model3` 객체를 통째로 저장한 파일입니다. 그래서 학습 스크립트를 실행하지 않고, 빈 `Model3` 클래스를 대신 넣어서 불러왔습니다.

| 파일 | 바이트 크기 | 형식 | 텐서 수 | numel 합 |
|---|---|---|---|---|
| `all.pt` | 45,182,110 | `Model3` 객체 | 183 | **11,274,747** |
| `deadline_v3/train/best.pt` | 45,166,142 | dict (`model_state_dict` 키) | 183 | **11,274,747** |

추가로 두 파일의 키와 값을 직접 비교했습니다.
- 키 이름 일치: True
- 텐서 shape 일치: True
- 가중치 값 동일: **False**
- BN running 통계를 뺀 학습 파라미터 수: 11,264,766

**판정:** 두 파일은 같은 구조(ResNet18 + DeepLabV3Plus, 파라미터 수 동일)지만 서로 다른 학습 결과입니다. 크기는 15,968바이트 차이인데, 저장 형식(객체 전체 저장과 dict 저장)과 메타데이터 차이 때문입니다.

## 2. 정확도 차이(제안 − AI Hub)의 95% CI 상한

상한은 **1.58**입니다. `results/reliability_metrics_v3.json` 521~527행:

```json
    "ALL:accuracy": {
      "pctp": 0.5671195313473731,
      "ci95": [
        -0.43262788963624976,
        1.5798093835101077
      ]
    }
```

이 값은 `"difference_87ch_minus_AIHub"` 블록 안에 있습니다. 파일 전체에서 1.62는 나오지 않습니다. 969행에도 1.58대 값(`1.5845140175442511`)이 있지만, 이것은 게이팅 적용 87ch ECC10의 `wrong_fault_mode` 구간이라 이 질문과 상관없습니다.

## 3. 사전 등록 가설 3개

`holdout/PROTOCOL.md`는 없습니다. holdout 폴더에 있는 것은 `HOLDOUT_RESULT.md`뿐입니다. 사전 등록 문서는 `HOLDOUT_RESULT.md` 3행이 가리키는 [plan/HOLDOUT_PLAN.md](../../experiments/2026-09-15_CAN_fusion/plan/HOLDOUT_PLAN.md)입니다. 프로젝트에 있는 다른 PROTOCOL.md(`experiments/2026-10-16_confirmatory_noninferiority/`)는 별도의 비열등성 실험 문서이고, H1~H3 가설이 없습니다.

**HOLDOUT_PLAN.md (65~69행), 사전 등록 원문:**

| 가설 | 판정 기준 |
|---|---|
| **H1. 성능 재현** | TEST macro-F1(차종 평균)의 95% 세션 부트스트랩 구간이 CV 추정치 **0.97366을 포함**하면 재현으로 본다. 구간이 그보다 아래에 있으면 CV가 낙관적이었다고 본다. |
| **H2. 게이팅 일반화** | DEV에서 정한 p50 임계값을 TEST에 적용했을 때 **DEMAG 미검출률이 게이팅 없을 때의 절반 이하**(즉 TEST 전체 미검출률의 50% 이하)로 떨어지면 일반화된 것으로 본다. |
| **H3. CAN 결합 무효 재확인** | TEST에서 `all25`와 `baseline87`의 macro-F1 차이가 **±0.2%p 이내**면 "유의한 차이 없음"이라는 기존 판정이 유지된 것으로 본다. |

**[HOLDOUT_RESULT.md](../../experiments/2026-09-15_CAN_fusion/evidence/holdout/HOLDOUT_RESULT.md) (7~11행), 결과 원문:**

| 가설 | 기준 | 결과 | 판정 |
|---|---|---|---|
| H1 성능 재현 | TEST F1 95% 구간이 CV 0.97366 포함 | 구간 [0.96810, 0.98120], TEST 평균 0.97579 | **통과** (재현) |
| H2 게이팅 일반화 | DEV p50 게이트에서 DEMAG 미검출이 절반 이하 | 4.11% → 0.46% (기준 2.06%) | **통과** |
| H3 CAN 결합 무효 | all25와 baseline 차이 ±0.2%p 이내 | -0.058%p | **통과** |
