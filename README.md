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
