"""라즈베리파이로 옮길 측정 묶음(신뢰성경진대회/pi_bundle/)을 PC에서 만든다.

묶음 하나로 PC와 라즈베리파이에서 같은 명령을 돌려 두 기기를 같은 날 직접 비교한다.

  python make_pi_bundle.py [--per-cell 4] [--out ../pi_bundle]

들어가는 것
- 코드: fast_preprocess.py, verify_fast_preprocess.py, bench_end_to_end.py,
        모터_특징추출_v2.py, rpi_stft_benchmark.py, bench_latency.py
- models/model_registry_fuse_rpm_ext.joblib (87채널, 커널 9,996개, 차종 3개)
- data/: 실제 스펙트로그램 PNG. 차종 x 클래스마다 per-cell 개 (기본 4개, 총 60구간, 180장)
- rpm_index_subset.json: 위 PNG의 회전수만
- can_sample.csv: 실제 CAN 파일 한 개 (회전수 평균만 쓴다)
- requirements.txt, run_pi.sh, run_pc.ps1
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]                                    # 매녀졸프
sys.path.insert(0, str(HERE))
import verify_fast_preprocess as vf  # noqa: E402

CAN = Path(r"D:\1.모터-감속기 데이터\IONIQ\DEMAG\0728\22_07_28_12_15_20_01_zsplit005_001.csv")

CODE = {
    HERE / "fast_preprocess.py": "fast_preprocess.py",
    HERE / "verify_fast_preprocess.py": "verify_fast_preprocess.py",
    HERE / "bench_end_to_end.py": "bench_end_to_end.py",
    PROJ / "모터_특징추출_v2.py": "모터_특징추출_v2.py",
    PROJ / "experiments/2026-09-15_CAN_fusion/src/rpi_stft_benchmark.py": "rpi_stft_benchmark.py",
    PROJ / "Motor-Reducer-Anomaly-Detection/sejeans/엣지벤치/bench_latency.py": "bench_latency.py",
    PROJ / "모델_ord_B/model_registry_fuse_rpm_ext.joblib": "models/model_registry_fuse_rpm_ext.joblib",
    CAN: "can_sample.csv",
}

REQUIREMENTS = """\
# 라즈베리파이 4, 64비트 Raspberry Pi OS (uname -m 이 aarch64), Python 3.10 이상
# sktime 과 scikit-learn 은 모델을 저장한 버전으로 고정한다. 다르면 모델 로드가 실패한다.
# numpy 는 고정하지 않는다 (엣지벤치/배포_87ch/실행법.md 참고).
sktime==1.1.0
scikit-learn==1.7.2
numba
joblib
pandas
pillow
matplotlib
"""

RUN_PI = """\
#!/usr/bin/env bash
# 라즈베리파이에서 실행: bash run_pi.sh
# 전체 약 1시간. 결과는 results/ 에 쌓인다. 중간에 끊겨도 끝난 단계의 결과는 남는다.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p results
"$PY" -c "import platform,sktime,sklearn,numpy,numba;print(platform.machine(),sktime.__version__,sklearn.__version__,numpy.__version__,numba.__version__)" | tee results/env.txt
vcgencmd measure_temp | tee -a results/env.txt || true
vcgencmd get_throttled | tee -a results/env.txt || true

echo "== 1. 모델 계산만 (8월 455 ms 재확인, 약 1분)"
"$PY" bench_latency.py --models ./models --threads 1 --batch 1 --reps 30 --out results/model_t1.json

echo "== 2. 처음부터 끝까지, 차량 경로 (약 12분)"
"$PY" bench_end_to_end.py --reps 30 \\
    --plans vehicle_orig_serial,vehicle_fast_serial,vehicle_fast_parallel3 \\
    --out results/end_to_end_vehicle.json

echo "== 3. 처음부터 끝까지, PNG 경로 (약 12분)"
"$PY" bench_end_to_end.py --reps 15 \\
    --plans png_orig_serial,png_fast_serial,png_fast_parallel3 \\
    --out results/end_to_end_png.json

echo "== 4. 개선 전처리 판정 일치와 시간, 실제 PNG 60구간 (약 20분)"
"$PY" verify_fast_preprocess.py --data ./data --rpm-index ./rpm_index_subset.json \\
    --model ./models/model_registry_fuse_rpm_ext.joblib --per-cell 4 \\
    --out results/verify.json

vcgencmd measure_temp | tee -a results/env.txt || true
vcgencmd get_throttled | tee -a results/env.txt || true
echo "완료. results/ 폴더를 PC로 가져온다."
"""

RUN_PC = """\
# PC에서 같은 묶음을 같은 날 실행: powershell -ExecutionPolicy Bypass -File run_pc.ps1
# 라즈베리파이와 같은 명령, 같은 입력이다. 결과는 results_pc/ 에 쌓인다.
# 매녀졸프\\.venv-lab6 의 python 을 쓴다 (sktime 1.1.0, scikit-learn 1.7.2).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$PY = if ($env:PYTHON) { $env:PYTHON } else { "..\\..\\.venv-lab6\\Scripts\\python.exe" }
New-Item -ItemType Directory -Force results_pc | Out-Null
& $PY -X utf8 bench_latency.py --models ./models --threads 1 --batch 1 --reps 30 --out results_pc/model_t1.json
& $PY -X utf8 bench_end_to_end.py --reps 30 --plans vehicle_orig_serial,vehicle_fast_serial,vehicle_fast_parallel3 --out results_pc/end_to_end_vehicle.json
& $PY -X utf8 bench_end_to_end.py --reps 15 --plans png_orig_serial,png_fast_serial,png_fast_parallel3 --out results_pc/end_to_end_png.json
& $PY -X utf8 verify_fast_preprocess.py --data ./data --rpm-index ./rpm_index_subset.json --model ./models/model_registry_fuse_rpm_ext.joblib --per-cell 4 --out results_pc/verify.json
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--out", default=str(HERE.parent / "pi_bundle"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for src, dst in CODE.items():
        d = out / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"!! 없음: {src}")
            continue
        shutil.copy2(src, d)
        print(f"복사 {dst}")

    rpm_idx = json.loads((PROJ / "rpm_index.json").read_text(encoding="utf-8"))
    samples = vf.pick_samples(rpm_idx, vf.DEFAULT_DATA, args.per_cell)
    base = Path(vf.DEFAULT_DATA)
    subset, total = {}, 0
    for v, cls, paths, rpm in samples:
        for p in paths:
            d = out / "data" / p.relative_to(base)
            d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists():
                shutil.copy2(p, d)
            total += d.stat().st_size
        stem = paths[0].name.removesuffix(".png")
        subset[stem] = rpm_idx[stem]
    (out / "rpm_index_subset.json").write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    print(f"표본 {len(samples)}구간, PNG {3 * len(samples)}장, {total / 1e6:.0f} MB")

    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "run_pi.sh").write_text(RUN_PI, encoding="utf-8", newline="\n")
    (out / "run_pc.ps1").write_text(RUN_PC, encoding="utf-8-sig")   # PowerShell 5.1 은 BOM 필요
    print(f"완료: {out}")


if __name__ == "__main__":
    main()
