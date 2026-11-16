"""차량 경로(센서 신호 -> STFT -> 특징 87채널) 전처리 시간을 PC에서 잰다. 모델 계산은 빼고 잰다.

가상 신호: 25,000 Hz x 10초 x 센서 3개 (rpi_stft_benchmark.py 의 synth_window).
STFT: nfft 2560, hop 125 -> (1281 -> 1280) x 2000, 학습 이미지 크기와 같다.
dB 값을 0~1 로 맞춰 에너지 맵으로 쓴다. AI Hub 이미지의 스케일과 같다는 보장은 없어서
시간 측정 전용이다(이 값을 지금 모델에 넣어 판정하면 안 된다).

비교: 원본 특징 코드 직렬 / 빠른 특징 코드 직렬 / 빠른 특징 코드 센서 3개 병렬(STFT 포함)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "2026-09-15_CAN_fusion" / "src"))
import fast_preprocess as fp  # noqa: E402
import rpi_stft_benchmark as sb  # noqa: E402

fx = fp.fx
FS, SEC, NFFT, W, H = 25000, 10.0, 2560, 2000, 1280
N = int(FS * SEC)
HOP = N // W
WINDOW = np.hanning(NFFT).astype(np.float32)
RPM = 3000.0
REPS = int(os.environ.get("REPS", 30))


def stft_energy(x):
    db = sb.stft_magnitude(x, NFFT, HOP, W, WINDOW)          # (1281, 2000), 행0 = 저주파
    lo, hi = float(db.min()), float(db.max())
    e = ((db - lo) / max(hi - lo, 1e-6)).astype(np.float32)[::-1]   # 행0 = 고주파
    rows = np.linspace(0, e.shape[0] - 1, H).round().astype(np.int32)
    return np.ascontiguousarray(e[rows])


def orig_features(e):
    fx.extract_timeseries_vectorized(e)
    fx.extract_ext_features(e)
    fx.extract_ext_features(fx.to_order_axis(e, RPM, n_out=fx.ORD_BINS))


def main():
    data = sb.synth_window(N, 3, FS)
    chans = [np.ascontiguousarray(data[:, c]) for c in range(3)]
    fp.warmup()

    def run_orig():
        for x in chans:
            orig_features(stft_energy(x))

    def run_fast():
        for x in chans:
            fp.sensor_features(stft_energy(x), RPM)

    def run_par():
        list(fp._pool().map(lambda x: fp.sensor_features(stft_energy(x), RPM), chans))

    def stft_only():
        for x in chans:
            stft_energy(x)

    res = {}
    for name, fn in [("stft_only_serial", stft_only), ("original_serial", run_orig),
                     ("fast_serial", run_fast), ("fast_parallel3", run_par)]:
        fn()
        ts = []
        for _ in range(REPS):
            a = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - a)
        res[name] = {"p50": round(float(np.median(ts)) * 1e3, 1),
                     "p95": round(float(np.percentile(ts, 95)) * 1e3, 1)}
        print(name, res[name], flush=True)

    out = {"device": "PC (not Raspberry Pi)", "cpu_count": os.cpu_count(), "reps": REPS,
           "config": {"fs": FS, "seconds": SEC, "nfft": NFFT, "hop": HOP, "width": W,
                      "height": H, "rpm": RPM, "input": "synthetic"},
           "time_ms_excluding_model": res,
           "note": "STFT energy scaling differs from AI Hub images; timing only."}
    p = HERE.parent / "results" / "vehicle_path_bench_pc.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    main()
