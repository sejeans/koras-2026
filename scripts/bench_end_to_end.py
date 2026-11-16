"""센서 신호 10초 1구간을 넣고 판정이 나올 때까지를 한 프로세스에서 이어서 잰다.

경로 두 개 x 코드 두 개 = 네 가지를 잰다.
  PNG 경로 : 신호 -> STFT -> jet 컬러 PNG 저장 -> PNG 읽기 -> 색을 에너지로 -> 특징 -> 모델 -> 판정
  차량 경로: 신호 -> STFT -> 에너지 -> 특징 -> 모델 -> 판정
  원본 코드: 모터_특징추출_v2.py 함수, 센서 3개 순서대로
  개선 코드: fast_preprocess.py, 센서 3개를 스레드 3개로 동시에 (모델은 1스레드)

입력
- 진동과 전류: 25,000 Hz x 10초 x 센서 3개 가상 신호. 원시 파형은 공개되지 않았다.
  STFT와 특징 계산량은 신호 내용과 거의 무관하다. PNG 압축 시간만 내용에 따라 달라진다.
- 회전수: 실제 CAN 파일(D:\\1.모터-감속기 데이터) 한 구간의 평균. 없으면 3000.
- 모델: 87채널, 커널 9,996개 (모델_ord_B), IONIQ.

PNG 경로의 PNG 저장 방식(압축 수준 6, jet 컬러맵)은 AI Hub 원본 이미지 생성 방식과 같다는 보장이 없다.
차량 경로의 에너지 스케일(dB를 0에서 1로 맞춤)도 AI Hub 이미지와 같다는 보장이 없다.
그래서 여기서 나온 판정 결과는 의미가 없고 시간만 본다.

실행: python bench_end_to_end.py [--reps 30] [--out 파일]
라즈베리파이에서도 같은 명령으로 돈다(모델_ord_B 와 D: 경로가 없으면 --model, --can 으로 지정).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # 묶음에서는 rpi_stft_benchmark.py 가 같은 폴더에 있다
sys.path.append(str(HERE.parents[1] / "experiments" / "2026-09-15_CAN_fusion" / "src"))
import fast_preprocess as fp  # noqa: E402
import rpi_stft_benchmark as sb  # noqa: E402

fx = fp.fx
FS, SEC, NFFT, W, H = 25000, 10.0, 2560, 2000, 1280
N = int(FS * SEC)
HOP = N // W
WINDOW = np.hanning(NFFT).astype(np.float32)
BASE4 = ["peak_freq_bin", "band_start", "band_end", "rms"]


def jet_lut():
    import matplotlib
    cm = matplotlib.colormaps["jet"]
    return (np.asarray([cm(i / 255.0)[:3] for i in range(256)]) * 255).round().astype(np.uint8)


JET = None


# ---------------------------------------------------------------- 단계
def stft_db(x):
    """(1281, 2000) dB, 행0 = 저주파."""
    return sb.stft_magnitude(x, NFFT, HOP, W, WINDOW)


def _rows():
    return np.linspace(0, NFFT // 2, H).round().astype(np.int32)


def db_to_energy(db):
    """차량 경로: dB -> 0~1 에너지, 행0 = 고주파, 높이 1280."""
    lo, hi = float(db.min()), float(db.max())
    e = ((db - lo) / max(hi - lo, 1e-6)).astype(np.float32)[::-1]
    return np.ascontiguousarray(e[_rows()])


def db_to_png_bytes(db):
    """PNG 경로: dB -> 8비트 -> jet RGB -> PNG 인코딩(압축 수준 6)."""
    lo, hi = float(db.min()), float(db.max())
    u8 = np.clip((db - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)[::-1]
    rgb = JET[u8[_rows()]]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def png_bytes_to_energy(b):
    with Image.open(io.BytesIO(b)) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return fx.rgb_to_energy(arr)


def orig_features(e, rpm):
    base, _ = fx.extract_timeseries_vectorized(e)
    f = {c: base[c].to_numpy(np.float32) for c in BASE4}
    ext = fx.extract_ext_features(e)
    f.update({c: ext[c].to_numpy() for c in ext.columns})
    oext = fx.extract_ext_features(fx.to_order_axis(e, rpm, n_out=fx.ORD_BINS))
    f.update({f"ord_{c}": oext[c].to_numpy() for c in oext.columns})
    return f


def assemble(sensor_feats, rpm, cols):
    mats = []
    for f in sensor_feats:
        m = np.stack([np.full(W, rpm, np.float32) if c == "rpm" else f[c] for c in cols], 1)
        if np.isnan(m).any():
            m = np.nan_to_num(pd.DataFrame(m).ffill().bfill().to_numpy(np.float32), nan=0.0)
        mats.append(m)
    return np.concatenate(mats, 1).T[None].astype(np.float32)        # (1, 87, 2000)


def judge(slot, X):
    Xn = ((X - slot["ch_mean"][None, :, None]) / slot["ch_std"][None, :, None]).astype(np.float32)
    F = slot["mr"].transform(Xn)
    F = np.asarray(F.to_numpy() if hasattr(F, "to_numpy") else F, dtype=np.float32)
    if slot.get("sc") is not None:
        F = slot["sc"].transform(F)
    p = slot["lr"].predict_proba(F)[0]
    return slot["lr"].classes_[int(p.argmax())]


# ---------------------------------------------------------------- 경로
def run_serial(path, code, chans, rpm, slot, cols):
    """단계별 시간을 센서 3개 합으로 기록한다."""
    t = {}

    def add(k, s):
        t[k] = t.get(k, 0.0) + (time.perf_counter() - s)

    a = time.perf_counter()
    feats = []
    for x in chans:
        s = time.perf_counter(); db = stft_db(x); add("stft", s)
        if path == "png":
            s = time.perf_counter(); b = db_to_png_bytes(db); add("png_write", s)
            s = time.perf_counter(); e = png_bytes_to_energy(b); add("png_read_lut", s)
        else:
            s = time.perf_counter(); e = db_to_energy(db); add("scale", s)
        s = time.perf_counter()
        feats.append(orig_features(e, rpm) if code == "orig" else fp.sensor_features(e, rpm))
        add("features", s)
    s = time.perf_counter(); X = assemble(feats, rpm, cols); add("assemble", s)
    s = time.perf_counter(); judge(slot, X); add("model", s)
    t["total"] = time.perf_counter() - a
    return t


def run_parallel(path, chans, rpm, slot, cols):
    """개선 코드: 센서별 전처리를 스레드 3개로 동시에, 모델은 그 뒤 1스레드."""
    def one(x):
        db = stft_db(x)
        e = png_bytes_to_energy(db_to_png_bytes(db)) if path == "png" else db_to_energy(db)
        return fp.sensor_features(e, rpm)

    a = time.perf_counter()
    feats = list(fp._pool().map(one, chans))
    b = time.perf_counter()
    X = assemble(feats, rpm, cols)
    judge(slot, X)
    c = time.perf_counter()
    return {"preprocess_parallel": b - a, "assemble_model": c - b, "total": c - a}


def pi_thermal():
    """라즈베리파이 온도, 스로틀 플래그, 클럭. 파이가 아니면 빈 dict."""
    import subprocess
    out = {}
    for key, cmd in [("temp", ["vcgencmd", "measure_temp"]),
                     ("throttled", ["vcgencmd", "get_throttled"]),
                     ("arm_clock", ["vcgencmd", "measure_clock", "arm"])]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
            if r:
                out[key] = r.split("=", 1)[-1]
        except Exception:
            pass
    return out


def summarize(rows):
    keys = rows[0].keys()
    return {k: {"p50": round(float(np.median([r[k] for r in rows])) * 1e3, 1),
                "p95": round(float(np.percentile([r[k] for r in rows], 95)) * 1e3, 1)}
            for k in keys}


def main():
    global JET
    bundle = (HERE / "models").exists()          # pi_bundle 폴더에서 실행 중인가
    root = HERE.parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--model", default=str(
        HERE / "models" / "model_registry_fuse_rpm_ext.joblib" if bundle
        else root / "모델_ord_B" / "model_registry_fuse_rpm_ext.joblib"))
    ap.add_argument("--vehicle", default="IONIQ")
    ap.add_argument("--can", default=str(
        HERE / "can_sample.csv" if bundle
        else r"D:\1.모터-감속기 데이터\IONIQ\DEMAG\0728\22_07_28_12_15_20_01_zsplit005_001.csv"))
    ap.add_argument("--out", default=str(
        HERE / "results" / "end_to_end.json" if bundle
        else HERE.parent / "results" / "end_to_end_bench_pc.json"))
    ap.add_argument("--plans", default="all",
                    help="쉼표로 구분. 예: vehicle_orig_serial,vehicle_fast_parallel3")
    args = ap.parse_args()

    JET = jet_lut()
    reg = joblib.load(args.model)
    slot = reg[(args.vehicle, "FUSED87")]
    if hasattr(slot["mr"], "n_jobs"):
        slot["mr"].n_jobs = 1
    fc = list(slot["feature_cols"])
    cols = fc[:len(fc) // 3] if len(fc) == 87 else fc

    rpm, rpm_src = 3000.0, "default"
    if Path(args.can).exists():
        can = pd.read_csv(args.can, header=None, usecols=[2], dtype=np.float32)
        rpm, rpm_src = float(can[2].mean()), Path(args.can).name
    data = sb.synth_window(N, 3, FS)
    chans = [np.ascontiguousarray(data[:, c]) for c in range(3)]
    print(f"rpm {rpm:.0f} ({rpm_src}), 신호 {N:,} 샘플 x 3, 반복 {args.reps}", flush=True)

    fp.warmup()
    thermal_before = pi_thermal()
    res = {}
    plans = [("png_orig_serial", lambda: run_serial("png", "orig", chans, rpm, slot, cols)),
             ("png_fast_serial", lambda: run_serial("png", "fast", chans, rpm, slot, cols)),
             ("png_fast_parallel3", lambda: run_parallel("png", chans, rpm, slot, cols)),
             ("vehicle_orig_serial", lambda: run_serial("vehicle", "orig", chans, rpm, slot, cols)),
             ("vehicle_fast_serial", lambda: run_serial("vehicle", "fast", chans, rpm, slot, cols)),
             ("vehicle_fast_parallel3", lambda: run_parallel("vehicle", chans, rpm, slot, cols))]
    if args.plans != "all":
        keep = {p.strip() for p in args.plans.split(",")}
        plans = [p for p in plans if p[0] in keep]
    for name, fn in plans:
        fn(); fn()                                    # 예열
        rows = [fn() for _ in range(args.reps)]
        res[name] = summarize(rows)
        print(name, {k: v["p50"] for k, v in res[name].items()}, flush=True)

    out = {"device": {"machine": platform.machine(), "platform": platform.platform(),
                      "cpu_count": os.cpu_count(), "python": sys.version.split()[0]},
           "config": {"fs": FS, "seconds": SEC, "nfft": NFFT, "hop": HOP, "width": W,
                      "height": H, "rpm": rpm, "rpm_source": rpm_src, "signal": "synthetic",
                      "model": Path(args.model).name, "vehicle": args.vehicle,
                      "kernels": 9996, "reps": args.reps, "blas_threads": 1},
           "thermal_before": thermal_before, "thermal_after": pi_thermal(),
           "time_ms": res,
           "note": "Collection time (10 s) excluded. PNG settings and energy scaling are not "
                   "the dataset's; predictions are meaningless, timing only."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
