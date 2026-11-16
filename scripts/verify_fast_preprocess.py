"""빠른 전처리(fast_preprocess.py)가 원본과 같은 판정을 내는지, 얼마나 빠른지 잰다.

표본: 3차종 x 5클래스 x 20개 = 300개 (PNG 3장 = 10초 1구간 = 모델 입력 1개).
각 표본을 원본 코드와 빠른 코드로 따로 전처리하고, 같은 87채널 모델(ord_B, 커널 9,996개)에 넣는다.
학습 데이터와 같게 특징을 float16/float32 로 저장 형식 변환한 뒤 모델에 넣는다.

비교 항목
- 특징값: 저장 형식 변환 후 원본과 정확히 같은 값의 비율, 최대 차이
- 예측: 클래스가 같은 비율, 클래스 확률의 최대 차이
- 시간(1스레드 BLAS): 원본 직렬 / 빠른 버전 직렬 / 빠른 버전 센서 3개 병렬

PC 전체:  python verify_fast_preprocess.py
묶음(PC, Pi 공통):
  python verify_fast_preprocess.py --data ./data --rpm-index ./rpm_index_subset.json \
      --model ./models/model_registry_fuse_rpm_ext.joblib --per-cell 4 --out results/verify.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fast_preprocess as fp  # noqa: E402

fx = fp.fx
ROOT = fp.ROOT
DEFAULT_DATA = r"D:\TS\1.모터-감속기_데이터"
SENSORS = ["Current_U", "Vib_Motor", "Vib_TM"]
CLASSES = ["DEMAG", "ECC10", "ECC20", "NORMAL", "REDUC"]
VEHICLES = ["IONIQ", "KONA", "NIRO"]
KEEP32 = {"peak_freq_bin", "band_start", "band_end", "rolloff", "centroid", "rpm",
          "ord_rolloff", "ord_centroid"}


def pick_samples(rpm_idx, base, per_cell, seed=20261113):
    """차종 x 클래스마다 per_cell 개. 묶음 폴더에서 부르면 들어 있는 표본을 전부 고른다."""
    BASE, PER_CELL = Path(base), per_cell
    rng = random.Random(seed)
    out = []
    for v in VEHICLES:
        for cls in CLASSES:
            d = BASE / v / cls
            if not d.exists():
                continue
            # 날짜 폴더 전체를 훑으면 느리므로 무작위 날짜 4개에서만 후보를 모은다
            dates = sorted(p for p in d.iterdir() if p.is_dir())
            dates = rng.sample(dates, min(4, len(dates)))
            cands = []
            for date in sorted(dates):
                files = sorted((date / "Current_U").glob("*_Current_U.png"))
                for f in rng.sample(files, min(3 * PER_CELL, len(files))):
                    stem = f.name.removesuffix("_Current_U.png")
                    paths = [date / s / f"{stem}_{s}.png" for s in SENSORS]
                    rec = rpm_idx.get(f"{stem}_Current_U")
                    if rec and rec.get("rpm") and all(p.exists() for p in paths):
                        cands.append((v, cls, paths, float(rec["rpm"])))
            out += rng.sample(cands, min(PER_CELL, len(cands)))
    return out


def orig_sensor(path, rpm):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    e = fx.rgb_to_energy(arr)
    base, _ = fx.extract_timeseries_vectorized(e)
    feats = {c: base[c].to_numpy(np.float32)
             for c in ["peak_freq_bin", "band_start", "band_end", "rms"]}
    ext = fx.extract_ext_features(e)
    feats.update({c: ext[c].to_numpy() for c in ext.columns})
    eo = fx.to_order_axis(e, rpm, n_out=fx.ORD_BINS)
    oext = fx.extract_ext_features(eo)
    feats.update({f"ord_{c}": oext[c].to_numpy() for c in oext.columns})
    return feats


def to_model_input(sensor_feats, rpm, cols):
    """센서별 특징 -> 학습과 같은 저장 형식 -> (87, 2000)."""
    mats = []
    for feats in sensor_feats:
        W = len(feats["rms"])
        df = {}
        for c in cols:
            v = np.full(W, rpm, np.float32) if c == "rpm" else feats[c]
            df[c] = v.astype(np.float32 if c in KEEP32 else np.float16).astype(np.float32)
        m = pd.DataFrame(df)[cols].to_numpy(np.float32)
        if np.isnan(m).any():
            m = np.nan_to_num(pd.DataFrame(m).ffill().bfill().to_numpy(np.float32), nan=0.0)
        mats.append(m)
    L = min(len(m) for m in mats)
    n = (L // 2000) * 2000
    fused = np.concatenate([m[:n] for m in mats], axis=1)
    return fused.T.reshape(fused.shape[1], -1, 2000).transpose(1, 0, 2)  # (k, 87, 2000)


def predict(slot, X):
    Xn = ((X - slot["ch_mean"][None, :, None]) / slot["ch_std"][None, :, None]).astype(np.float32)
    F = slot["mr"].transform(Xn)
    F = np.asarray(F.to_numpy() if hasattr(F, "to_numpy") else F, dtype=np.float32)
    if slot.get("sc") is not None:
        F = slot["sc"].transform(F)
    return slot["lr"].predict_proba(F)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA, help="차종/클래스/날짜/센서/*.png 구조의 폴더")
    ap.add_argument("--rpm-index", default=str(ROOT / "rpm_index.json"))
    ap.add_argument("--model", default=str(ROOT / "모델_ord_B" / "model_registry_fuse_rpm_ext.joblib"))
    ap.add_argument("--per-cell", type=int, default=int(os.environ.get("PER_CELL", 20)),
                    help="차종 x 클래스 한 칸당 표본 수 (기본 20, 300개)")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "fast_preprocess_verify_pc.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    rpm_idx = json.loads(Path(args.rpm_index).read_text(encoding="utf-8"))
    reg = joblib.load(args.model)
    for slot in reg.values():
        if hasattr(slot["mr"], "n_jobs"):
            slot["mr"].n_jobs = 1
    fc = list(next(iter(reg.values()))["feature_cols"])
    cols = fc[:len(fc) // 3] if len(fc) == 87 else fc
    assert len(cols) == 29, cols
    print(f"특징 {len(cols)}개 x 센서 3 = 87채널: {cols}")

    samples = pick_samples(rpm_idx, args.data, args.per_cell)
    print(f"표본 {len(samples)}개 선택 ({time.time()-t0:.0f}s)")

    fp.warmup()
    # 원본도 캐시를 데워 둔다
    v, cls, paths, rpm = samples[0]
    [orig_sensor(p, rpm) for p in paths]
    fp.from_pngs(paths, rpm, parallel=True)

    t_orig, t_fast, t_par = [], [], []
    feat_eq, feat_n, feat_maxdiff = {}, {}, {}
    agree, n_pred, prob_maxdiff = 0, 0, 0.0
    per_vehicle = {v: [0, 0] for v in VEHICLES}
    for i, (v, cls, paths, rpm) in enumerate(samples, 1):
        a = time.perf_counter()
        fo = [orig_sensor(p, rpm) for p in paths]
        b = time.perf_counter()
        ff = fp.from_pngs(paths, rpm, parallel=False)
        c = time.perf_counter()
        fpar = fp.from_pngs(paths, rpm, parallel=True)
        d = time.perf_counter()
        t_orig.append(b - a)
        t_fast.append(c - b)
        t_par.append(d - c)

        for so, sf, sp in zip(fo, ff, fpar):
            for k in sf:
                assert np.array_equal(sf[k], sp[k], equal_nan=True), "병렬과 직렬 결과가 다르다"
                dt = np.float32 if k in KEEP32 else np.float16
                xo, xf = so[k].astype(dt), sf[k].astype(dt)
                eq = (xo == xf) | (np.isnan(xo) & np.isnan(xf))
                feat_eq[k] = feat_eq.get(k, 0) + int(eq.sum())
                feat_n[k] = feat_n.get(k, 0) + eq.size
                diff = np.nanmax(np.abs(xo.astype(np.float64) - xf.astype(np.float64)))
                feat_maxdiff[k] = max(feat_maxdiff.get(k, 0.0), float(diff))

        slot = reg[(v, "FUSED87")]
        po = predict(slot, to_model_input(fo, rpm, cols))
        pf = predict(slot, to_model_input(ff, rpm, cols))
        same = (po.argmax(1) == pf.argmax(1))
        agree += int(same.sum())
        n_pred += len(same)
        per_vehicle[v][0] += int(same.sum())
        per_vehicle[v][1] += len(same)
        prob_maxdiff = max(prob_maxdiff, float(np.abs(po - pf).max()))
        if i % 30 == 0:
            print(f"  {i}/{len(samples)}  예측일치 {agree}/{n_pred}  "
                  f"원본 {np.median(t_orig)*1e3:.0f}ms / 빠른직렬 {np.median(t_fast)*1e3:.0f}ms"
                  f" / 빠른병렬 {np.median(t_par)*1e3:.0f}ms  ({time.time()-t0:.0f}s)")

    def ms(x):
        return {"p50": round(float(np.median(x)) * 1e3, 1),
                "p95": round(float(np.percentile(x, 95)) * 1e3, 1)}

    res = {
        "device": {"machine": platform.machine(), "platform": platform.platform(),
                   "cpu_count": os.cpu_count()},
        "samples": len(samples),
        "model": f"{Path(args.model).name} (87ch, 9,996 kernels)",
        "time_ms": {"original_serial": ms(t_orig), "fast_serial": ms(t_fast),
                    "fast_parallel3": ms(t_par)},
        "speedup_p50": {"fast_serial": round(np.median(t_orig) / np.median(t_fast), 2),
                        "fast_parallel3": round(np.median(t_orig) / np.median(t_par), 2)},
        "prediction_agreement": {"same_class": agree, "total": n_pred,
                                 "rate": agree / n_pred,
                                 "per_vehicle": per_vehicle,
                                 "max_abs_prob_diff": prob_maxdiff},
        "feature_exact_equal_rate_after_storage_cast": {
            k: round(feat_eq[k] / feat_n[k], 6) for k in feat_eq},
        "feature_max_abs_diff_after_storage_cast": feat_maxdiff,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
