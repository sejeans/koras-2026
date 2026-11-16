"""87채널 전처리의 빠른 버전. 원본(모터_특징추출_v2.py)은 건드리지 않는다.

바꾼 것은 두 가지다.
1. 분포 특징 12개(extract_ext_features)를 numba 함수 하나로 합쳤다.
   원본은 numpy 로 (1280, 2000) 배열을 nansum 9번, log 2번, nancumsum 1번 훑는다.
   여기서는 열마다 두 번만 훑는다. 원본처럼 NaN 을 건너뛴다.
2. 센서 3개를 스레드 3개로 동시에 처리한다. numba 함수는 nogil 이라 병렬로 돈다.

기본 특징 4개, 색 변환, 차수축 변환은 원본 함수를 그대로 부른다.
누적합을 float64 로 하기 때문에 원본(float32)과 마지막 자리가 다를 수 있다.
학습 데이터는 이 값을 float16 으로 저장했으므로 그 차이는 저장 단계에서 대부분 사라진다.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
# 라즈베리파이 묶음(pi_bundle)에서는 모터_특징추출_v2.py 가 같은 폴더에 있다.
# PC 프로젝트에서는 두 단계 위(매녀졸프\)에 있다.
ROOT = HERE if (HERE / "모터_특징추출_v2.py").exists() else HERE.parents[1]
sys.path.insert(0, str(ROOT))
import 모터_특징추출_v2 as fx  # noqa: E402

EXT_NAMES = [f"band_ratio_{i}" for i in range(8)] + \
            ["centroid", "entropy", "rolloff", "log_flatness"]


@numba.njit(nogil=True, cache=True)
def _ext_kernel(energy, n_bands, rolloff_p):
    """extract_ext_features 와 같은 12개 값을 (12, W) 로 돌려준다.

    energy 의 행 0 은 이미지 상단(고주파)이다. 원본처럼 뒤집어서 행 0 = 저주파로 계산한다.
    """
    H, W = energy.shape
    edges = np.empty(n_bands + 1, np.int64)
    for i in range(n_bands + 1):
        edges[i] = int(i * H / n_bands)   # np.linspace(0, H, n+1).astype(int) 와 같다

    tot = np.zeros(W)
    band = np.zeros((n_bands, W))
    cen = np.zeros(W)
    slog = np.zeros(W)
    nvalid = np.zeros(W)

    # 1회차: 합계, 대역합, 무게중심, log 평균
    b = 0
    for r in range(H):
        while r >= edges[b + 1]:
            b += 1
        src = H - 1 - r
        for w in range(W):
            x = energy[src, w]
            if np.isnan(x):
                continue
            if x < 0.0:
                x = 0.0
            tot[w] += x
            band[b, w] += x
            cen[w] += x * r
            slog[w] += np.log(x + 1e-12)
            nvalid[w] += 1.0

    out = np.empty((12, W))
    for w in range(W):
        t = tot[w] + 1e-9
        for i in range(n_bands):
            out[i, w] = band[i, w] / t
        out[8, w] = cen[w] / t
        if nvalid[w] > 0:
            out[11, w] = slog[w] / nvalid[w] - np.log(tot[w] / nvalid[w] + 1e-12)
        else:
            out[11, w] = np.nan

    # 2회차: 엔트로피, rolloff
    ent = np.zeros(W)
    cum = np.zeros(W)
    roll = np.full(W, -1.0)
    for r in range(H):
        src = H - 1 - r
        for w in range(W):
            x = energy[src, w]
            if np.isnan(x):
                continue
            if x < 0.0:
                x = 0.0
            t = tot[w] + 1e-9
            p = x / t
            ent[w] -= p * np.log(p + 1e-12)
            cum[w] += x
            if roll[w] < 0.0 and cum[w] / t >= rolloff_p:
                roll[w] = r
    for w in range(W):
        out[9, w] = ent[w]
        out[10, w] = roll[w] if roll[w] >= 0.0 else 0.0   # 원본 argmax 는 없으면 0
    return out


def ext_features(energy: np.ndarray) -> dict[str, np.ndarray]:
    arr = _ext_kernel(np.ascontiguousarray(energy, dtype=np.float32),
                      fx.N_BANDS, fx.ROLLOFF_P)
    return {n: arr[i].astype(np.float32) for i, n in enumerate(EXT_NAMES)}


def sensor_features(energy: np.ndarray, rpm: float) -> dict[str, np.ndarray]:
    """센서 1개의 에너지 맵에서 rpm 을 뺀 28개 특징."""
    base, _ = fx.extract_timeseries_vectorized(energy)
    feats = {c: base[c].to_numpy(np.float32)
             for c in ["peak_freq_bin", "band_start", "band_end", "rms"]}
    feats.update(ext_features(energy))
    eo = fx.to_order_axis(energy, float(rpm), n_out=fx.ORD_BINS)
    feats.update({f"ord_{k}": v for k, v in ext_features(eo).items()})
    return feats


def warmup():
    """numba 컴파일. 실시간 측정 전에 한 번 부른다."""
    fx._get_voxel()
    ext_features(np.random.default_rng(0).random((64, 16), dtype=np.float32))


_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=3)
    return _POOL


def from_energies(energies, rpm, parallel=True):
    """센서 3개 에너지 맵 -> 센서별 특징 dict 리스트."""
    if parallel:
        return list(_pool().map(lambda e: sensor_features(e, rpm), energies))
    return [sensor_features(e, rpm) for e in energies]


def _png_to_features(path, rpm):
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return sensor_features(fx.rgb_to_energy(arr), rpm)


def from_pngs(paths, rpm, parallel=True):
    """PNG 3장 -> 센서별 특징. 읽기와 색 변환까지 센서별 스레드에서 한다."""
    if parallel:
        return list(_pool().map(lambda p: _png_to_features(p, rpm), paths))
    return [_png_to_features(p, rpm) for p in paths]
