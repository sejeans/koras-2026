"""스펙트로그램 PNG 3장(센서 3종, 10초 1구간)에서 87채널 입력을 만드는 전처리 시간을 PC에서 잰다.

라즈베리파이 측정(0.455초)에는 이 전처리가 빠져 있다. 라즈베리파이가 연결되어 있지 않아
PC 단일 프로세스로만 잰다. 라즈베리파이 값으로 옮겨 쓰면 안 된다.

단계: PNG 읽기 -> 컬러맵 역변환(LUT) -> 기본 특징 4개 -> 분포 특징 12개 -> 차수축 변환 후 분포 특징 12개
(rpm은 CAN에서 받는 값이라 시간에 넣지 않는다)
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ['OMP_NUM_THREADS'] = '1'
import 모터_특징추출_v2 as fx  # noqa: E402
from PIL import Image  # noqa: E402

BASE = Path(r'D:\TS\1.모터-감속기_데이터')
SENSORS = ['Current_U', 'Vib_Motor', 'Vib_TM']
REPS = 30


def pick_samples(n=REPS, seed=42):
    rng = random.Random(seed)
    out = []
    for v in ['IONIQ', 'KONA', 'NIRO']:
        for cls in ['NORMAL', 'DEMAG']:
            d = BASE / v / cls
            dates = sorted(p for p in d.iterdir() if p.is_dir())
            date = rng.choice(dates)
            files = sorted((date / 'Current_U').glob('*_Current_U.png'))
            for f in rng.sample(files, n // 6):
                stem = f.name.removesuffix('_Current_U.png')
                paths = [date / s / f'{stem}_{s}.png' for s in SENSORS]
                if all(p.exists() for p in paths):
                    out.append(paths)
    return out


def one(paths, rpm=3000.0):
    t = {}
    s = time.perf_counter()
    arrs = []
    for p in paths:
        with Image.open(p) as im:
            arrs.append(np.asarray(im.convert('RGB'), dtype=np.uint8))
    t['read_png'] = time.perf_counter() - s
    s = time.perf_counter()
    energies = [fx.rgb_to_energy(a) for a in arrs]
    t['lut'] = time.perf_counter() - s
    s = time.perf_counter()
    for e in energies:
        fx.extract_timeseries_vectorized(e)
    t['base4'] = time.perf_counter() - s
    s = time.perf_counter()
    for e in energies:
        fx.extract_ext_features(e)
    t['ext12'] = time.perf_counter() - s
    s = time.perf_counter()
    for e in energies:
        fx.extract_ext_features(fx.to_order_axis(e, rpm, n_out=fx.ORD_BINS))
    t['ord12'] = time.perf_counter() - s
    t['total'] = sum(t.values())
    t['width'] = int(energies[0].shape[1])
    return t


def main():
    s = time.perf_counter()
    fx._get_voxel()
    lut_build = time.perf_counter() - s
    samples = pick_samples()
    one(samples[0])  # 캐시와 첫 호출 비용 제외
    rows = [one(p) for p in samples]
    keys = ['read_png', 'lut', 'base4', 'ext12', 'ord12', 'total']
    res = {'device': 'PC (not Raspberry Pi)', 'cpu_count': os.cpu_count(), 'samples': len(rows),
           'lut_build_once_s': lut_build, 'image_width_px': sorted({r['width'] for r in rows}),
           'p50_ms': {k: 1000 * float(np.median([r[k] for r in rows])) for k in keys},
           'p95_ms': {k: 1000 * float(np.percentile([r[k] for r in rows], 95)) for k in keys}}
    out = HERE.parent / 'results/preprocess_bench_pc.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
