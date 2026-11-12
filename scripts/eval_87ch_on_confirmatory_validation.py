"""확증 분할(2026-10-16)의 검증 세션 14,792건에 이미 학습된 87채널 모델을 평가한다.

목적: AI Hub 모델이 epoch 선택에 쓴 바로 그 검증 데이터에서 87채널 모델의 성능을 구해,
두 모델을 같은 표본으로 비교한다. 재학습은 하지 않는다. 87채널 모델은 이 검증 세션을
학습에도, 설정 선택에도 쓰지 않았다(train_87ch_full.py 162행).

출력: results/confirmatory_validation_87ch/{vehicle}_predictions.parquet, metrics.json
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = WORK.parent
CACHE = WORK / 'cache'
CACHE.mkdir(exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / 'experiments/2026-09-15_CAN_fusion/src'))
import can_fusion_experiment as base  # noqa: E402

CONF = ROOT / 'experiments/2026-10-16_confirmatory_noninferiority'
OLD = ROOT / 'experiments/2026-09-15_CAN_fusion/evidence/can_fusion_full'
OUT = WORK / 'results/confirmatory_validation_87ch'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    assignments = pd.read_csv(CONF / 'split_final/assignments.csv')
    manifest = pd.read_parquet(OLD / 'manifest.parquet')
    segments = json.loads((OLD / 'segments.json').read_text(encoding='utf-8'))
    role = assignments.set_index('sample_id').role
    manifest['role'] = manifest.sample_id.map(role)
    assert manifest.role.notna().all()
    frames = []
    for vehicle in ['IONIQ', 'KONA', 'NIRO']:
        pred_path = OUT / f'{vehicle}_predictions.parquet'
        if pred_path.exists():
            frames.append(pd.read_parquet(pred_path))
            continue
        model_dir = CONF / 'model87_full' / vehicle
        scope = json.loads((model_dir / 'fit_scope.json').read_text(encoding='utf-8'))
        selected = manifest[(manifest.vehicle == vehicle) & (manifest.role == 'validation')]
        selected = selected.reset_index(drop=True)
        # 누수 점검: 검증 세션은 학습 세션과 겹치지 않아야 한다.
        overlap = set(selected.session) & set(scope['train_sessions'])
        assert not overlap, f'{vehicle}: train/validation session overlap {len(overlap)}'
        assert not set(selected.sample_id) & set(scope['train_ids'])
        base.log(f'{vehicle}: validation samples {len(selected):,}, sessions {selected.session.nunique()}')
        enc = joblib.load(model_dir / 'encoder.joblib')
        clf = joblib.load(model_dir / 'model_87ch.joblib')
        xpath = CACHE / f'{vehicle}_val_x.npy'
        x = base.load_x(selected, segments, xpath, resume=True)
        start = time.perf_counter()
        probs = []
        for lo in range(0, len(x), 256):
            hi = min(lo + 256, len(x))
            z = enc['mr'].transform(base.normalize(np.asarray(x[lo:hi]), enc['ch_mean'], enc['ch_std']))
            probs.append(clf['head'].predict_proba(clf['scaler'].transform(np.asarray(z, dtype=np.float32))))
            if (lo // 256) % 8 == 0:
                base.log(f'{vehicle}: {hi:,}/{len(x):,} ({(time.perf_counter()-start)/60:.1f} min)')
        p = np.vstack(probs)
        classes = list(clf['head'].classes_)
        out = selected[['sample_id', 'vehicle', 'session', 'fault_class']].copy()
        out['predicted'] = np.array(classes)[p.argmax(1)]
        for i, c in enumerate(classes):
            out[f'p_{c}'] = p[:, i]
        out.to_parquet(pred_path, index=False)
        frames.append(out)
        del x
        if hasattr(xpath, 'unlink'):
            for f in [xpath, xpath.with_suffix('.complete.json')]:
                if f.exists():
                    f.unlink()
        base.log(f'{vehicle}: done, cache removed')
    merged = pd.concat(frames, ignore_index=True)
    assert len(merged) == int((assignments.role == 'validation').sum())
    merged.to_parquet(OUT / 'predictions.parquet', index=False)
    (OUT / 'metrics.json').write_text(json.dumps(base.score(merged.fault_class, merged.predicted),
                                                indent=2), encoding='utf-8')
    (OUT / 'DONE').write_text('complete', encoding='utf-8')
    print('done')


if __name__ == '__main__':
    main()
