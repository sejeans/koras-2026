"""확증 분할(2026-10-16, 학습 80% / 검증 10% / 시험 10%) 결과를 신뢰성 지표로 정리한다.

- 87채널: 시험 14,879건 예측(기존), 검증 14,792건 예측(eval_87ch_on_confirmatory_validation.py)
- AI Hub: 학습이 epoch 4 이후 중단되어 시험 평가가 없다. 검증 혼동행렬(history.json)만 있다.
  표본별 예측이 없으므로 AI Hub 쪽은 점추정만 낸다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reliability_metrics_v3 import Boot, reliability_table, error_composition, FAULTS, CLASSES

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = WORK.parent
CONF = ROOT / 'experiments/2026-10-16_confirmatory_noninferiority'
HIST = ROOT / 'AI모델/EV_Model_work/confirmatory_20261016/train/history.json'
AIHUB_ORDER = ['NORMAL', 'ECC10', 'ECC20', 'DEMAG', 'REDUC']


def from_confusion(cm, labels):
    cm = np.asarray(cm)
    i = {c: labels.index(c) for c in labels}
    n = i['NORMAL']
    res = {'accuracy_pct': 100 * np.trace(cm) / cm.sum(), 'samples': int(cm.sum())}
    res['NORMAL_false_alarm_pct'] = 100 * (cm[n].sum() - cm[n, n]) / cm[n].sum()
    fault_rows = [i[f] for f in FAULTS]
    res['ALL_FAULTS_missed_pct'] = 100 * cm[fault_rows, n].sum() / cm[fault_rows].sum()
    for f in FAULTS:
        r = cm[i[f]]
        res[f] = {'correct_mode_pct': 100 * r[i[f]] / r.sum(), 'missed_as_normal_pct': 100 * r[n] / r.sum(),
                  'wrong_fault_mode_pct': 100 * (r.sum() - r[i[f]] - r[n]) / r.sum(), 'n': int(r.sum())}
    return res


def table87(df):
    df = df.copy()
    df['session_id'] = df.sample_id.str.split('/').str[0] + '/' + df.session.astype(str)
    boot = Boot(df.session_id.to_numpy())
    t, _ = reliability_table(df, 'predicted', boot)
    comp = error_composition(df, 'predicted')
    votes = {}
    for c in CLASSES:
        s = df[df.fault_class == c].groupby('session_id').predicted.agg(lambda x: x.value_counts().idxmax())
        votes[c] = {'sessions': int(len(s)), 'vote_correct': int((s == c).sum()), 'vote_normal': int((s == 'NORMAL').sum())}
    return {'samples': len(df), 'sessions': int(df.session_id.nunique()), 'reliability': t,
            'error_composition': comp, 'session_majority_vote': votes}


def main():
    res = {}
    test = pd.read_parquet(CONF / 'model87_full/predictions.parquet')
    res['87ch_test'] = table87(test)
    vpath = WORK / 'results/confirmatory_validation_87ch/predictions.parquet'
    if vpath.exists():
        val = pd.read_parquet(vpath)
        res['87ch_validation'] = table87(val)
    hist = json.loads(HIST.read_text(encoding='utf-8'))
    best = [h for h in hist if h.get('selected_as_best')][-1]
    cm = best['inner_validation']['confusion_matrix']
    assert np.asarray(cm).sum() == 14792
    res['AIHub_validation'] = {'epoch': best['epoch'], 'epochs_completed': len(hist),
                               'label_order_assumed': AIHUB_ORDER, **from_confusion(cm, AIHUB_ORDER)}
    # 라벨 순서 점검: 혼동행렬 행 합계가 검증 데이터의 클래스별 표본 수와 같아야 한다.
    # (클래스별 표본 수가 모두 달라 순서가 하나로 정해진다)
    assign = pd.read_csv(CONF / 'split_final/assignments.csv')
    counts = assign[assign.role == 'validation'].fault_class.value_counts()
    assert list(np.asarray(cm).sum(1)) == [int(counts[c]) for c in AIHUB_ORDER], 'label order check failed'
    out = WORK / 'results/confirmatory_metrics.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    for k in ['87ch_test', '87ch_validation']:
        if k in res:
            r = res[k]['reliability']
            print(k, res[k]['samples'], res[k]['sessions'], 'acc', round(r['accuracy']['pct'], 2), [round(x, 2) for x in r['accuracy']['ci95']],
                  'FA', round(r['NORMAL']['false_alarm']['pct'], 2), [round(x, 2) for x in r['NORMAL']['false_alarm']['ci95']],
                  'DEMAG miss', round(r['DEMAG']['missed_as_normal']['pct'], 2), [round(x, 2) for x in r['DEMAG']['missed_as_normal']['ci95']],
                  'all miss', round(r['ALL_FAULTS']['missed_as_normal']['pct'], 2),
                  'DN share', round(res[k]['error_composition']['demag_normal_share_pct'], 1), res[k]['session_majority_vote'])
            for f in FAULTS:
                print('   ', f, {m: round(v['pct'], 2) for m, v in r[f].items()})
    print('AIHub_validation', json.dumps(res['AIHub_validation'], default=float))


if __name__ == '__main__':
    main()
