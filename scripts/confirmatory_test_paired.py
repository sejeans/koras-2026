"""확증 분할 시험 데이터(14,879건)에서 87채널과 AI Hub(best.pt, epoch 3)를 짝지어 비교한다.

입력
- 87채널: experiments/2026-10-16_confirmatory_noninferiority/model87_full/predictions.parquet
- AI Hub: AI모델/EV_Model_work/confirmatory_20261016/test/predictions.csv (2026-11-12 평가)

사전 계획(PROTOCOL.md)의 비열등성 기준도 함께 계산한다.
- 정확도 차이(87ch - AI Hub)의 단측 95% 하한 > -1.0%p
- 감자 결함 재현율 차이의 단측 95% 하한 > -2.0%p
주의: AI Hub 학습은 계획(최대 10 epoch, patience 2)대로 끝나지 않고 epoch 4 이후 중단됐다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reliability_metrics_v3 import Boot, reliability_table, error_composition

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = WORK.parent
P87 = ROOT / 'experiments/2026-10-16_confirmatory_noninferiority/model87_full/predictions.parquet'
PAI = ROOT / 'AI모델/EV_Model_work/confirmatory_20261016/test/predictions.csv'


def main():
    a = pd.read_parquet(P87).rename(columns={'predicted': 'pred_87'})
    b = pd.read_csv(PAI).rename(columns={'predicted': 'pred_ai'})
    df = a.merge(b[['sample_id', 'actual', 'pred_ai']], on='sample_id', validate='one_to_one')
    assert len(df) == len(a) == len(b) == 14879, (len(a), len(b), len(df))
    assert (df.actual == df.fault_class).all(), 'label mismatch'
    df['session_id'] = df.sample_id.str.split('/').str[0] + '/' + df.session.astype(str)
    boot = Boot(df.session_id.to_numpy())
    tables, raws = {}, {}
    for name, col in [('87ch', 'pred_87'), ('AIHub', 'pred_ai')]:
        tables[name], raws[name] = reliability_table(df, col, boot)
    diff = {f'{k[0]}:{k[1]}': boot.diff(raws['87ch'][k], raws['AIHub'][k]) for k in raws['87ch']}
    # 사전 계획의 단측 95% 하한 (5 백분위수)
    ni = {}
    for key, margin in [(('ALL', 'accuracy'), -1.0), (('DEMAG', 'correct_mode'), -2.0)]:
        d = 100 * (raws['87ch'][key][1] - raws['AIHub'][key][1])
        lo = float(np.nanpercentile(d, 5))
        ni[f'{key[0]}:{key[1]}'] = {'diff_pctp': raws['87ch'][key][0]['pct'] - raws['AIHub'][key][0]['pct'],
                                    'one_sided_95_lower': lo, 'margin': margin, 'pass': lo > margin}
    both = (df.pred_87 == df.fault_class) & (df.pred_ai == df.fault_class)
    paired = {'both_correct': int(both.sum()),
              'only_87ch_correct': int(((df.pred_87 == df.fault_class) & (df.pred_ai != df.fault_class)).sum()),
              'only_AIHub_correct': int(((df.pred_87 != df.fault_class) & (df.pred_ai == df.fault_class)).sum()),
              'both_wrong': int(((df.pred_87 != df.fault_class) & (df.pred_ai != df.fault_class)).sum())}
    res = {'samples': len(df), 'sessions': int(df.session_id.nunique()),
           'aihub_checkpoint': 'best.pt epoch 3 of 4 completed (training stopped during epoch 5)',
           'reliability': tables, 'difference_87ch_minus_AIHub': diff, 'noninferiority_prespecified': ni,
           'paired_outcomes': paired,
           'error_composition': {n: error_composition(df, c) for n, c in [('87ch', 'pred_87'), ('AIHub', 'pred_ai')]}}
    out = WORK / 'results/confirmatory_test_paired.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    for m in ['87ch', 'AIHub']:
        r = tables[m]
        print(m, 'acc', round(r['accuracy']['pct'], 2), [round(x, 2) for x in r['accuracy']['ci95']],
              '| FA', round(r['NORMAL']['false_alarm']['pct'], 2), [round(x, 2) for x in r['NORMAL']['false_alarm']['ci95']],
              '| DEMAG miss', round(r['DEMAG']['missed_as_normal']['pct'], 2), [round(x, 2) for x in r['DEMAG']['missed_as_normal']['ci95']],
              '| all miss', round(r['ALL_FAULTS']['missed_as_normal']['pct'], 2))
        for f in ['ECC10', 'ECC20', 'DEMAG', 'REDUC']:
            print('   ', f, {k: round(v['pct'], 2) for k, v in r[f].items()})
    print('diff', {k: (round(v['pctp'], 2), [round(x, 2) for x in v['ci95']]) for k, v in diff.items()})
    print('noninferiority', ni)
    print('paired', paired)
    print('composition', {n: (v['errors'], v['pairs']) for n, v in res['error_composition'].items()})


if __name__ == '__main__':
    main()
