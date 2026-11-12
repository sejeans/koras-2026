"""v3 공정 비교 시험 데이터(16,046건, 391세션)에서 신뢰성 지표를 다시 계산한다.

재학습 없음. 입력은 이미 저장된 두 모델의 예측 파일과 CAN 요약 캐시뿐이다.

계산하는 것
1. 고장 모드별 검출률, 미검출률(고장을 정상으로 판정), 오식별률(다른 고장으로 판정)
2. 오경보율(정상을 고장으로 판정)
3. 87채널 오류 중 감자 결함과 정상 사이 오류의 비중
4. 운전 조건(회전수 변동, 토크 변동)별 오류율
5. 선택적 진단: 임계값을 학습 데이터(inner_train)로만 정한 뒤 시험 데이터에 적용
6. 세션 단위 판정(세션 다수결)

불확실성은 세션(차종/측정 timestamp)을 재표집하는 bootstrap 10,000회로 구한다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = WORK.parent
V3 = ROOT / 'experiments/2026-10-09_fair_deadline_v3'
CANDIR = ROOT / 'experiments/2026-09-15_CAN_fusion/evidence/can_fusion_full'
AIHUB = ROOT / 'AI모델/EV_Model_work/deadline_v3/test/predictions.csv'
OUT = WORK / 'results'
B = 10_000
SEED = 42
FAULTS = ['ECC10', 'ECC20', 'DEMAG', 'REDUC']
CLASSES = ['NORMAL'] + FAULTS


def load():
    member = json.loads((V3 / 'membership.json').read_text(encoding='utf-8'))['rows']
    cols = json.loads((CANDIR / 'config.json').read_text(encoding='utf-8'))['variants']['all25']
    ctx = []
    for v in ['IONIQ', 'KONA', 'NIRO']:
        sel = pd.read_parquet(CANDIR / v / 'selected.parquet')
        can = np.load(CANDIR / v / 'can.npy')
        t = sel[['sample_id', 'vehicle', 'fault_class']].copy()
        for i, c in enumerate(cols):
            t[c] = can[:, i]
        ctx.append(t)
    ctx = pd.concat(ctx, ignore_index=True)
    ctx['role'] = ctx.sample_id.map(lambda s: member[s]['role'])
    ctx['session_id'] = ctx.sample_id.map(lambda s: member[s]['vehicle'] + '/' + member[s]['session'])
    p87 = pd.read_parquet(V3 / 'as_analysis/probabilities_16046.parquet')[['sample_id', 'predicted', 'P_NORMAL']]
    ai = pd.read_csv(AIHUB)[['sample_id', 'actual', 'predicted']]
    test = ctx[ctx.role == 'outer_test'].merge(p87.rename(columns={'predicted': 'pred_87'}),
                                              on='sample_id', validate='one_to_one')
    test = test.merge(ai.rename(columns={'predicted': 'pred_ai'}), on='sample_id', validate='one_to_one')
    assert len(test) == 16046 and (test.actual == test.fault_class).all()
    assert test.session_id.nunique() == 391
    return ctx, test


class Boot:
    """세션 군집 bootstrap. 비율 지표를 (분자, 분모) 세션 합으로 계산한다."""

    def __init__(self, sessions):
        self.codes, self.uniq = pd.factorize(sessions)
        n = len(self.uniq)
        rng = np.random.default_rng(SEED)
        self.w = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.float32)

    def per_session(self, mask):
        return np.bincount(self.codes, weights=mask.astype(float), minlength=len(self.uniq))

    def rate(self, num_mask, den_mask):
        num, den = self.per_session(num_mask & den_mask), self.per_session(den_mask)
        point = num.sum() / den.sum()
        bn, bd = self.w @ num, self.w @ den
        draws = bn / np.where(bd == 0, np.nan, bd)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        return {'pct': 100 * point, 'ci95': [100 * lo, 100 * hi], 'n_num': int(num.sum()), 'n_den': int(den.sum())}, draws

    def diff(self, a, b):
        """같은 표본에 대한 두 비율 지표의 차이(a-b), 같은 재표집을 공유한다."""
        (ra, da), (rb, db) = a, b
        d = da - db
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return {'pctp': ra['pct'] - rb['pct'], 'ci95': [100 * lo, 100 * hi]}


def reliability_table(df, pred, boot):
    y = df.fault_class.to_numpy()
    p = df[pred].to_numpy()
    rows, raw = {}, {}
    for f in FAULTS:
        is_f = y == f
        rows[f] = {}
        for name, m in [('correct_mode', p == f), ('detected_any_fault', p != 'NORMAL'),
                        ('missed_as_normal', p == 'NORMAL'),
                        ('wrong_fault_mode', (p != 'NORMAL') & (p != f))]:
            r, d = boot.rate(m, is_f)
            rows[f][name] = r
            raw[(f, name)] = (r, d)
    is_n = y == 'NORMAL'
    r, d = boot.rate(p != 'NORMAL', is_n)
    rows['NORMAL'] = {'false_alarm': r, 'false_alarm_as_DEMAG': boot.rate(p == 'DEMAG', is_n)[0]}
    raw[('NORMAL', 'false_alarm')] = (r, d)
    is_fault = y != 'NORMAL'
    r, d = boot.rate(p == 'NORMAL', is_fault)
    rows['ALL_FAULTS'] = {'missed_as_normal': r}
    raw[('ALL', 'missed')] = (r, d)
    r, d = boot.rate(p == y, np.ones(len(y), bool))
    rows['accuracy'] = r
    raw[('ALL', 'accuracy')] = (r, d)
    return rows, raw


def error_composition(df, pred):
    y, p = df.fault_class.to_numpy(), df[pred].to_numpy()
    err = y != p
    pairs = pd.Series([f'{a}->{b}' for a, b in zip(y[err], p[err])]).value_counts()
    dn = int(((y == 'DEMAG') & (p == 'NORMAL')).sum() + ((y == 'NORMAL') & (p == 'DEMAG')).sum())
    return {'errors': int(err.sum()), 'error_rate_pct': 100 * err.mean(),
            'demag_normal_both_directions': dn,
            'demag_normal_share_pct': 100 * dn / err.sum(),
            'pairs': {k: int(v) for k, v in pairs.items()}}


def by_condition(df, pred, col, q=10):
    bins = pd.qcut(df[col], q, labels=False, duplicates='drop')
    g = df.assign(err=(df.fault_class != df[pred]), b=bins).groupby('b')
    t = g.agg(lo=(col, 'min'), hi=(col, 'max'), n=('err', 'size'), err_pct=('err', 'mean'))
    t['err_pct'] *= 100
    return t.reset_index().to_dict(orient='records')


def session_level(df, pred):
    """세션 다수결: 세션 안 표본 판정 중 가장 많은 클래스를 세션 판정으로 쓴다."""
    out = {}
    for sid, g in df.groupby('session_id'):
        out[sid] = {'actual': g.fault_class.iloc[0], 'vote': g[pred].value_counts().idxmax(),
                    'n': len(g)}
    s = pd.DataFrame(out).T
    res = {}
    for c in CLASSES:
        t = s[s.actual == c]
        res[c] = {'sessions': len(t), 'vote_correct': int((t.vote == c).sum()),
                  'vote_normal': int((t.vote == 'NORMAL').sum())}
    return res


def main():
    ctx, test = load()
    boot = Boot(test.session_id.to_numpy())
    result = {'scope': 'v3 outer test 16,046 samples / 391 sessions, fixed predictions, no retraining',
              'bootstrap': f'session cluster bootstrap B={B}, seed={SEED}, percentile 95% CI'}
    tables, raws = {}, {}
    for name, col in [('87ch', 'pred_87'), ('AIHub', 'pred_ai')]:
        tables[name], raws[name] = reliability_table(test, col, boot)
    result['reliability'] = tables
    result['difference_87ch_minus_AIHub'] = {
        f'{k[0]}:{k[1]}': boot.diff(raws['87ch'][k], raws['AIHub'][k]) for k in raws['87ch']}
    result['error_composition'] = {n: error_composition(test, c) for n, c in [('87ch', 'pred_87'), ('AIHub', 'pred_ai')]}
    result['by_rpm_std_decile'] = {n: by_condition(test, c, 'rpm_std') for n, c in [('87ch', 'pred_87'), ('AIHub', 'pred_ai')]}
    result['by_torque_act_std_decile'] = {n: by_condition(test, c, 'torque_act_std') for n, c in [('87ch', 'pred_87'), ('AIHub', 'pred_ai')]}
    result['session_majority_vote'] = {n: session_level(test, c) for n, c in [('87ch', 'pred_87'), ('AIHub', 'pred_ai')]}

    # 선택적 진단: 임계값은 inner_train 표본의 차종별 torque_act_std 중앙값. 시험 데이터는 쓰지 않는다.
    train = ctx[ctx.role == 'inner_train']
    thr = train.groupby('vehicle').torque_act_std.median().to_dict()
    gate = test.torque_act_std.to_numpy() >= test.vehicle.map(thr).to_numpy()
    gated = test[gate]
    gb = Boot(gated.session_id.to_numpy())
    g_tables, g_raw = {}, {}
    for name, col in [('87ch', 'pred_87'), ('AIHub', 'pred_ai')]:
        g_tables[name], g_raw[name] = reliability_table(gated, col, gb)
    cov = {c: float(gate[test.fault_class == c].mean() * 100) for c in CLASSES}
    sess = test.assign(g=gate).groupby('session_id').g.agg(['any', 'mean'])
    result['selective_diagnosis'] = {
        'rule': 'diagnose only when torque_act_std >= per-vehicle median of inner_train samples',
        'thresholds': thr,
        'coverage_pct_all': float(gate.mean() * 100),
        'coverage_pct_by_class': cov,
        'sessions_with_any_eligible_pct': float(sess['any'].mean() * 100),
        'eligible_share_per_session_median_pct': float(sess['mean'].median() * 100),
        'gated': g_tables,
        'difference_87ch_minus_AIHub_gated': {
            f'{k[0]}:{k[1]}': gb.diff(g_raw['87ch'][k], g_raw['AIHub'][k]) for k in g_raw['87ch']},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / 'reliability_metrics_v3.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps({'reliability': tables, 'composition': result['error_composition'],
                      'gate': {k: v for k, v in result['selective_diagnosis'].items() if k != 'difference_87ch_minus_AIHub_gated'}},
                     ensure_ascii=False, indent=1, default=float)[:12000])


if __name__ == '__main__':
    main()
