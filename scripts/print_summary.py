"""results/reliability_metrics_v3.json 핵심 수치를 짧게 출력한다."""
import json
import sys
from pathlib import Path

d = json.loads((Path(__file__).resolve().parents[1] / 'results/reliability_metrics_v3.json').read_text(encoding='utf-8'))
F = ['ECC10', 'ECC20', 'DEMAG', 'REDUC']


def ci(v):
    return f"{v['pct']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] ({v['n_num']}/{v['n_den']})"


def show(tables, title):
    print('#', title)
    for m in ['87ch', 'AIHub']:
        r = tables[m]
        print(f"== {m}: acc {ci(r['accuracy'])}")
        print(f"   false alarm  {ci(r['NORMAL']['false_alarm'])}")
        print(f"   miss(all)    {ci(r['ALL_FAULTS']['missed_as_normal'])}")
        for f in F:
            print(f"   {f:6s} correct {ci(r[f]['correct_mode'])} | miss {ci(r[f]['missed_as_normal'])} | wrong {ci(r[f]['wrong_fault_mode'])}")


show(d['reliability'], 'ALL TEST')
print('\n# diff 87-AI', {k: (round(v['pctp'], 2), [round(x, 2) for x in v['ci95']]) for k, v in d['difference_87ch_minus_AIHub'].items()})
print('\n# composition', json.dumps(d['error_composition'], ensure_ascii=False))
for key in ['by_rpm_std_decile', 'by_torque_act_std_decile']:
    for m in ['87ch', 'AIHub']:
        print(key, m, [(round(r['lo'], 1), round(r['hi'], 1), round(r['err_pct'], 2)) for r in d[key][m]])
print('\n# session vote', json.dumps(d['session_majority_vote']))
g = d['selective_diagnosis']
print('\n# gate', g['thresholds'], 'coverage', round(g['coverage_pct_all'], 1), {k: round(v, 1) for k, v in g['coverage_pct_by_class'].items()},
      'sess_any', round(g['sessions_with_any_eligible_pct'], 1), 'median', round(g['eligible_share_per_session_median_pct'], 1))
show(g['gated'], 'GATED')
print('\n# gated diff', {k: (round(v['pctp'], 2), [round(x, 2) for x in v['ci95']]) for k, v in g['difference_87ch_minus_AIHub_gated'].items()})
