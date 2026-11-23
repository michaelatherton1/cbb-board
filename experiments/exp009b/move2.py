import pandas as pd, numpy as np
exec(open('move.py').read().split('# 1)')[0].split('def ols')[0])
d = d.sort_values('date')
# bias correction from PRIOR dates only: expanding mean of (open - model) through the previous day
daily = d.groupby('date').apply(lambda x: (x.total_open - x.pred_total).sum(), include_groups=False)
cnt = d.groupby('date').size()
prior = (daily.cumsum().shift(1) / cnt.cumsum().shift(1)).fillna(0)
d['adj'] = d.date.map(prior)
d['m_adj'] = d.pred_total + d.adj - d.total_open
print("bias adj range", round(prior.min(), 2), round(prior.max(), 2))
for col in ['m_open', 'm_adj']:
    s = d[d.date > d.date.min() + pd.Timedelta(days=3)]
    sl = np.polyfit(s[col], s.move, 1)[0]
    print(f"\n{col}: slope {sl:.3f}, corr {s.move.corr(s[col]):.3f}, n={len(s)}")
    for lo in [2, 3, 5]:
        for sgn, name in [(1, 'model above open (over)'), (-1, 'model below open (under)')]:
            x = s[(s[col] * sgn) >= lo]; mv = x.move[x.move != 0]
            print(f"   {name:26s} |diff|>={lo}: n={len(x):3d}  moved toward model {((np.sign(mv) == sgn)).mean():.3f}  "
                  f"mean move {(x.move * sgn).mean():+.2f}  mean (actual-open) {(x.res_open * sgn).mean():+.2f}  (actual-close) {(x.res_close * sgn).mean():+.2f}")
