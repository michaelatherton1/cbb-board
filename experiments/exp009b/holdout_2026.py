"""EXP-009b pre-registered test on the 2026 season. Run once."""
import pandas as pd, numpy as np, sys
src = sys.argv[1]
b = pd.read_parquet(src)
d = b[(b.season == 2026) & b.total_open.notna() & b.total_close.notna()].copy()
if 'd1' in d: d = d[d.d1]
d['date'] = pd.to_datetime(d.date)
d = d[d.date > d.date.min() + pd.Timedelta(days=3)].sort_values('date') if False else d.sort_values('date')
daily = d.groupby('date').apply(lambda x: (x.total_open - x.pred_total).sum(), include_groups=False)
cnt = d.groupby('date').size()
prior = (daily.cumsum().shift(1) / cnt.cumsum().shift(1)).fillna(0)
d['m_adj'] = d.pred_total + d.date.map(prior) - d.total_open
d['move'] = d.total_close - d.total_open
d = d[d.date > d.date.min() + pd.Timedelta(days=3)]
slope = np.polyfit(d.m_adj, d.move, 1)[0]
g = [x for _, x in d.groupby('date')]; rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    s = pd.concat([g[i] for i in rng.integers(0, len(g), len(g))]); bs.append(np.polyfit(s.m_adj, s.move, 1)[0])
lo, hi = np.percentile(bs, [2.5, 97.5])
big = d[d.m_adj.abs() >= 3]; mv = big[big.move != 0]
toward = (np.sign(mv.move) == np.sign(mv.m_adj)).mean()
print(f"2026 games n={len(d)} dates {d.date.min().date()}..{d.date.max().date()}  bias-adj range {prior.min():.2f}..{prior.max():.2f}")
print(f"1) slope {slope:.3f}  95% CI by date [{lo:.3f}, {hi:.3f}]  -> {'PASS' if lo > 0 else 'FAIL'}")
print(f"2) |m_adj|>=3: n={len(big)}, moved={len(mv)}, toward model {toward:.3f} -> {'PASS' if toward >= 0.55 else 'FAIL'}")
sg = np.sign(big.m_adj)
print(f"   reported: mean move toward model {(big.move * sg).mean():+.2f} pts; (actual - close) toward model {((big.total - big.total_close) * sg).mean():+.2f}")
for s_, name in [(1, 'over'), (-1, 'under')]:
    x = mv[np.sign(mv.m_adj) == s_]; print(f"   {name}: moved n={len(x)}, toward {(np.sign(x.move) == s_).mean():.3f}")
print("CONFIRMED" if (lo > 0 and toward >= 0.55) else "NOT CONFIRMED")
