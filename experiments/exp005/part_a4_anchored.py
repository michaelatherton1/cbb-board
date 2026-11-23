"""Score the production pricer (cbb/ladder.py) on 2021-2023 closes, favorite perspective, rungs within +-30.5.

'anchored' = center chosen so the closing line itself is a 50/50 proposition (what price_ladder does with a
de-vigged -110/-110 book line); 'c=line' = center at the closing line (how Part A fitted the shape).
"""
import sys
sys.path.insert(0, "/home/claude/wt/exp-005-kalshi-ladder/cbb")
import numpy as np, pandas as pd
from scipy.stats import norm
import ladder
from part_a import data, grid, scores, TEST

pd.set_option("display.width", 250)
d = data()
te = d[d.season.isin(TEST)].copy()
fm = te.m.abs().to_numpy(); fmar = np.where(te.m >= 0, 1, -1) * te.margin.to_numpy()
T = te.total_close.fillna(141).to_numpy(); day = te.day.to_numpy(); neu = te.neutral.to_numpy()
idx, t, k, y = grid(fm, fmar, kmax=30.5)
P = {"N_const_11.14": 1 - norm.cdf((t - fm[idx]) / 11.143)}
for mode in ("c=line", "anchored"):
    surv_rows = np.empty(len(t))
    for i in range(len(te)):
        kw = dict(total=T[i], day=int(day[i]), neutral=bool(neu[i]))
        c = fm[i] if mode == "c=line" else ladder.implied_center("spread", "home", -fm[i], 0.5, **kw)
        dist = ladder.make_dist("spread", c, **kw)
        n, p = dist.pmf()
        sv = 1 - np.cumsum(p)
        sel = idx == i
        j = np.clip(np.floor(t[sel]).astype(int) - n[0], 0, len(n) - 1)
        surv_rows[sel] = sv[j]
    P[f"ladder.py {mode}"] = surv_rows
# totals: anchored == c=line for half-point lines; score ladder.py's total dist
tt = te[te.total_close.notna()]
Tc = tt.total_close.to_numpy(); am = tt.m.abs().to_numpy()
i2, t2, k2, y2 = grid(Tc, tt.total.to_numpy(), kmax=40.5)
pt = np.empty(len(t2))
for i in range(len(tt)):
    dist = ladder.make_dist("total", Tc[i], abs_spread=am[i], day=int(tt.day.iloc[i]), neutral=bool(tt.neutral.iloc[i]))
    n, p = dist.pmf(); sv = 1 - np.cumsum(p); sel = i2 == i
    pt[sel] = sv[np.clip(np.floor(t2[sel]).astype(int) - n[0], 0, len(n) - 1)]
Pt = {"N_const_16.90": 1 - norm.cdf(k2 / 16.904), "ladder.py": pt}

print("SPREAD (favorite perspective), 2021-23")
for name, sel in [("|k|<=15.5", np.abs(k) <= 15.5), ("|k|<=30.5", np.ones(len(k), bool)), ("|k|>=10", np.abs(k) >= 10)]:
    print(name, pd.DataFrame({m: scores(p[sel], y[sel]) for m, p in P.items()}).T.round(5).to_dict("index"))
kb = np.where(k >= 0, np.floor(k) + 0.5, np.ceil(k) - 0.5)
df = pd.DataFrame(dict(kb=kb, y=y, **P)); g = df.groupby("kb")
tab = g.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
for c in P: tab[c] = g[c].mean()
tab.to_csv("calib_spread_fav_production.csv")
sel_k = [-25.5, -20.5, -15.5, -10.5, -5.5, -2.5, -0.5, 0.5, 2.5, 5.5, 10.5, 15.5, 20.5, 25.5]
print((tab.loc[tab.index.isin(sel_k)]).round(4).to_string())
print("\nTOTAL, 2021-23")
for name, sel in [("|k|<=15.5", np.abs(k2) <= 15.5), ("|k|<=40.5", np.ones(len(k2), bool)), ("|k|>=15", np.abs(k2) >= 15)]:
    print(name, pd.DataFrame({m: scores(p[sel], y2[sel]) for m, p in Pt.items()}).T.round(5).to_dict("index"))
kb2 = np.where(k2 >= 0, np.floor(k2) + 0.5, np.ceil(k2) - 0.5)
df = pd.DataFrame(dict(kb=kb2, y=y2, **Pt)); g = df.groupby("kb")
tab2 = g.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
for c in Pt: tab2[c] = g[c].mean()
tab2.to_csv("calib_total_production.csv")
print(tab2.loc[tab2.index.isin(sel_k + [-30.5, 30.5, 35.5])].round(4).to_string())
