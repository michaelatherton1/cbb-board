"""Tail check: split Student-t (variance-matched to the fitted split-normal sigmas) vs split normal.
Fit on 2018-2020 (key weights refit per df), score 2021-2023 favorite-perspective rungs and totals."""
import numpy as np, pandas as pd
from scipy.stats import norm, t as tdist
from part_a import data, X_spread, X_total, grid, scores, weight_vec, TRAIN, TEST
from part_a3 import fit_split, fav
pd.set_option("display.width", 250)
NMAX = 140
d = data()
tr, te = d[d.season.isin(TRAIN)].copy(), d[d.season.isin(TEST)].copy()
Xtr, Xte = X_spread(tr), X_spread(te)
fm_tr, fmar_tr = fav(tr); fm_te, fmar_te = fav(te)
ths = fit_split(fmar_tr - fm_tr, Xtr); p = Xtr.shape[1]
sL_tr, sR_tr = np.exp(Xtr @ ths[:p]), np.exp(Xtr @ ths[p:]); sL_te, sR_te = np.exp(Xte @ ths[:p]), np.exp(Xte @ ths[p:])

def cdf(x, sL, sR, nu):
    if nu is None:
        return np.where(x < 0, norm.cdf(x / sL), norm.cdf(x / sR))
    f = np.sqrt((nu - 2) / nu)
    return np.where(x < 0, tdist.cdf(x / (sL * f), nu), tdist.cdf(x / (sR * f), nu))

def pmf(c, sL, sR, nu, w=None):
    n = np.arange(-NMAX, NMAX + 1); c, a, b = (np.asarray(v, float)[:, None] for v in (c, sL, sR))
    q = cdf(n + 0.5 - c, a, b, nu) - cdf(n - 0.5 - c, a, b, nu)
    if w is not None: q = q * weight_vec(n, w)
    return n, q / q.sum(1, keepdims=True)

def keyw(c, sL, sR, res, nu, K=12):
    n, q0 = pmf(c, sL, sR, nu); w = np.ones(K + 1)
    obs = np.array([(np.abs(res) == j).sum() for j in range(K + 1)], float)
    for _ in range(6):
        q = q0 * weight_vec(n, w); q /= q.sum(1, keepdims=True)
        e = np.array([q[:, np.abs(n) == j].sum() for j in range(K + 1)]); w *= np.where(e > 0, obs / np.maximum(e, 1e-12), 1.0)
    return w

idx, t, k, y = grid(fm_te, fmar_te, kmax=30.5)
P = {}
for nu in (None, 30, 15, 10, 7):
    w = keyw(fm_tr, sL_tr, sR_tr, fmar_tr, nu)
    n, q = pmf(fm_te, sL_te, sR_te, nu, w); sv = 1 - np.cumsum(q, 1)
    P[f"nu={nu or 'inf'}"] = sv[idx, np.clip(np.floor(t).astype(int) - n[0], 0, len(n) - 1)]
print("SPREAD fav perspective 2021-23")
for name, sel in [("|k|<=15.5", np.abs(k) <= 15.5), ("|k|<=30.5", np.ones(len(k), bool)), ("|k|>=15", np.abs(k) >= 15)]:
    print(name, {m: round(scores(pp[sel], y[sel])["logloss"], 5) for m, pp in P.items()})
kb = np.where(k >= 0, np.floor(k) + 0.5, np.ceil(k) - 0.5)
df = pd.DataFrame(dict(kb=kb, y=y, **P)); g = df.groupby("kb"); tab = g.y.agg(["mean"]).rename(columns={"mean": "obs"})
for c in P: tab[c] = g[c].mean()
print(tab.loc[[-25.5, -20.5, -15.5, -10.5, -5.5, -0.5, 0.5, 5.5, 10.5, 15.5, 20.5, 25.5, 30.5]].round(4).to_string())

# totals
dt = d[d.total_close.notna()]; tr, te = dt[dt.season.isin(TRAIN)], dt[dt.season.isin(TEST)]
th = np.array([2.8407, 0.0391, -0.0449, 0.0648, -0.0066]); kL, kR = 0.9339, 1.0191
s_te = np.exp(X_total(te) @ th); T = te.total_close.to_numpy()
i2, t2, k2, y2 = grid(T, te.total.to_numpy(), kmax=40.5)
Pt = {}
for nu in (None, 30, 15, 10, 7):
    Pt[f"nu={nu or 'inf'}"] = 1 - cdf(k2, s_te[i2] * kL, s_te[i2] * kR, nu)
print("\nTOTAL 2021-23")
for name, sel in [("|k|<=15.5", np.abs(k2) <= 15.5), ("|k|<=40.5", np.ones(len(k2), bool)), ("|k|>=15", np.abs(k2) >= 15)]:
    print(name, {m: round(scores(pp[sel], y2[sel])["logloss"], 5) for m, pp in Pt.items()})
kb = np.where(k2 >= 0, np.floor(k2) + 0.5, np.ceil(k2) - 0.5)
df = pd.DataFrame(dict(kb=kb, y=y2, **Pt)); g = df.groupby("kb"); tab = g.y.agg(["mean"]).rename(columns={"mean": "obs"})
for c in Pt: tab[c] = g[c].mean()
print(tab.loc[[-30.5, -20.5, -10.5, -0.5, 0.5, 10.5, 20.5, 25.5, 30.5, 35.5, 40.5]].round(4).to_string())
