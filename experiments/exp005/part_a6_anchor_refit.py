"""Refit the spread shape under the production convention: the center is the point at which the closing line
is a 50/50 proposition (as price_ladder does with a de-vigged book price), not the line itself.

Iterate: centers given shape -> split-normal MLE + key weights given centers. Fit 2018-2020, score 2021-2023.
"""
import json
import numpy as np, pandas as pd
from scipy.stats import norm
from part_a import data, X_spread, grid, scores, weight_vec, TRAIN, TEST
from part_a3 import fit_split, fav
pd.set_option("display.width", 250)
NMAX = 140
N = np.arange(-NMAX, NMAX + 1)


def pmf(c, sL, sR, w):
    c, a, b = (np.asarray(v, float)[:, None] for v in (c, sL, sR))
    x1, x0 = N + 0.5 - c, N - 0.5 - c
    F = lambda x: np.where(x < 0, norm.cdf(x / a), norm.cdf(x / b))
    q = (F(x1) - F(x0)) * weight_vec(N, w)
    return q / q.sum(1, keepdims=True)


def cover_nopush(c, sL, sR, w, line):
    q = pmf(c, sL, sR, w)
    win = (q * (N[None, :] > line[:, None])).sum(1)
    lose = (q * (N[None, :] < line[:, None])).sum(1)
    return win / (win + lose)


def anchored(fm, sL, sR, w, deltas=np.arange(-3, 3.001, 0.05)):
    P = np.stack([cover_nopush(fm + dl, sL, sR, w, fm) for dl in deltas], 1)   # increasing in delta
    out = np.empty(len(fm))
    for i in range(len(fm)):
        out[i] = fm[i] + np.interp(0.5, P[i], deltas)
    return out


def keyw(c, sL, sR, res, K=12):
    w = np.ones(K + 1)
    obs = np.array([(np.abs(res) == j).sum() for j in range(K + 1)], float)
    q0 = pmf(c, sL, sR, np.ones(K + 1))
    for _ in range(6):
        q = q0 * weight_vec(N, w); q /= q.sum(1, keepdims=True)
        e = np.array([q[:, np.abs(N) == j].sum() for j in range(K + 1)])
        w = w * np.where(e > 0, obs / np.maximum(e, 1e-12), 1.0)
    return w


d = data()
tr, te = d[d.season.isin(TRAIN)].copy(), d[d.season.isin(TEST)].copy()
Xtr, Xte = X_spread(tr), X_spread(te); p = Xtr.shape[1]
fm_tr, fmar_tr = fav(tr); fm_te, fmar_te = fav(te)
# start: c = line
c_tr = fm_tr.copy()
hist = []
for it in range(4):
    ths = fit_split(fmar_tr - c_tr, Xtr)
    sL, sR = np.exp(Xtr @ ths[:p]), np.exp(Xtr @ ths[p:])
    w = keyw(c_tr, sL, sR, fmar_tr)
    c_new = anchored(fm_tr, sL, sR, w)
    hist.append(dict(it=it, mean_shift=float(np.mean(c_new - fm_tr)), thL=np.round(ths[:p], 4).tolist(),
                     thR=np.round(ths[p:], 4).tolist(), w=np.round(w, 4).tolist()))
    print(hist[-1])
    c_tr = c_new
# final shape from the last centers
ths = fit_split(fmar_tr - c_tr, Xtr); sL_tr, sR_tr = np.exp(Xtr @ ths[:p]), np.exp(Xtr @ ths[p:])
w = keyw(c_tr, sL_tr, sR_tr, fmar_tr)
sL_te, sR_te = np.exp(Xte @ ths[:p]), np.exp(Xte @ ths[p:])
c_te = anchored(fm_te, sL_te, sR_te, w)
idx, t, k, y = grid(fm_te, fmar_te, kmax=30.5)
q = pmf(c_te, sL_te, sR_te, w); sv = 1 - np.cumsum(q, 1)
pr = sv[idx, np.clip(np.floor(t).astype(int) - N[0], 0, len(N) - 1)]
prev = pd.read_csv("calib_spread_fav_production.csv").set_index("kb")
for name, sel in [("|k|<=15.5", np.abs(k) <= 15.5), ("|k|<=30.5", np.ones(len(k), bool)), ("|k|>=10", np.abs(k) >= 10)]:
    print(name, "refit-anchored", scores(pr[sel], y[sel]))
kb = np.where(k >= 0, np.floor(k) + 0.5, np.ceil(k) - 0.5)
df = pd.DataFrame(dict(kb=kb, y=y, refit_anchored=pr)); g = df.groupby("kb")
tab = prev.join(g.refit_anchored.mean())
tab.to_csv("calib_spread_fav_production.csv")
print(tab.loc[[-25.5, -20.5, -15.5, -10.5, -5.5, -2.5, -0.5, 0.5, 2.5, 5.5, 10.5, 15.5, 20.5, 25.5]].round(4).to_string())
json.dump(dict(theta_L=ths[:p].tolist(), theta_R=ths[p:].tolist(), key_w=w.tolist(), history=hist),
          open("spread_params_anchored.json", "w"), indent=1)
