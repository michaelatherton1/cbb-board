"""Part A, final model selection.

Spreads: favorite-perspective split normal (median at the line, separate left/right sigma, both log-linear in
features) + integer key-number weights, vs the symmetric versions.
Totals: split normal around the line (left/right multipliers on a heteroscedastic sigma), vs Normal and the
scaled empirical shape.
All fitted on 2018-2020 closes, scored on 2021-2023 closes over Kalshi-style half-point rungs.
"""
import json
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from part_a import (data, X_spread, X_total, fit_logsigma, fit_key_weights, weight_vec, grid, scores,
                    scaled_emp_surv, TRAIN, TEST)

pd.set_option("display.width", 250)
OUT = "/home/claude/scratch/exp005"
NMAX = 140


def fit_split(r, X):
    """Split normal, median 0: log sL = X thL, log sR = X thR (density MLE)."""
    p = X.shape[1]
    def nll(th):
        lsL, lsR = X @ th[:p], X @ th[p:]
        neg = r < 0
        ls = np.where(neg, lsL, lsR)
        return np.mean(ls + 0.5 * r ** 2 * np.exp(-2 * ls))
    th0 = np.zeros(2 * p); th0[0] = th0[p] = np.log(r.std())
    return minimize(nll, th0, method="BFGS").x


def split_cdf(x, sL, sR):
    """CDF of a split normal with median 0."""
    return np.where(x < 0, norm.cdf(x / sL), norm.cdf(x / sR))


def pmf_split(center, sL, sR, w=None):
    """Integer pmf of center + split-normal noise (median at center), times key weights on |n|."""
    n = np.arange(-NMAX, NMAX + 1)
    c, a, b = (np.asarray(v, float)[:, None] for v in (center, sL, sR))
    p = split_cdf(n + 0.5 - c, a, b) - split_cdf(n - 0.5 - c, a, b)
    if w is not None:
        p = p * weight_vec(n, w)
    return n, p / p.sum(1, keepdims=True)


def surv_split(center, sL, sR, t, w=None, idx=None):
    n, p = pmf_split(center, sL, sR, w)
    surv = 1 - np.cumsum(p, 1)
    if idx is None:
        idx = np.arange(len(center))
    j = np.clip(np.floor(np.asarray(t)).astype(int) - n[0], 0, len(n) - 1)
    return surv[idx, j]


def fit_key_weights_split(center, sL, sR, result, K=12, iters=6):
    n, p = pmf_split(center, sL, sR)
    w = np.ones(K + 1)
    obs = np.array([(np.abs(result) == j).sum() for j in range(K + 1)], float)
    for _ in range(iters):
        q = p * weight_vec(n, w); q /= q.sum(1, keepdims=True)
        exp = np.array([q[:, np.abs(n) == j].sum() for j in range(K + 1)])
        w = w * np.where(exp > 0, obs / np.maximum(exp, 1e-12), 1.0)
    return w


def fav(df):
    s = np.where(df.m >= 0, 1.0, -1.0)
    return np.abs(df.m.to_numpy()), s * df.margin.to_numpy()


def bucket_table(k, y, P, edges, labels):
    kb = np.select([k <= e for e in edges], labels[:-1], labels[-1])
    df = pd.DataFrame(dict(kb=kb, y=y, **P)); g = df.groupby("kb")
    tab = g.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
    for c in P:
        tab[c] = g[c].mean() - tab.obs
    return tab


def spreads(d):
    tr, te = d[d.season.isin(TRAIN)].copy(), d[d.season.isin(TEST)].copy()
    Xtr, Xte = X_spread(tr), X_spread(te)
    fm_tr, fmar_tr = fav(tr); fm_te, fmar_te = fav(te)
    # symmetric
    th = fit_logsigma(tr.r.to_numpy(), Xtr)
    sig_tr, sig_te = np.exp(Xtr @ th), np.exp(Xte @ th)
    w_sym = fit_key_weights(tr.m.to_numpy(), sig_tr, tr.margin.to_numpy())
    # split, favorite perspective (|m| features are already sign-free)
    rf_tr = fmar_tr - fm_tr
    ths = fit_split(rf_tr, Xtr)
    p = Xtr.shape[1]
    sL_tr, sR_tr = np.exp(Xtr @ ths[:p]), np.exp(Xtr @ ths[p:])
    sL_te, sR_te = np.exp(Xte @ ths[:p]), np.exp(Xte @ ths[p:])
    w_split = fit_key_weights_split(fm_tr, sL_tr, sR_tr, fmar_tr)
    # split without the key weights, for reference
    idx, t, k, y = grid(fm_te, fmar_te, kmax=30.5)
    from part_a import surv_discrete
    P = {"N_const": 1 - norm.cdf((t - fm_te[idx]) / np.sqrt(np.mean(tr.r ** 2))),
         "Nkey_het": surv_discrete(fm_te, sig_te, t, w_sym, idx),
         "Split_het": surv_split(fm_te, sL_te, sR_te, t, None, idx),
         "SplitKey_het": surv_split(fm_te, sL_te, sR_te, t, w_split, idx)}
    res = {}
    for name, sel in [("|k|<=15.5", np.abs(k) <= 15.5), ("|k|<=30.5", np.ones(len(k), bool)),
                      ("|k|>=10", np.abs(k) >= 10), ("|t|<=2.5", np.abs(t) <= 2.5)]:
        res[name] = {m: scores(pp[sel], y[sel]) for m, pp in P.items()}
    tab = bucket_table(k, y, P, [-20, -15, -10, -5, -0.01, 5, 10, 15, 20],
                       ["<=-20", "(-20,-15]", "(-15,-10]", "(-10,-5]", "(-5,0)", "[0,5)", "[5,10)", "[10,15)", "[15,20)", ">=20"])
    # per |m| bucket, far rungs
    mb = pd.cut(fm_te[idx], [-0.1, 5.5, 12.5, 20.5, 99]).astype(str)
    side = np.where(k >= 10, "fav by line+10 or more", np.where(k <= -10, "fav short by 10+ (P fav > line-10..)", "mid"))
    df = pd.DataFrame(dict(mb=mb, side=side, y=y, **P)); df = df[df.side != "mid"]
    g = df.groupby(["side", "mb"]); t2 = g.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
    for c in P:
        t2[c] = g[c].mean() - t2.obs
    # calibration by k (1-pt rows) for the report
    kb1 = np.where(k >= 0, np.floor(k) + 0.5, np.ceil(k) - 0.5)
    df1 = pd.DataFrame(dict(kb=kb1, y=y, **P)); g1 = df1.groupby("kb")
    t1 = g1.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
    for c in P:
        t1[c] = g1[c].mean()
    params = dict(theta_sym=th.tolist(), key_w_sym=w_sym.tolist(), theta_L=ths[:p].tolist(), theta_R=ths[p:].tolist(),
                  key_w_split=w_split.tolist(), features=["const", "(T-140)/10", "|m|/10", "(|m|/10)^2", "exp(-day/21)", "neutral"])
    return res, tab, t2, t1, params


def totals(d):
    d = d[d.total_close.notna()]
    tr, te = d[d.season.isin(TRAIN)].copy(), d[d.season.isin(TEST)].copy()
    Xtr, Xte = X_total(tr), X_total(te)
    th = fit_logsigma(tr.rt.to_numpy(), Xtr)
    sig_tr, sig_te = np.exp(Xtr @ th), np.exp(Xte @ th)
    # split normal, both sides log-linear (density MLE)
    ths = fit_split(tr.rt.to_numpy(), Xtr)
    p = Xtr.shape[1]
    sL_te, sR_te = np.exp(Xte @ ths[:p]), np.exp(Xte @ ths[p:])
    # split by multipliers fitted on the train ladder grid (event log loss), robust to OT outliers
    T_tr = tr.total_close.to_numpy()
    i0, t0, k0, y0 = grid(T_tr, tr.total.to_numpy(), kmax=30.5)
    s0 = sig_tr[i0]
    def ll(v):
        a, b = v
        pr = np.where(k0 >= 0, 1 - norm.cdf(k0 / (s0 * b)), 1 - norm.cdf(k0 / (s0 * a)))
        return scores(pr, y0)["logloss"]
    best = minimize(ll, [0.95, 1.0], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-7)).x
    kL, kR = float(best[0]), float(best[1])
    T = te.total_close.to_numpy()
    idx, t, k, y = grid(T, te.total.to_numpy(), kmax=40.5)
    s = sig_te[idx]
    z = (tr.rt / sig_tr).to_numpy()
    z_med = z - np.median(z)
    P = {"N_het": 1 - norm.cdf(k / s),
         "SplitMLE_het": np.where(k >= 0, 1 - norm.cdf(k / sR_te[idx]), 1 - norm.cdf(k / sL_te[idx])),
         "SplitGrid_het": np.where(k >= 0, 1 - norm.cdf(k / (s * kR)), 1 - norm.cdf(k / (s * kL))),
         "Zemp_het_medctr": scaled_emp_surv(z_med, k / s)}
    res = {}
    for name, sel in [("|k|<=15.5", np.abs(k) <= 15.5), ("|k|<=40.5", np.ones(len(k), bool)), ("|k|>=15", np.abs(k) >= 15)]:
        res[name] = {m: scores(pp[sel], y[sel]) for m, pp in P.items()}
    tab = bucket_table(k, y, P, [-25, -15, -10, -5, -0.01, 5, 10, 15, 25],
                       ["<=-25", "(-25,-15]", "(-15,-10]", "(-10,-5]", "(-5,0)", "[0,5)", "[5,10)", "[10,15)", "[15,25)", ">=25"])
    kb1 = np.where(k >= 0, np.floor(k) + 0.5, np.ceil(k) - 0.5)
    df1 = pd.DataFrame(dict(kb=kb1, y=y, **P)); g1 = df1.groupby("kb")
    t1 = g1.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
    for c in P:
        t1[c] = g1[c].mean()
    params = dict(theta=th.tolist(), kL=kL, kR=kR, theta_L=ths[:p].tolist(), theta_R=ths[p:].tolist(),
                  features=["const", "(T-140)/10", "|m|/10", "exp(-day/21)", "neutral"])
    return res, tab, t1, params


if __name__ == "__main__":
    d = data()
    res, tab, t2, t1s, ps = spreads(d)
    print("=== SPREAD (favorite perspective), 2021-23 ===")
    for k, v in res.items():
        print(k); print(pd.DataFrame(v).T.round(5))
    print("pred - obs by k bucket"); print(tab.round(4).to_string())
    print(t2.round(4).to_string())
    print(json.dumps(ps, indent=0))
    rt, tabt, t1t, pt = totals(d)
    print("\n=== TOTAL, 2021-23 ===")
    for k, v in rt.items():
        print(k); print(pd.DataFrame(v).T.round(5))
    print("pred - obs by k bucket"); print(tabt.round(4).to_string())
    print(json.dumps(pt, indent=0))
    t1s.to_csv(f"{OUT}/calib_spread_fav_by_k.csv"); t1t.to_csv(f"{OUT}/calib_total_by_k.csv")
    json.dump(dict(spread=ps, total=pt), open(f"{OUT}/final_params.json", "w"), indent=1)
