"""Model choice using pre-2021 data only (coordinator rule: choose on 2012-2020, never look at 2021+).

Leave-one-season-out over 2018, 2019, 2020: fit on the other two seasons, score the held-out one.
Only seasons 2018-2020 are loaded. Spreads in favorite perspective (half-point rungs within +-30.5 of the line);
totals over half-point rungs within +-40.5.
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from load_hist import load
from part_a import X_spread, X_total, fit_logsigma, grid, scores, weight_vec
from part_a3 import fit_split, fav
pd.set_option("display.width", 250)
SEASONS = [2018, 2019, 2020]
NMAX = 140
N = np.arange(-NMAX, NMAX + 1)


def data_pre2021():
    d = load(seasons=SEASONS)
    assert d.season.max() <= 2020
    d = d[~(d.sign_disagree & ~d.neutral)]
    d = d[~(d.neutral & d.ml_home_close.isna())]
    d = d[d.total_close.between(100, 200) | d.total_close.isna()]
    d["early"] = np.exp(-d.day / 21.0)
    d["r"] = d.margin - d.m
    d["rt"] = d.total - d.total_close
    return d.reset_index(drop=True)


def pmf(c, sL, sR, w=None):
    c, a, b = (np.asarray(v, float)[:, None] for v in (c, sL, sR))
    F = lambda x: np.where(x < 0, norm.cdf(x / a), norm.cdf(x / b))
    q = F(N + 0.5 - c) - F(N - 0.5 - c)
    if w is not None:
        q = q * weight_vec(N, w)
    return q / q.sum(1, keepdims=True)


def keyw(c, sL, sR, res, K=12):
    w = np.ones(K + 1)
    obs = np.array([(np.abs(res) == j).sum() for j in range(K + 1)], float)
    q0 = pmf(c, sL, sR)
    for _ in range(6):
        q = q0 * weight_vec(N, w); q /= q.sum(1, keepdims=True)
        e = np.array([q[:, np.abs(N) == j].sum() for j in range(K + 1)])
        w = w * np.where(e > 0, obs / np.maximum(e, 1e-12), 1.0)
    return w


def anchored(fm, sL, sR, w, deltas=np.arange(-2.0, 2.001, 0.1)):
    P = []
    for dl in deltas:
        q = pmf(fm + dl, sL, sR, w)
        win = (q * (N[None, :] > fm[:, None])).sum(1); lose = (q * (N[None, :] < fm[:, None])).sum(1)
        P.append(win / (win + lose))
    P = np.stack(P, 1)
    return np.array([f + np.interp(0.5, P[i], deltas) for i, f in enumerate(fm)])


def surv_at(q, idx, t):
    sv = 1 - np.cumsum(q, 1)
    return sv[idx, np.clip(np.floor(t).astype(int) - N[0], 0, len(N) - 1)]


def spread_fold(tr, va):
    Xtr, Xva = X_spread(tr), X_spread(va); p = Xtr.shape[1]
    fm_tr, fmar_tr = fav(tr); fm_va, fmar_va = fav(va)
    idx, t, k, y = grid(fm_va, fmar_va, kmax=30.5)
    P = {}
    s0 = float(np.sqrt(np.mean(tr.r ** 2)))
    P["N_const"] = 1 - norm.cdf((t - fm_va[idx]) / s0)
    th = fit_logsigma(tr.r.to_numpy(), Xtr)
    sg_tr, sg_va = np.exp(Xtr @ th), np.exp(Xva @ th)
    P["N_het"] = 1 - norm.cdf((t - fm_va[idx]) / sg_va[idx])
    w = keyw(fm_tr, sg_tr, sg_tr, fmar_tr)
    P["Nkey_het"] = surv_at(pmf(fm_va, sg_va, sg_va, w), idx, t)
    ths = fit_split(fmar_tr - fm_tr, Xtr)
    sL_tr, sR_tr = np.exp(Xtr @ ths[:p]), np.exp(Xtr @ ths[p:])
    w = keyw(fm_tr, sL_tr, sR_tr, fmar_tr)
    sL_va, sR_va = np.exp(Xva @ ths[:p]), np.exp(Xva @ ths[p:])
    P["SplitKey_c=line"] = surv_at(pmf(fm_va, sL_va, sR_va, w), idx, t)
    # production convention: center makes the line 50/50; shape refit around those centers
    c_tr = fm_tr.copy()
    for _ in range(3):
        ths = fit_split(fmar_tr - c_tr, Xtr)
        sL_tr, sR_tr = np.exp(Xtr @ ths[:p]), np.exp(Xtr @ ths[p:])
        w = keyw(c_tr, sL_tr, sR_tr, fmar_tr)
        c_tr = anchored(fm_tr, sL_tr, sR_tr, w)
    ths = fit_split(fmar_tr - c_tr, Xtr)
    sL_tr, sR_tr = np.exp(Xtr @ ths[:p]), np.exp(Xtr @ ths[p:])
    w = keyw(c_tr, sL_tr, sR_tr, fmar_tr)
    sL_va, sR_va = np.exp(Xva @ ths[:p]), np.exp(Xva @ ths[p:])
    c_va = anchored(fm_va, sL_va, sR_va, w)
    P["SplitKey_anchored"] = surv_at(pmf(c_va, sL_va, sR_va, w), idx, t)
    return idx, t, k, y, P


def total_fold(tr, va):
    tr, va = tr[tr.total_close.notna()], va[va.total_close.notna()]
    Xtr, Xva = X_total(tr), X_total(va)
    T = va.total_close.to_numpy()
    idx, t, k, y = grid(T, va.total.to_numpy(), kmax=40.5)
    P = {}
    s0 = float(np.sqrt(np.mean(tr.rt ** 2)))
    P["N_const"] = 1 - norm.cdf(k / s0)
    th = fit_logsigma(tr.rt.to_numpy(), Xtr)
    sg_tr, sg_va = np.exp(Xtr @ th), np.exp(Xva @ th)
    P["N_het"] = 1 - norm.cdf(k / sg_va[idx])
    i0, t0, k0, y0 = grid(tr.total_close.to_numpy(), tr.total.to_numpy(), kmax=30.5)
    s_0 = sg_tr[i0]
    ll = lambda v: scores(np.where(k0 >= 0, 1 - norm.cdf(k0 / (s_0 * v[1])), 1 - norm.cdf(k0 / (s_0 * v[0]))), y0)["logloss"]
    kL, kR = minimize(ll, [0.95, 1.0], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-7)).x
    s = sg_va[idx]
    P["Split_het"] = np.where(k >= 0, 1 - norm.cdf(k / (s * kR)), 1 - norm.cdf(k / (s * kL)))
    return idx, t, k, y, P, (kL, kR)


def summarize(rows, subsets):
    out = {}
    for name, f in subsets.items():
        acc = {}
        for (idx, t, k, y, P) in rows:
            sel = f(t, k)
            for m, p in P.items():
                s = scores(p[sel], y[sel]); n = sel.sum()
                a = acc.setdefault(m, [0.0, 0.0, 0])
                a[0] += s["brier"] * n; a[1] += s["logloss"] * n; a[2] += n
        out[name] = pd.DataFrame({m: dict(brier=a[0] / a[2], logloss=a[1] / a[2]) for m, a in acc.items()}).T
    return out


def calib(rows, ks):
    dfs = []
    for (idx, t, k, y, P) in rows:
        kb = np.where(k >= 0, np.floor(k) + 0.5, np.ceil(k) - 0.5)
        dfs.append(pd.DataFrame(dict(kb=kb, y=y, **P)))
    df = pd.concat(dfs); g = df.groupby("kb")
    tab = g.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
    for c in df.columns[2:]:
        tab[c] = g[c].mean()
    return tab.loc[tab.index.isin(ks)]


if __name__ == "__main__":
    d = data_pre2021()
    print("seasons loaded:", sorted(d.season.unique()), "games:", len(d))
    srows, trows, kk = [], [], []
    for v in SEASONS:
        tr, va = d[(d.season != v)].copy(), d[d.season == v].copy()
        srows.append(spread_fold(tr, va))
        i, t, k, y, P, kLR = total_fold(tr, va)
        trows.append((i, t, k, y, P)); kk.append((v, *np.round(kLR, 4)))
        print("fold", v, "done; total split multipliers", kk[-1], flush=True)
    print("\n=== SPREAD, leave-one-season-out 2018-2020 (favorite perspective) ===")
    subs = {"|k|<=15.5": lambda t, k: np.abs(k) <= 15.5, "|k|<=30.5": lambda t, k: np.ones(len(k), bool),
            "|k|>=10": lambda t, k: np.abs(k) >= 10, "|t|<=2.5 (near zero margin)": lambda t, k: np.abs(t) <= 2.5}
    for name, tab in summarize(srows, subs).items():
        print(name); print(tab.round(5).to_string())
    ks = [-25.5, -20.5, -15.5, -10.5, -5.5, -2.5, -0.5, 0.5, 1.5, 2.5, 5.5, 10.5, 15.5, 20.5, 25.5]
    cs = calib(srows, ks); print(cs.round(4).to_string()); cs.to_csv("cv_pre2021_spread_calib.csv")
    # near-zero absolute thresholds: favorite margin > t for t = 0.5..4.5
    print("\n=== TOTAL, leave-one-season-out 2018-2020 ===")
    subs = {"|k|<=15.5": lambda t, k: np.abs(k) <= 15.5, "|k|<=40.5": lambda t, k: np.ones(len(k), bool),
            "|k|>=15": lambda t, k: np.abs(k) >= 15}
    for name, tab in summarize(trows, subs).items():
        print(name); print(tab.round(5).to_string())
    ks = [-30.5, -25.5, -20.5, -15.5, -10.5, -5.5, -0.5, 0.5, 5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 35.5]
    ct = calib(trows, ks); print(ct.round(4).to_string()); ct.to_csv("cv_pre2021_total_calib.csv")
