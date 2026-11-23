"""EXP-005 Part A: distribution of the actual result around the closing spread / total.

Fit on 2018-2020 closes, evaluate on 2021-2023 (2026/2027 never touched).
Events are Kalshi-style rungs: half-integer thresholds t with |t - center| <= 15.5, y = 1{result > t}.

    python part_a.py            # prints sigma tables, calibration tables, Brier/log loss per model
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from load_hist import load

TRAIN, TEST = [2018, 2019, 2020], [2021, 2022, 2023]
KMAX = 15.5
OUT = "/home/claude/scratch/exp005"


# ------------------------------------------------------------------------------------------ data
def data():
    d = load()
    d = d[~(d.sign_disagree & ~d.neutral)]                 # non-neutral spread/ML sign conflict: drop (0.3%)
    d = d[~(d.neutral & d.ml_home_close.isna())]            # neutral without ML: sign can't be checked
    d = d[d.total_close.between(100, 200) | d.total_close.isna()]
    d["early"] = np.exp(-d.day / 21.0)
    d["r"] = d.margin - d.m
    d["rt"] = d.total - d.total_close
    return d.reset_index(drop=True)


# -------------------------------------------------------------------------------- sigma models
def X_spread(d):
    return np.c_[np.ones(len(d)), (d.total_close.fillna(141) - 140) / 10, d.m.abs() / 10,
                 (d.m.abs() / 10) ** 2, d.early, d.neutral.astype(float)]


def X_total(d):
    return np.c_[np.ones(len(d)), (d.total_close - 140) / 10, d.m.abs() / 10, d.early, d.neutral.astype(float)]


def fit_logsigma(r, X):
    """Normal MLE with mean 0 and log sigma = X @ theta."""
    def nll(th):
        ls = X @ th
        return np.mean(ls + 0.5 * r ** 2 * np.exp(-2 * ls))
    th0 = np.zeros(X.shape[1]); th0[0] = np.log(r.std())
    res = minimize(nll, th0, method="BFGS")
    return res.x


# --------------------------------------------------------------------------------- event grid
def grid(center, result, kmax=KMAX):
    """All half-integer thresholds t with |t - center| <= kmax. Returns (row idx, t, k = t - center, y)."""
    idx, ts = [], []
    for i, c in enumerate(center):
        lo, hi = np.ceil(c - kmax - 0.5) + 0.5, np.floor(c + kmax - 0.5) + 0.5
        t = np.arange(lo, hi + 0.01, 1.0)
        t = t[np.abs(t - c) <= kmax + 1e-9]
        idx.append(np.full(len(t), i)); ts.append(t)
    idx, ts = np.concatenate(idx), np.concatenate(ts)
    return idx, ts, ts - center[idx], (result[idx] > ts).astype(float)


# ------------------------------------------------------------------------------------ discrete
NMAX = 120


def normal_pmf(center, sigma, lo=-NMAX, hi=NMAX):
    n = np.arange(lo, hi + 1)
    c, s = np.asarray(center)[:, None], np.asarray(sigma)[:, None]
    return n, norm.cdf((n + 0.5 - c) / s) - norm.cdf((n - 0.5 - c) / s)


def fit_key_weights(center, sigma, result, K=12, iters=5):
    """Multiplicative weight per |margin| (0..K) so the reweighted Normal pmf reproduces the observed margin
    frequencies (the 'key number' adjustment). Weight 1 beyond K."""
    n, p = normal_pmf(center, sigma)
    w = np.ones(K + 1)
    res = np.asarray(result).astype(int)
    obs = np.array([(np.abs(res) == j).sum() for j in range(K + 1)], float)
    for _ in range(iters):
        q = p * weight_vec(n, w)
        q /= q.sum(1, keepdims=True)
        exp = np.array([q[:, np.abs(n) == j].sum() for j in range(K + 1)])
        w = w * np.where(exp > 0, obs / np.maximum(exp, 1e-12), 1.0)
    return w


def weight_vec(n, w):
    K = len(w) - 1
    a = np.abs(n)
    return np.where(a <= K, w[np.minimum(a, K)], 1.0)


def surv_discrete(center, sigma, t, w=None, idx=None):
    """P(result > t). center/sigma are per game; idx maps each threshold row to its game (identity if None).
    Normal pmf on integers, optional key weights, renormalized per game."""
    center, sigma, t = np.asarray(center, float), np.asarray(sigma, float), np.asarray(t, float)
    if idx is None:
        idx = np.arange(len(center))
    n, p = normal_pmf(center, sigma)
    if w is not None:
        p = p * weight_vec(n, w)
    p /= p.sum(1, keepdims=True)
    surv = 1 - np.cumsum(p, 1)                       # surv[:, j] = P(result > n[j])
    j = np.clip(np.floor(t).astype(int) - n[0], 0, len(n) - 1)
    return surv[idx, j]


# -------------------------------------------------------------------------------- empirical
def emp_surv(train_r, train_frac, k, frac):
    """Empirical P(r > k) using only training games on the same half-point lattice (frac of the center),
    so integer results never tie with the threshold."""
    out = np.empty(len(k))
    for f in (0.0, 0.5):
        tr = np.sort(train_r[np.isclose(train_frac, f)])
        sel = np.isclose(frac, f)
        out[sel] = 1 - np.searchsorted(tr, k[sel], side="right") / len(tr)
    return out


def emp_surv_bucketed(tr, te, rcol, ccol, bcol, k, frac, idx):
    out = np.empty(len(k))
    for b in te[bcol].unique():
        trb = tr[tr[bcol] == b]
        sel = (te[bcol].to_numpy()[idx] == b)
        out[sel] = emp_surv(trb[rcol].to_numpy(), (trb[ccol] % 1).to_numpy(), k[sel], frac[sel])
    return out


def scaled_emp_surv(z_train, k_over_sigma, bw=0.04):
    """Kernel-smoothed survival of standardized residuals (shape from data, scale from the sigma model)."""
    zs = np.sort(z_train)
    # Gaussian-kernel smoothed survival: mean over training z of Phi((z - x)/bw)
    grid_x = np.linspace(-6, 6, 2401)
    sv = np.array([norm.cdf((zs - x) / bw).mean() for x in grid_x])
    return np.interp(k_over_sigma, grid_x, sv)


# ------------------------------------------------------------------------------------- scoring
def scores(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return dict(brier=float(np.mean((p - y) ** 2)), logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))


def calib_table(k, y, preds, step=1.0):
    kb = np.round(k / step) * step if step != 1.0 else np.where(k >= 0, np.floor(k) + 0.5, np.ceil(k) - 0.5)
    df = pd.DataFrame(dict(kb=kb, y=y, **preds))
    g = df.groupby("kb")
    t = g.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
    for c in preds:
        t[c] = g[c].mean() - t.obs
    return t


def run_spread(d):
    tr, te = d[d.season.isin(TRAIN)].copy(), d[d.season.isin(TEST)].copy()
    out = {}
    # constant sigma
    s0 = float(np.sqrt(np.mean(tr.r ** 2)))
    th = fit_logsigma(tr.r.to_numpy(), X_spread(tr))
    out["sigma_const"] = s0
    out["logsigma_theta"] = dict(zip(["const", "T10", "absm10", "absm10_sq", "early", "neutral"], th.round(4).tolist()))
    tr["sig"], te["sig"] = np.exp(X_spread(tr) @ th), np.exp(X_spread(te) @ th)
    # key weights on train
    w_const = fit_key_weights(tr.m.to_numpy(), np.full(len(tr), s0), tr.margin.to_numpy())
    w_het = fit_key_weights(tr.m.to_numpy(), tr.sig.to_numpy(), tr.margin.to_numpy())
    out["key_weights_const"] = w_const.round(3).tolist()
    out["key_weights_het"] = w_het.round(3).tolist()

    idx, t, k, y = grid(te.m.to_numpy(), te.margin.to_numpy())
    c, sig = te.m.to_numpy()[idx], te.sig.to_numpy()[idx]
    frac = (c % 1)
    mg, sg = te.m.to_numpy(), te.sig.to_numpy()
    preds = {}
    preds["N_const"] = 1 - norm.cdf((t - c) / s0)
    preds["N_het"] = 1 - norm.cdf((t - c) / sig)
    preds["Nkey_const"] = surv_discrete(mg, np.full(len(mg), s0), t, w_const, idx)
    preds["Nkey_het"] = surv_discrete(mg, sg, t, w_het, idx)
    preds["Emp_pooled"] = emp_surv(tr.r.to_numpy(), (tr.m % 1).to_numpy(), k, frac)
    for D in (tr, te):
        D["mb"] = pd.cut(D.m.abs(), [-0.1, 2.5, 5.5, 9.5, 14.5, 19.5, 99]).astype(str)
    preds["Emp_by_absm"] = emp_surv_bucketed(tr, te, "r", "m", "mb", k, frac, idx)
    z = (tr.r / tr.sig).to_numpy()
    preds["Zemp_het"] = scaled_emp_surv(z, k / sig)
    out["scores_all"] = {m: scores(p, y) for m, p in preds.items()}
    near = np.abs(k) <= 5.5
    far = np.abs(k) >= 10
    out["scores_near(|k|<=5.5)"] = {m: scores(p[near], y[near]) for m, p in preds.items()}
    out["scores_far(|k|>=10)"] = {m: scores(p[far], y[far]) for m, p in preds.items()}
    # rungs near zero margin (|t|<=2.5): where the no-tie hole matters
    z0 = np.abs(t) <= 2.5
    out["scores_t_near_0(|t|<=2.5)"] = {m: scores(p[z0], y[z0]) for m, p in preds.items()}
    cal = calib_table(k, y, preds)
    calz = calib_table(t, y, {m: p for m, p in preds.items()}).loc[-4.5:4.5]
    calz = calz[[c for c in calz.columns]]
    return out, cal, calz, tr, te


def run_total(d):
    d = d[d.total_close.notna()].copy()
    tr, te = d[d.season.isin(TRAIN)].copy(), d[d.season.isin(TEST)].copy()
    out = {}
    s0 = float(np.sqrt(np.mean(tr.rt ** 2)))
    th = fit_logsigma(tr.rt.to_numpy(), X_total(tr))
    out["sigma_const"] = s0
    out["logsigma_theta"] = dict(zip(["const", "T10", "absm10", "early", "neutral"], th.round(4).tolist()))
    tr["sig"], te["sig"] = np.exp(X_total(tr) @ th), np.exp(X_total(te) @ th)
    out["mean_resid_train"] = float(tr.rt.mean()); out["median_resid_train"] = float(tr.rt.median())
    idx, t, k, y = grid(te.total_close.to_numpy(), te.total.to_numpy())
    c, sig = te.total_close.to_numpy()[idx], te.sig.to_numpy()[idx]
    frac = c % 1
    preds = {}
    preds["N_const"] = 1 - norm.cdf((t - c) / s0)
    preds["N_het"] = 1 - norm.cdf((t - c) / sig)
    # small mean shift (train mean residual): totals went slightly over the close historically
    mu = tr.rt.mean()
    preds["N_het_shift"] = 1 - norm.cdf((t - c - mu) / sig)
    preds["Emp_pooled"] = emp_surv(tr.rt.to_numpy(), (tr.total_close % 1).to_numpy(), k, frac)
    for D in (tr, te):
        D["tb"] = pd.cut(D.total_close, [0, 132, 140, 148, 300]).astype(str)
    preds["Emp_by_T"] = emp_surv_bucketed(tr, te, "rt", "total_close", "tb", k, frac, idx)
    z = (tr.rt / tr.sig).to_numpy()
    preds["Zemp_het"] = scaled_emp_surv(z, k / sig)
    out["scores_all"] = {m: scores(p, y) for m, p in preds.items()}
    near, far = np.abs(k) <= 5.5, np.abs(k) >= 10
    out["scores_near(|k|<=5.5)"] = {m: scores(p[near], y[near]) for m, p in preds.items()}
    out["scores_far(|k|>=10)"] = {m: scores(p[far], y[far]) for m, p in preds.items()}
    cal = calib_table(k, y, preds)
    # skewness check: the 'z' quantiles
    out["z_quantiles_train"] = {q: float(np.quantile(z, q)) for q in (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)}
    return out, cal, tr, te


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    d = data()
    print("games:", len(d), d.groupby("season").size().to_dict())
    so, scal, scalz, str_, ste = run_spread(d)
    print("\n=== SPREAD ===")
    print(json.dumps({k: v for k, v in so.items() if not k.startswith("scores")}, indent=1))
    for k in [k for k in so if k.startswith("scores")]:
        print(k); print(pd.DataFrame(so[k]).T.round(5))
    print("\nCalibration (pred - obs) by k = t - m (2021-2023), rows = rung offset bucket:")
    print(scal.round(4).to_string())
    print("\nCalibration by absolute threshold t (|t|<=4.5):")
    print(scalz.round(4).to_string())
    to, tcal, ttr, tte = run_total(d)
    print("\n=== TOTAL ===")
    print(json.dumps({k: v for k, v in to.items() if not k.startswith("scores")}, indent=1))
    for k in [k for k in to if k.startswith("scores")]:
        print(k); print(pd.DataFrame(to[k]).T.round(5))
    print("\nCalibration (pred - obs) by k = t - T (2021-2023):")
    print(tcal.round(4).to_string())
    json.dump(dict(spread=so, total=to), open(f"{OUT}/part_a_results.json", "w"), indent=1)
    scal.to_csv(f"{OUT}/calib_spread.csv"); tcal.to_csv(f"{OUT}/calib_total.csv"); scalz.to_csv(f"{OUT}/calib_spread_abs_t.csv")
