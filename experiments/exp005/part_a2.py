"""Part A follow-up: far tails (|k| up to 30), favorite-perspective spread calibration, split-normal totals."""
import numpy as np, pandas as pd, json
from scipy.stats import norm
from part_a import data, X_spread, X_total, fit_logsigma, fit_key_weights, surv_discrete, grid, scores, scaled_emp_surv, TRAIN, TEST
pd.set_option("display.width", 250)
d = data()

# ---------------- spreads, favorite perspective, far tails
tr, te = d[d.season.isin(TRAIN)].copy(), d[d.season.isin(TEST)].copy()
th = fit_logsigma(tr.r.to_numpy(), X_spread(tr))
tr["sig"], te["sig"] = np.exp(X_spread(tr) @ th), np.exp(X_spread(te) @ th)
w = fit_key_weights(tr.m.to_numpy(), tr.sig.to_numpy(), tr.margin.to_numpy())
s0 = float(np.sqrt(np.mean(tr.r ** 2)))
# favorite perspective: fm = |m| (>=0), fmargin = sign(m)*margin ; pick-ems (m==0) -> home
sgn = np.where(te.m >= 0, 1.0, -1.0)
fm, fmar = np.abs(te.m.to_numpy()), sgn * te.margin.to_numpy()
idx, t, k, y = grid(fm, fmar, kmax=30.5)
sig = te.sig.to_numpy()
P = {"N_const": 1 - norm.cdf((t - fm[idx]) / s0),
     "N_het": 1 - norm.cdf((t - fm[idx]) / sig[idx]),
     "Nkey_het": surv_discrete(fm, sig, t, w, idx)}
z = (tr.r / tr.sig).to_numpy()
P["Zemp_het"] = scaled_emp_surv(z, k / sig[idx])
kb = np.select([k <= -20, k <= -15, k <= -10, k <= -5, k < 0, k < 5, k < 10, k < 15, k < 20], ["a<=-20", "b(-20,-15]", "c(-15,-10]", "d(-10,-5]", "e(-5,0)", "f[0,5)", "g[5,10)", "h[10,15)", "i[15,20)"], "j>=20")
df = pd.DataFrame(dict(kb=kb, y=y, **P))
g = df.groupby("kb")
tab = g.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
for c in P: tab[c] = g[c].mean() - tab.obs
print("SPREAD favorite perspective: event fav_margin > |m| + k; pred-obs by k bucket (2021-23)")
print(tab.round(4).to_string())
for name, sel in [("all |k|<=30.5", np.ones(len(k), bool)), ("|k|>=15", np.abs(k) >= 15), ("|k|>=20", np.abs(k) >= 20)]:
    print(name, {m: {kk: round(v, 5) for kk, v in scores(p[sel], y[sel]).items()} for m, p in P.items()})
# by |m| bucket for the far favorite side (k>=10) and dog side (k<=-10)
df["mb"] = pd.cut(fm[idx], [-0.1, 5.5, 12.5, 20.5, 99]).astype(str)
df["side"] = np.where(k >= 10, "fav covers big (k>=10)", np.where(k <= -10, "fav fails big (k<=-10)", "mid"))
g2 = df[df.side != "mid"].groupby(["side", "mb"])
t2 = g2.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
for c in P: t2[c] = g2[c].mean() - t2.obs
print(t2.round(4).to_string())

# ---------------- totals: split normal (median at T), far tails
dt = d[d.total_close.notna()]
tr, te = dt[dt.season.isin(TRAIN)].copy(), dt[dt.season.isin(TEST)].copy()
th_t = fit_logsigma(tr.rt.to_numpy(), X_total(tr))
tr["sig"], te["sig"] = np.exp(X_total(tr) @ th_t), np.exp(X_total(te) @ th_t)
ztr = (tr.rt / tr.sig).to_numpy()
# split-normal: half-normal scale on each side of the median (median ~ the line)
med = 0.0
kL = np.sqrt(np.mean(ztr[ztr < med] ** 2)); kR = np.sqrt(np.mean(ztr[ztr > med] ** 2))
print("\nTOTAL split-normal multipliers (on het sigma): left", round(kL, 4), "right", round(kR, 4),
      "share below line (train)", round((tr.rt < 0).mean(), 4), "at", round((tr.rt == 0).mean(), 4))
# also constant-sigma split
sL = np.sqrt(np.mean(tr.rt[tr.rt < 0] ** 2)); sR = np.sqrt(np.mean(tr.rt[tr.rt > 0] ** 2))
print("constant split sigmas: left", round(sL, 3), "right", round(sR, 3))
T = te.total_close.to_numpy(); sg = te.sig.to_numpy()
idx, t, k, y = grid(T, te.total.to_numpy(), kmax=40.5)
def split(k, s, kL, kR):
    return np.where(k >= 0, 1 - norm.cdf(k / (s * kR)), 1 - norm.cdf(k / (s * kL)))
P = {"N_het": 1 - norm.cdf(k / sg[idx]),
     "Split_het": split(k, sg[idx], kL, kR),
     "Split_const": split(k, np.ones(len(k)), sL, sR),
     "Zemp_het": scaled_emp_surv(ztr, k / sg[idx])}
kb = np.select([k <= -25, k <= -15, k <= -10, k <= -5, k < 0, k < 5, k < 10, k < 15, k < 25], ["a<=-25", "b(-25,-15]", "c(-15,-10]", "d(-10,-5]", "e(-5,0)", "f[0,5)", "g[5,10)", "h[10,15)", "i[15,25)"], "j>=25")
df = pd.DataFrame(dict(kb=kb, y=y, **P)); g = df.groupby("kb")
tab = g.y.agg(["count", "mean"]).rename(columns={"mean": "obs"})
for c in P: tab[c] = g[c].mean() - tab.obs
print("TOTAL: event total > T + k; pred-obs by k bucket (2021-23)")
print(tab.round(4).to_string())
for name, sel in [("all |k|<=15.5", np.abs(k) <= 15.5), ("all |k|<=40.5", np.ones(len(k), bool)), ("|k|>=15", np.abs(k) >= 15)]:
    print(name, {m: {kk: round(v, 5) for kk, v in scores(p[sel], y[sel]).items()} for m, p in P.items()})
json.dump(dict(spread_theta=th.tolist(), key_w=w.tolist(), total_theta=th_t.tolist(), split_kL=kL, split_kR=kR),
          open("/home/claude/scratch/exp005/fitted_params.json", "w"), indent=1)
