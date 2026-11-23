import pandas as pd, numpy as np
r = pd.read_csv('/home/claude/wt/main-merge/data/out/results_games.csv')
r['date'] = pd.to_datetime(r.date_s)
d = r[r.total_open.notna() & r.total_close.notna() & r.d1].copy()
d['move'] = d.total_close - d.total_open
d['m_open'] = d.pred_total - d.total_open
d['res_close'] = d.total - d.total_close
d['res_open'] = d.total - d.total_open
print("games", len(d), "dates", d.date.min().date(), "-", d.date.max().date())
print("move: mean %.2f sd %.2f, share |move|>=1: %.2f, >=3: %.2f" % (d.move.mean(), d.move.std(), (d.move.abs()>=1).mean(), (d.move.abs()>=3).mean()))
print("MAE open %.3f close %.3f model %.3f" % (d.res_open.abs().mean(), d.res_close.abs().mean(), (d.total-d.pred_total).abs().mean()))
print("bias (actual - x): open %+.2f close %+.2f model %+.2f" % (d.res_open.mean(), d.res_close.mean(), (d.total-d.pred_total).mean()))
def ols(y, X):
    X = np.column_stack([np.ones(len(X))] + [X[:, i] for i in range(X.shape[1])])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / (len(y) - X.shape[1]))
    return b, se
def by_date_boot(f, n=2000):
    g = list(d.groupby('date')); rng = np.random.default_rng(1); out = []
    for _ in range(n):
        idx = rng.integers(0, len(g), len(g)); out.append(f(pd.concat([g[i][1] for i in idx])))
    return np.percentile(out, [10, 90])
# 1) does model - open predict the move?
b, se = ols(d.move.to_numpy(), d[['m_open']].to_numpy())
print("\n1) move ~ (model - open): slope %.3f (se %.3f), corr %.3f" % (b[1], se[1], d.move.corr(d.m_open)))
print("   80%% range by date:", by_date_boot(lambda x: np.polyfit(x.m_open, x.move, 1)[0]).round(3))
for lo, hi in [(3, 99), (5, 99)]:
    s = d[d.m_open.abs() >= lo]
    agree = (np.sign(s.move) == np.sign(s.m_open))[s.move != 0]
    print(f"   |model-open|>={lo}: n={len(s)}, line moved toward model {agree.mean():.3f} (n moved={len(agree)})")
# 2) does the move itself carry on (close not efficient)?
b, se = ols(d.res_close.to_numpy(), d[['move']].to_numpy())
print("\n2) (actual - close) ~ move: slope %.3f (se %.3f)" % (b[1], se[1]))
# 3) model info beyond the close / beyond the open
b, se = ols(d.res_close.to_numpy(), (d.pred_total - d.total_close).to_numpy()[:, None])
print("3) (actual - close) ~ (model - close): slope %.3f (se %.3f)" % (b[1], se[1]))
b, se = ols(d.res_open.to_numpy(), d[['m_open']].to_numpy())
print("   (actual - open) ~ (model - open): slope %.3f (se %.3f)" % (b[1], se[1]))
# 4) best blend weight w*model + (1-w)*open for the actual total
ws = np.arange(0, 0.51, 0.05)
mae = [((d.total - (w * d.pred_total + (1 - w) * d.total_open)).abs().mean()) for w in ws]
print("\n4) blend MAE by model weight:", dict(zip(ws.round(2), np.round(mae, 3))))
d['day'] = (d.date - d.date.min()).dt.days
for lo, hi in [(0, 10), (11, 30)]:
    s = d[d.day.between(lo, hi)]
    print(f"   days {lo}-{hi} n={len(s)}: slope move~(model-open) {np.polyfit(s.m_open, s.move, 1)[0]:.3f}; best w "
          f"{ws[np.argmin([((s.total-(w*s.pred_total+(1-w)*s.total_open)).abs().mean()) for w in ws])]:.2f}")
