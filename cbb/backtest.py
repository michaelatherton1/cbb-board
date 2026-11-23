"""Out-of-sample backtest: calibrate on TRAIN seasons, evaluate on TEST seasons, compare to market lines."""
import sys, json
from dataclasses import fields
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).parent))
from model import Params, walk_forward

DATA = Path(__file__).resolve().parent.parent / "data"
TRAIN = list(range(2012, 2021))   # calibration (params tuned on 2012-2017, variants chosen on 2018-2020)
TEST = list(range(2021, 2027))    # untouched holdout


def load_params(path=DATA / "best_params.json"):
    d = json.loads(Path(path).read_text())
    return Params(**{k: v for k, v in d.items() if k in {f.name for f in fields(Params)}})


EARLY_TAU = 21.0   # days; decay of the early-season non-conference home bump


def early_nc_home(df):
    """1.0 for a true home game outside conference play on opening night, decaying with days into season."""
    day = df["day"].to_numpy() if "day" in df else np.zeros(len(df))
    flag = (~df.neutral.astype(bool).to_numpy()) & (~df.conf_game.astype(bool).to_numpy())
    return flag * np.exp(-day / EARLY_TAU)


def margin_design(df):
    return np.c_[df.raw_margin.to_numpy(), np.ones(len(df)), early_nc_home(df)]


def fit_calibration(tr):
    tr = tr.copy()
    if "day" not in tr:
        tr["day"] = (tr.date - tr.groupby("season").date.transform("min")).dt.days
    bmx = np.linalg.lstsq(margin_design(tr), tr.margin.to_numpy(), rcond=None)[0]
    bm = bmx[:2]
    bt = np.polyfit(tr.raw_total, tr.total, 1)
    pm = margin_design(tr) @ bmx
    resid = tr.margin - pm
    # win prob sigma by max likelihood on a grid
    win = (tr.margin > 0).astype(float)
    grid = np.arange(8, 14, 0.05)
    ll = [-(win * np.log(norm.cdf(pm / s)) + (1 - win) * np.log(norm.cdf(-pm / s))).mean() for s in grid]
    return dict(bm=bm.tolist(), b_early_nc_home=float(bmx[2]), bt=bt.tolist(), sigma=float(grid[int(np.argmin(ll))]),
                resid_sd=float(resid.std()), total_sd=float((tr.total - np.polyval(bt, tr.raw_total)).std()))


def apply_calibration(pred, cal):
    pred = pred.copy()
    if "day" not in pred:
        pred["day"] = (pred.date - pred.groupby("season").date.transform("min")).dt.days
    pred["pred_margin"] = margin_design(pred) @ np.r_[cal["bm"], cal.get("b_early_nc_home", 0.0)]
    pred["pred_total"] = np.polyval(cal["bt"], pred.raw_total)
    pred["p_home"] = norm.cdf(pred.pred_margin / cal["sigma"])
    return pred


def attach_lines(pred):
    ls = []
    for y in sorted(pred.season.unique()):
        p = DATA / f"lines_{y}.parquet"
        if p.exists():
            ls.append(pd.read_parquet(p))
    if not ls:
        return pred
    lines = pd.concat(ls)
    lines = lines[lines.home_team_id.astype(float).notna()]
    keep = ["game_id", "provider", "spread_close", "total_close", "ml_home_close", "ml_away_close",
            "spread_open", "total_open"]
    out = pred.merge(lines[keep], on="game_id", how="left")
    # ESPN's older consensus feeds attach the spread to the wrong side for ~10-20% of neutral-site games.
    # The moneylines carry team ids and are right there, so on neutral floors take the favorite from the ML.
    has_ml = out.ml_home_close.notna() & out.ml_away_close.notna() & (out.ml_home_close != out.ml_away_close)
    sign = np.where(out.ml_home_close < out.ml_away_close, -1.0, 1.0)
    fix = out.neutral & has_ml & out.spread_close.notna()
    out["spread_raw"] = out.spread_close
    out.loc[fix, "spread_close"] = sign[fix] * out.loc[fix, "spread_close"].abs()
    if "spread_open" in out:
        fo = fix & out.spread_open.notna()
        out.loc[fo, "spread_open"] = sign[fo] * out.loc[fo, "spread_open"].abs()
    return out


def metrics(df):
    em, et = df.margin - df.pred_margin, df.total - df.pred_total
    win = (df.margin > 0).astype(float)
    p = df.p_home.clip(1e-4, 1 - 1e-4)
    return pd.Series(dict(
        n=len(df), margin_mae=em.abs().mean(), margin_rmse=np.sqrt((em ** 2).mean()),
        total_mae=et.abs().mean(), total_rmse=np.sqrt((et ** 2).mean()),
        winner_acc=((df.pred_margin > 0) == (df.margin > 0)).mean(),
        log_loss=-(win * np.log(p) + (1 - win) * np.log(1 - p)).mean(),
        brier=((p - win) ** 2).mean()))


def market_compare(df):
    d = df[df.spread_close.notna() & df.total_close.notna()]
    mkt_m = -d.spread_close
    return pd.Series(dict(
        n=len(d),
        model_margin_mae=(d.margin - d.pred_margin).abs().mean(),
        market_margin_mae=(d.margin - mkt_m).abs().mean(),
        model_total_mae=(d.total - d.pred_total).abs().mean(),
        market_total_mae=(d.total - d.total_close).abs().mean(),
        corr_model_market_margin=np.corrcoef(d.pred_margin, mkt_m)[0, 1],
    ))


def ats(df, line_col="spread_close", edges=(0, 1, 2, 3, 4, 5, 6)):
    """Bet the side the model prefers when |model margin - market margin| >= edge. -110 assumed."""
    d = df[df[line_col].notna()].copy()
    d["edge"] = d.pred_margin + d[line_col]          # >0 -> model likes home vs the number
    d["res"] = d.margin + d[line_col]                # >0 -> home covered
    rows = []
    for e in edges:
        b = d[d.edge.abs() >= e]
        b = b[b.res != 0]
        won = (np.sign(b.edge) == np.sign(b.res))
        rows.append(dict(edge=e, bets=len(b), win_pct=won.mean(), roi=(won * (100 / 110) - (~won)).mean()))
    return pd.DataFrame(rows)


def ou(df, line_col="total_close", edges=(0, 2, 4, 6, 8, 10)):
    d = df[df[line_col].notna()].copy()
    d["edge"] = d.pred_total - d[line_col]
    d["res"] = d.total - d[line_col]
    rows = []
    for e in edges:
        b = d[(d.edge.abs() >= e) & (d.res != 0)]
        won = (np.sign(b.edge) == np.sign(b.res))
        rows.append(dict(edge=e, bets=len(b), win_pct=won.mean(), roi=(won * (100 / 110) - (~won)).mean(),
                         over_share=(b.edge > 0).mean()))
    return pd.DataFrame(rows)


def run(p=None, roster=None, label="base"):
    games = pd.read_parquet(DATA / "games.parquet")
    p = p or load_params()
    kw = {} if roster is None else dict(roster=roster)
    pred, _ = walk_forward(games, TRAIN + TEST, p, **kw)
    cal = fit_calibration(pred[pred.season.isin(TRAIN)])
    pred = attach_lines(apply_calibration(pred, cal))
    pred["month"] = pred.date.dt.month
    pred.to_parquet(DATA / f"backtest_{label}.parquet")
    return pred, cal


if __name__ == "__main__":
    pred, cal = run()
    # production calibration uses every completed season (the backtest above only used pre-holdout seasons)
    prod = fit_calibration(pred)
    (DATA / "calibration.json").write_text(json.dumps(prod, indent=1))
    te = pred[pred.season.isin(TEST)]
    pd.set_option("display.width", 200)
    print("calibration", {k: (np.round(v, 3) if not isinstance(v, list) else np.round(v, 3).tolist()) for k, v in cal.items()})
    print("\nTEST by season\n", te.groupby("season").apply(metrics, include_groups=False).round(3))
    print("\nTEST by month\n", te.groupby("month").apply(metrics, include_groups=False).round(3))
    print("\nTEST overall\n", metrics(te).round(3))
    print("\nvs market (test seasons with lines)\n", te.groupby("season").apply(market_compare, include_groups=False).round(3))
    print("\nATS vs close\n", ats(te).round(3))
    print("\nO/U vs close\n", ou(te).round(3))
