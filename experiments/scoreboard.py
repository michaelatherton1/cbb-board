"""Locked scorecard for model experiments. Agents may NOT edit this file or cbb/backtest.py.

Runs the existing walk-forward backtest (cbb/backtest.run: tuned on 2012-2020, scored on the untouched
2021-2026 holdout) and writes one JSON per variant:

    python experiments/scoreboard.py baseline                # production settings
    python experiments/scoreboard.py <label> --roster        # same, with the roster-continuity prior
    python experiments/scoreboard.py <label> --params FILE   # alternative Params json
    python experiments/scoreboard.py <label> --post MOD:FUNC # post-process predictions (e.g. a market blend)
                                                            # FUNC(pred_df) -> pred_df, applied after the
                                                            # backtest (lines attached) and before scoring

Headline metric: holdout margin MAE (lower is better), also split by season and by days into the season,
plus the same games' DraftKings/ESPN closing-line MAE when lines_<season>.parquet exist (fetch_lines.py).
"""
import argparse, json, sys, time
from dataclasses import fields
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cbb"))
import backtest  # noqa: E402
from model import Params  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
DAY_BUCKETS = [(0, 14, "days 0-14"), (15, 30, "days 15-30"), (31, 60, "days 31-60"), (61, 999, "day 61+")]


def summarize(te):
    te = te.copy()
    te["day"] = (te.date - te.groupby("season").date.transform("min")).dt.days
    em = (te.margin - te.pred_margin).abs()
    et = (te.total - te.pred_total).abs()
    out = {"n": int(len(te)), "margin_mae": float(em.mean()), "total_mae": float(et.mean()),
           "by_season": {}, "by_day": {}, "vs_market": None}
    for s, g in te.groupby("season"):
        out["by_season"][int(s)] = {"n": int(len(g)), "margin_mae": float((g.margin - g.pred_margin).abs().mean()),
                                    "total_mae": float((g.total - g.pred_total).abs().mean())}
    for lo, hi, name in DAY_BUCKETS:
        g = te[te.day.between(lo, hi)]
        out["by_day"][name] = {"n": int(len(g)), "margin_mae": float((g.margin - g.pred_margin).abs().mean())}
    if "spread_close" in te:
        d = te[te.spread_close.notna()]
        if len(d):
            mk = (d.margin + d.spread_close).abs()
            md = (d.margin - d.pred_margin).abs()
            out["vs_market"] = {"n": int(len(d)), "model_margin_mae": float(md.mean()), "market_margin_mae": float(mk.mean()),
                                "by_day": {}}
            d = d.assign(day=d.day)
            for lo, hi, name in DAY_BUCKETS:
                b = d[d.day.between(lo, hi)]
                out["vs_market"]["by_day"][name] = {
                    "n": int(len(b)), "model": float((b.margin - b.pred_margin).abs().mean()),
                    "market": float((b.margin + b.spread_close).abs().mean())}
            ats = backtest.ats(d)
            out["vs_market"]["ats"] = ats.round(4).to_dict(orient="records")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--roster", action="store_true", help="use the roster-continuity preseason prior")
    ap.add_argument("--params", help="json file of Params overrides (default: data/best_params.json)")
    ap.add_argument("--post", help="module:function that transforms the prediction frame before scoring")
    a = ap.parse_args()
    t0 = time.time()
    p = backtest.load_params()
    if a.params:
        over = json.loads(Path(a.params).read_text())
        names = {f.name for f in fields(Params)}
        p = Params(**{**{f.name: getattr(p, f.name) for f in fields(Params)}, **{k: v for k, v in over.items() if k in names}})
    roster = None
    if a.roster:
        import roster as R
        roster = R.RosterPrior()
    pred, cal = backtest.run(p, roster=roster, label=a.label)
    if a.post:
        import importlib
        mod, fn = a.post.split(":")
        pred = getattr(importlib.import_module(mod), fn)(pred)
    te = pred[pred.season.isin(backtest.TEST)]
    res = {"label": a.label, "roster": a.roster, "post": a.post, "params": {f.name: getattr(p, f.name) for f in fields(Params)},
           "seconds": round(time.time() - t0), "holdout": summarize(te)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{a.label}.json").write_text(json.dumps(res, indent=1))
    h = res["holdout"]
    print(f"[{a.label}] holdout n={h['n']} margin MAE={h['margin_mae']:.3f} total MAE={h['total_mae']:.3f} ({res['seconds']}s)")
    for k, v in h["by_day"].items():
        print(f"   {k:11s} n={v['n']:5d} margin MAE={v['margin_mae']:.3f}")
    if h["vs_market"]:
        vm = h["vs_market"]
        print(f"   vs closing line (n={vm['n']}): model {vm['model_margin_mae']:.3f} | market {vm['market_margin_mae']:.3f}")


if __name__ == "__main__":
    main()
