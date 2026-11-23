"""EXP-007: monthly re-check of EXP-002 - does the model add information to the opening line yet?

    python experiments/blend_check.py [--season 2027] [--results data/out/results_games.csv] [--no-fetch]

Uses the Record tab's pre-game model lines (data/out/results_games.csv: pred_margin / pred_total, rated the morning of
each game) and RAW opening lines from the ESPN mirror (cbb/fetch_lines.py -> data/lines_<season>.parquet; raw opens
are home-perspective; don't use attach_lines' opens, which take neutral-site signs from the closing moneyline).

For each days-into-season bucket it finds the weight w on the model that minimizes MAE vs the actual result for
    blend = w * model + (1 - w) * opening line
with a by-date bootstrap range. Pre-registered rule (set before any of these runs): EXP-002 is worth re-opening
(a one-shot holdout test on 2026, the only holdout season with opening lines) when a bucket with >= 300 games has
best w >= 0.15 AND its bootstrap 80% range excludes 0. Appends a dated section to experiments/blend_check.md.
"""
import argparse, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cbb"))
DATA = ROOT / "data"
BUCKETS = [(0, 14), (15, 30), (31, 60), (61, 90), (91, 999)]
W = np.round(np.arange(0, 1.0001, 0.05), 2)


def best_w(actual, model, line):
    maes = [np.abs(actual - (w * model + (1 - w) * line)).mean() for w in W]
    i = int(np.argmin(maes))
    return W[i], maes[i]


def boot_range(d, col_a, col_m, col_l, n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = d.day.unique()
    ws = []
    for _ in range(n):
        s = d[d.day.isin(rng.choice(dates, len(dates)))]
        ws.append(best_w(s[col_a].to_numpy(), s[col_m].to_numpy(), s[col_l].to_numpy())[0])
    return np.quantile(ws, 0.1), np.quantile(ws, 0.9)


def check(season, results, fetch=True):
    if fetch:
        import prep, fetch_lines
        prep.download([season], force_current=season)
        fetch_lines.run(season)
    r = pd.read_csv(results)
    r = r[r.season == season] if "season" in r else r
    raw = pd.read_parquet(DATA / f"lines_{season}.parquet")[["game_id", "spread_open", "total_open", "spread_close", "total_close"]]
    d = r[["game_id", "day", "margin", "total", "pred_margin", "pred_total"]].merge(raw, on="game_id")
    d = d.dropna(subset=["margin", "pred_margin", "spread_open"])
    d["open_m"], d["close_m"] = -d.spread_open, -d.spread_close
    rows = []
    for lo, hi in BUCKETS + [(0, 999)]:
        b = d[d.day.between(lo, hi)]
        if len(b) < 50:
            continue
        w, mae = best_w(b.margin.to_numpy(), b.pred_margin.to_numpy(), b.open_m.to_numpy())
        q10, q90 = boot_range(b, "margin", "pred_margin", "open_m")
        rows.append(dict(days=f"{lo}-{hi}" if hi < 999 else (f"{lo}+" if lo else "all"), games=len(b),
                         model=np.abs(b.margin - b.pred_margin).mean(), open=np.abs(b.margin - b.open_m).mean(),
                         close=np.abs(b.margin - b.close_m).mean() if b.close_m.notna().all() else np.nan,
                         best_w=w, blend=mae, w_80pct=f"{q10:.2f}-{q90:.2f}",
                         # the "all" row is shown for context but is not a bucket under the rule
                         signal=bool((lo, hi) != (0, 999) and len(b) >= 300 and w >= 0.15 and q10 > 0)))
    tab = pd.DataFrame(rows)
    t = d.dropna(subset=["total", "pred_total", "total_open"])
    tw, tm = best_w(t.total.to_numpy(), t.pred_total.to_numpy(), t.total_open.to_numpy())
    totals = dict(games=len(t), model=np.abs(t.total - t.pred_total).mean(), open=np.abs(t.total - t.total_open).mean(),
                  best_w=tw, blend=tm)
    return tab, totals, d.day.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2027)
    ap.add_argument("--results", default=str(DATA / "out" / "results_games.csv"))
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "blend_check.md"))
    a = ap.parse_args()
    tab, totals, maxday = check(a.season, a.results, fetch=not a.no_fetch)
    reopen = bool(tab.signal.any()) if len(tab) else False
    now = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M CT")
    md = [f"\n## {now}: season {a.season}, through day {maxday}\n",
          "Margin MAE vs result by days into season (w = weight on the model; 0 = opening line only):\n",
          tab.round(3).to_markdown(index=False), "",
          f"Totals: {totals['games']} games, model {totals['model']:.3f} | open {totals['open']:.3f} | "
          f"best w {totals['best_w']:.2f} -> {totals['blend']:.3f}", "",
          f"**Re-open EXP-002? {'YES' if reopen else 'no'}** (rule: a bucket with >= 300 games, best w >= 0.15, "
          f"80% range excluding 0)\n"]
    text = "\n".join(md)
    print(text)
    p = Path(a.out)
    if not p.exists():
        p.write_text("# EXP-007: monthly opening-line re-check\n\nSee experiments/blend_check.py for the method and "
                     "the pre-registered rule.\n")
    with open(p, "a") as f:
        f.write(text)


if __name__ == "__main__":
    main()
