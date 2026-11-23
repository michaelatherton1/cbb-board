"""Model record for the current season: each completed D-I game, predicted the morning of game day
(only earlier games used), against DraftKings closing lines from ESPN.

    python cbb/results.py            -> data/out/results.json (+ results_games.csv)
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import prep
from model import walk_forward
from backtest import load_params, apply_calibration, attach_lines
from fetch_lines import fetch

DATA, OUT = prep.DATA, prep.DATA / "out"
WIN, LOSS = 100 / 110, -1.0        # $1 risked at -110


def update_lines(season, game_ids):
    """Fetch closing lines for games not yet cached (ESPN game JSON mirrored on GitHub)."""
    path = DATA / f"lines_{season}.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["game_id"])
    need = sorted(set(int(g) for g in game_ids) - set(have.game_id.astype(int)))
    if need:
        with ThreadPoolExecutor(16) as ex:
            rows = [r for r in ex.map(fetch, need) if isinstance(r, dict)]
        if rows:
            have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("game_id", keep="last")
            have.to_parquet(path)
    return len(need)


def _record(edge, res):
    m = (res != 0) & edge.notna() & res.notna()
    won = np.sign(edge[m]) == np.sign(res[m])
    return int(won.sum()), int((~won).sum()), round(float(won.sum() * WIN + (~won).sum() * LOSS), 2)


def summarize(d):
    w = d[d.spread_close.notna()]
    t = d[d.total_close.notna()]
    ats_e, ats_r = w.pred_margin + w.spread_close, w.margin + w.spread_close
    ou_e, ou_r = t.pred_total - t.total_close, t.total - t.total_close
    big = lambda e, k: e.where(e.abs() >= k)
    a = _record(ats_e, ats_r)
    a3 = _record(big(ats_e, 3), ats_r)
    o = _record(ou_e, ou_r)
    o3 = _record(big(ou_e, 3), ou_r)
    return dict(
        games=int(len(d)),
        winners=int(((d.pred_margin > 0) == (d.margin > 0)).sum()),
        fav_winners=int(((w.spread_close < 0) == (w.margin > 0)).sum()), lined=int(len(w)),
        model_mae=round(float((w.margin - w.pred_margin).abs().mean()), 2) if len(w) else None,
        market_mae=round(float((w.margin + w.spread_close).abs().mean()), 2) if len(w) else None,
        model_total_mae=round(float((t.total - t.pred_total).abs().mean()), 2) if len(t) else None,
        market_total_mae=round(float((t.total - t.total_close).abs().mean()), 2) if len(t) else None,
        ats_w=a[0], ats_l=a[1], ats_units=a[2], ats3_w=a3[0], ats3_l=a3[1], ats3_units=a3[2],
        ou_w=o[0], ou_l=o[1], ou_units=o[2], ou3_w=o3[0], ou3_l=o3[1], ou3_units=o3[2])


def main(season, refresh=False):
    if refresh:
        prep.download(range(max(2010, season - 17), season + 1), force_current=season)
    games = prep.load_all(range(2010, season + 1))
    pred, _ = walk_forward(games, [season], load_params(), start=2010)
    cal = json.loads((DATA / "calibration.json").read_text())
    pred = apply_calibration(pred, cal)
    fetched = update_lines(season, pred.game_id)
    d = attach_lines(pred).sort_values(["date", "game_id"])
    d["date_s"] = d.date.dt.strftime("%Y-%m-%d")

    days = []
    cum_ats = cum_ou = 0.0
    for day, g in d.groupby("date_s"):
        s = summarize(g)
        cum_ats += s["ats_units"]
        cum_ou += s["ou_units"]
        days.append(dict(date=day, **s, cum_ats_units=round(cum_ats, 2), cum_ou_units=round(cum_ou, 2)))

    def game_row(r):
        ats = None if pd.isna(r.spread_close) or (r.margin + r.spread_close) == 0 else bool(
            np.sign(r.pred_margin + r.spread_close) == np.sign(r.margin + r.spread_close))
        ou = None if pd.isna(r.total_close) or (r.total - r.total_close) == 0 else bool(
            np.sign(r.pred_total - r.total_close) == np.sign(r.total - r.total_close))
        return dict(date=r.date_s, away=r.away_name, home=r.home_name, neutral=bool(r.neutral),
                    model_home_spread=round(-float(r.pred_margin), 1), model_total=round(float(r.pred_total), 1),
                    model_home_win=round(float(r.p_home), 3),
                    close_home_spread=None if pd.isna(r.spread_close) else float(r.spread_close),
                    close_total=None if pd.isna(r.total_close) else float(r.total_close),
                    away_pts=int(r.away_pts), home_pts=int(r.home_pts),
                    winner_right=bool((r.pred_margin > 0) == (r.margin > 0)), ats_right=ats, ou_right=ou)

    recent_days = sorted(d.date_s.unique())[-3:]
    out = dict(season=season, through=d.date_s.max(), lines_fetched_now=fetched,
               season_summary=summarize(d), days=days,
               last7=summarize(d[d.date >= d.date.max() - pd.Timedelta(days=6)]),
               recent_games=[game_row(r) for r in d[d.date_s.isin(recent_days)].itertuples()],
               notes="Pre-game model lines (rated the morning of each game day) vs DraftKings closing lines via ESPN. "
                     "Units = $1 risked per game at -110.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    d.drop(columns=["date"]).to_csv(OUT / "results_games.csv", index=False)
    return out


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ap = argparse.ArgumentParser()
    today = datetime.now(ZoneInfo("America/New_York")).date()
    ap.add_argument("--season", type=int, default=today.year + 1 if today.month >= 7 else today.year)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    r = main(a.season, a.refresh)
    print(json.dumps({k: r[k] for k in ("through", "season_summary", "last7")}, indent=1))
    print(pd.DataFrame(r["days"])[["date", "games", "winners", "ats_w", "ats_l", "ou_w", "ou_l", "cum_ats_units",
                                   "cum_ou_units"]].tail(8).to_string(index=False))
