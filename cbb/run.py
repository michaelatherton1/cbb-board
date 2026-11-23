"""One command for the scheduled job.

    python cbb/run.py                 # refresh data + model once a day, then prices -> dashboard
    python cbb/run.py --prices-only   # just re-pull prices and rebuild the dashboard
    python cbb/run.py --quotes FILE   # use saved quotes instead of live prices (testing)

Writes data/out/* (CSV/JSON) and docs/cbb/index.html (the dashboard).
"""
import argparse, json, sys, time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import prep, daily, results, board, webpage

DATA, OUT = prep.DATA, prep.DATA / "out"


def season_of(d):
    return d.year + 1 if d.month >= 7 else d.year


def team_records(season):
    """Win-loss so far this season for every team (all games, D-I or not)."""
    try:
        g = prep.load_season(season)
    except Exception:
        return {}
    w = pd.concat([g.home_id.where(g.home_pts > g.away_pts), g.away_id.where(g.away_pts > g.home_pts)]).dropna().astype(int)
    l = pd.concat([g.home_id.where(g.home_pts < g.away_pts), g.away_id.where(g.away_pts < g.home_pts)]).dropna().astype(int)
    wc, lc = w.value_counts(), l.value_counts()
    return {int(t): f"{int(wc.get(t, 0))}-{int(lc.get(t, 0))}" for t in set(wc.index) | set(lc.index)}


def main(asof, days=2, prices_only=False, quotes=None, venues_only=None):
    t0 = time.time()
    cur = season_of(asof)
    stamp = OUT / f"model_{asof}.json"
    if not prices_only and not stamp.exists():
        # 1) teams + schedule: re-download this season's box scores and schedule, then re-rate and project
        prep.download(range(max(2010, cur - 17), cur + 1), force_current=cur)
        proj, rat, meta = daily.main(asof, days, refresh=False)
        # 2) how the model has done so far (pre-game lines vs DraftKings closes)
        res = results.main(cur)
        stamp.write_text(json.dumps(dict(meta, built=datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="minutes"),
                                         results_through=res["through"])))
        print(f"model: {meta}  ({time.time() - t0:.0f}s)", flush=True)
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    # 3) prices from every venue with a feed, matched to games and scored against the model
    b, edges, arbs, status = board.main(asof, days, quotes, venues_only)
    print(f"prices: {status}  arbs={len(arbs)} edges={len(edges)}  ({time.time() - t0:.0f}s)", flush=True)
    # 4) dashboard
    proj = board.load_projections(asof, days)
    rat = pd.read_csv(OUT / f"ratings_{asof}.csv") if (OUT / f"ratings_{asof}.csv").exists() else None
    res = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else None
    cal = json.loads((DATA / "calibration.json").read_text())
    payload = webpage.build(proj, b, status, res, rat, cal, asof,
                         model_asof=(res or {}).get("through"), records=team_records(cur))
    path = webpage.render(payload)
    print(f"dashboard: {path}  ({time.time() - t0:.0f}s)")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(datetime.now(ZoneInfo("America/New_York")).date()))
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--prices-only", action="store_true")
    ap.add_argument("--quotes")
    ap.add_argument("--venues", nargs="*")
    a = ap.parse_args()
    main(date.fromisoformat(a.date), a.days, a.prices_only, a.quotes, a.venues)
