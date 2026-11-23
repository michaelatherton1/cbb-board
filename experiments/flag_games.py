"""List today's games for the news check (EXP-003): model vs sportsbook consensus differ by >= N points.

    python experiments/flag_games.py --data-dir <path to main's data/out> [--date YYYY-MM-DD] [--out flagged.csv]

Reads the newest projections_*.csv and board_*.csv that contain the target date (the overnight job already projects
"tomorrow", so this works before the morning rebuild). Book consensus = median home spread across DraftKings,
FanDuel and BetMGM (board rows with market == "spread" and side == "home"). Keeps games that tip at least
--min-minutes after now, sorted by tip; if more than --max-games qualify, keeps the largest gaps.
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

BOOKS = ["DraftKings Sportsbook", "FanDuel Sportsbook", "BetMGM"]
CT = ZoneInfo("America/Chicago")


def newest_with_date(folder, pattern, date):
    for f in sorted(Path(folder).glob(pattern), reverse=True):
        d = pd.read_csv(f)
        if "date" in d and (d.date.astype(str) == date).any():
            return f, d[d.date.astype(str) == date]
    return None, None


def flag(data_dir, date, min_diff=3.0, max_games=10, min_minutes=30, now=None):
    pf, proj = newest_with_date(data_dir, "projections_*.csv", date)
    bf, board = newest_with_date(data_dir, "board_*.csv", date)
    if proj is None:
        raise SystemExit(f"no projections for {date} in {data_dir}")
    if board is None:
        raise SystemExit(f"no board (prices) for {date} in {data_dir}")
    books = board[(board.market == "spread") & board.venue.isin(BOOKS) & (board.side == "home")]
    cons = books.groupby("game_id").agg(book_home_spread=("line", "median"), books=("venue", "nunique"))
    m = proj.merge(cons, on="game_id", how="left")
    m["tip"] = pd.to_datetime(m.tipoff, utc=True)
    m["tip_ct"] = m.tip.dt.tz_convert(CT).dt.strftime("%H:%M")
    m["diff"] = (m.home_spread - m.book_home_spread).round(1)
    now = now or datetime.now(timezone.utc)
    live = m[(m.tip > now + timedelta(minutes=min_minutes)) & m.book_home_spread.notna()]
    out = live[live["diff"].abs() >= min_diff]
    if len(out) > max_games:
        out = out.reindex(out["diff"].abs().sort_values(ascending=False).index).head(max_games)
    out = out.sort_values("tip")
    cols = ["game_id", "tip_ct", "away", "home", "neutral", "home_spread", "book_home_spread", "books", "diff", "total"]
    info = dict(projections=pf.name, board=bf.name, games=len(m), not_started_with_books=len(live),
                flagged=len(live[live["diff"].abs() >= min_diff]), kept=len(out))
    return out[cols].rename(columns={"home_spread": "model_home_spread"}), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "out"))
    ap.add_argument("--date", default=datetime.now(CT).date().isoformat())
    ap.add_argument("--min-diff", type=float, default=3.0)
    ap.add_argument("--max-games", type=int, default=10)
    ap.add_argument("--min-minutes", type=int, default=30)
    ap.add_argument("--out")
    a = ap.parse_args()
    out, info = flag(a.data_dir, a.date, a.min_diff, a.max_games, a.min_minutes)
    print(info)
    print(out.to_markdown(index=False) if len(out) else "no flagged games")
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
