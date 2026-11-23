"""Build the 2018-2023 closing-line history used for Part A (distribution around the line)."""
import numpy as np, pandas as pd
from pathlib import Path
DATA = Path("/home/claude/wt/exp-005-kalshi-ladder/data")

def load(seasons=range(2018, 2024)):
    g = pd.read_parquet(DATA / "games.parquet")
    g = g[g.season.isin(list(seasons))].copy()
    g["margin"] = g.home_pts - g.away_pts
    g["total"] = g.home_pts + g.away_pts
    ls = pd.concat([pd.read_parquet(DATA / f"lines_{y}.parquet") for y in seasons])
    ls = ls[ls.home_team_id.astype(float).notna()]
    ls = ls.drop_duplicates("game_id")
    d = g.merge(ls[["game_id", "provider", "spread_close", "total_close", "ml_home_close", "ml_away_close",
                    "home_team_id", "details"]], on="game_id", how="inner")
    # home team id in the line feed must match the box score home team
    d["id_ok"] = d.home_team_id.astype(float).astype("Int64") == d.home_id
    # ML-based sign fix (as cbb/backtest.py attach_lines), but applied to all games with a usable ML
    has_ml = d.ml_home_close.notna() & d.ml_away_close.notna() & (d.ml_home_close != d.ml_away_close)
    ml_sign = np.where(d.ml_home_close < d.ml_away_close, -1.0, 1.0)   # -1: home favored
    sp = d.spread_close
    d["sign_disagree"] = has_ml & sp.notna() & (sp != 0) & (np.sign(sp) != ml_sign)
    fix = d.neutral & has_ml & sp.notna()
    d["spread"] = sp
    d.loc[fix, "spread"] = ml_sign[fix] * sp[fix].abs()
    d["m"] = -d.spread                        # market home margin
    d["day"] = (d.date - d.groupby("season").date.transform("min")).dt.days
    return d

if __name__ == "__main__":
    d = load()
    print(len(d), d.id_ok.mean())
    print(d.groupby(["season", "neutral"]).sign_disagree.mean().unstack())
    print(d.groupby("provider").apply(lambda x: pd.Series(dict(n=len(x), sd=(x.margin - x.m).std(),
          mean=(x.margin - x.m).mean(), tsd=(x.total - x.total_close).std(), dis=x.sign_disagree.mean()))))
