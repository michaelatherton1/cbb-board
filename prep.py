"""Build a clean one-row-per-game table from sportsdataverse ESPN team box scores + schedules."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
RELEASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
BOX_URL = RELEASE + "/espn_mens_college_basketball_team_boxscores/team_box_{}.parquet"
SCHED_URL = RELEASE + "/espn_mens_college_basketball_schedules/mbb_schedule_{}.parquet"


def download(seasons, force_current=None):
    """Download box + schedule parquet files (re-download force_current season every run)."""
    import requests
    DATA.mkdir(exist_ok=True)
    for y in seasons:
        for url, name in [(BOX_URL, f"team_box_{y}.parquet"), (SCHED_URL, f"sched_{y}.parquet")]:
            p = DATA / name
            if p.exists() and y != force_current:
                continue
            r = requests.get(url.format(y), timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)


def poss(fga, orb, tov, fta):
    return fga - orb + tov + 0.475 * fta


def load_season(y):
    box = pd.read_parquet(DATA / f"team_box_{y}.parquet")
    sch = pd.read_parquet(DATA / f"sched_{y}.parquet")

    # D-I teams: anything with a conference id in the schedule (non-D-I opponents have none)
    cid = pd.concat([
        sch[["home_id", "home_conference_id"]].set_axis(["tid", "cid"], axis=1),
        sch[["away_id", "away_conference_id"]].set_axis(["tid", "cid"], axis=1)])
    cid = cid.dropna()
    cid["tid"] = cid.tid.astype(int)
    conf = cid.drop_duplicates("tid", keep="last").set_index("tid").cid.astype(int)

    box = box.copy()
    box["team_id"] = box.team_id.astype(int)
    box["poss_est"] = poss(box.field_goals_attempted, box.offensive_rebounds,
                           box.total_turnovers, box.free_throws_attempted)
    h = box[box.team_home_away == "home"].set_index("game_id")
    a = box[box.team_home_away == "away"].set_index("game_id")
    g = pd.DataFrame({
        "season": y,
        "date": pd.to_datetime(h.game_date),
        "season_type": h.season_type,
        "home_id": h.team_id, "away_id": a.team_id.reindex(h.index),
        "home_name": h.team_location, "away_name": a.team_location.reindex(h.index),
        "home_pts": h.team_score.astype(float), "away_pts": a.team_score.reindex(h.index).astype(float),
        "home_poss": h.poss_est, "away_poss": a.poss_est.reindex(h.index),
    })
    g = g.dropna(subset=["away_id"])
    g["away_id"] = g.away_id.astype(int)
    meta = sch.assign(game_id=sch.game_id.astype(int)).drop_duplicates("game_id").set_index("game_id")
    g["neutral"] = meta.neutral_site.reindex(g.index).fillna(False).astype(bool)
    g["conf_game"] = meta.conference_competition.reindex(g.index).fillna(False).astype(bool)
    g["poss"] = (g.home_poss + g.away_poss) / 2
    g["home_conf"] = g.home_id.map(conf)
    g["away_conf"] = g.away_id.map(conf)
    g["d1"] = g.home_conf.notna() & g.away_conf.notna()
    # basic sanity: drop broken box scores
    ok = (g.poss.between(45, 110)) & (g.home_pts > 20) & (g.away_pts > 20)
    g = g[ok]
    g.index.name = "game_id"
    return g.reset_index()


def upcoming(y, start, end):
    """Scheduled, not-yet-completed games between start and end (inclusive)."""
    sch = pd.read_parquet(DATA / f"sched_{y}.parquet")
    sch["date"] = pd.to_datetime(sch.game_date)
    u = sch[(~sch.status_type_completed.astype(bool)) & sch.date.between(pd.Timestamp(start), pd.Timestamp(end))]
    return pd.DataFrame({
        "game_id": u.game_id.astype(int), "season": y, "date": u.date,
        "tipoff": u.game_date_time,
        "home_id": u.home_id.astype(int), "away_id": u.away_id.astype(int),
        "home_name": u.home_location, "away_name": u.away_location,
        "neutral": u.neutral_site.fillna(False).astype(bool),
        "conf_game": u.conference_competition.fillna(False).astype(bool),
        "home_conf": u.home_conference_id, "away_conf": u.away_conference_id,
    }).assign(d1=lambda x: x.home_conf.notna() & x.away_conf.notna())


def load_all(seasons):
    return pd.concat([load_season(y) for y in seasons], ignore_index=True)


if __name__ == "__main__":
    g = load_all(range(2010, 2028))
    g.to_parquet(DATA / "games.parquet")
    print(g.groupby("season").agg(games=("game_id", "size"), d1=("d1", "sum"),
                                  poss=("poss", "mean"), neutral=("neutral", "mean")).round(2))
