"""Daily run: refresh current-season data, rate every D-I team as of today, project upcoming games.

    python cbb/daily.py                      # today + tomorrow
    python cbb/daily.py --date 2026-11-11 --days 3

Outputs (data/out/): projections_<date>.csv, ratings_<date>.csv
"""
import argparse, json, sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).parent))
import prep
from model import SeasonModel, make_prior, predict, season_teams, walk_forward
from backtest import load_params, apply_calibration

DATA = prep.DATA
OUT = DATA / "out"


def season_for(d):
    return d.year + 1 if d.month >= 7 else d.year


def american(p):
    p = np.clip(p, 0.0099, 0.9901)          # display cap at +/-10000
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p).round(0)


def season_model(games, cur, asof, up, last, p, conf):
    """Season `cur` ratings model as of `asof`: D-I games strictly before `asof`, and the same preseason
    prior walk_forward (the backtest) uses - last season's final ratings pulled toward this season's
    conference averages.

    conf: Series team_id -> conference id for season `cur`, from its schedule (prep.conf_map). Every team in
    it is rated, so the conference averages cover the same members as the backtest's full-season team set.
    Returns (games used, SeasonModel).
    """
    now = games[(games.season == cur) & games.d1 & (games.date < pd.Timestamp(asof))]
    teams = np.union1d(season_teams(now) if len(now) else np.array([], int),
                       np.r_[up.home_id.to_numpy(), up.away_id.to_numpy()])
    teams = np.union1d(teams, conf.index.to_numpy())
    sm = SeasonModel(now if len(now) else games[games.season == cur].head(0), teams,
                     make_prior(last, teams, p, conf=conf), p)
    return now, sm


def main(asof: date, days: int, refresh=True):
    cur = season_for(asof)
    seasons = list(range(max(2010, cur - 15), cur + 1))   # same prior chain as the backtest (from 2012)
    if refresh:
        prep.download(seasons, force_current=cur)
    p = load_params()
    cal = json.loads((DATA / "calibration.json").read_text())

    games = prep.load_all(seasons)
    hist = games[games.season < cur]
    _, finals = walk_forward(hist, [cur - 1], p, start=seasons[0])   # chain priors up to last season's final
    last = finals[cur - 1]

    up = prep.upcoming(cur, asof, asof + timedelta(days=days - 1))
    up = up[up.d1]
    now, sm = season_model(games, cur, asof, up, last, p, prep.conf_map(cur))
    r = sm.solve(pd.Timestamp(asof))

    poss, ph, pa = predict(r, sm.idx, up.home_id, up.away_id, up.neutral, up.conf_game)
    season_start = games[games.season == cur].date.min()
    if pd.isna(season_start):
        season_start = pd.Timestamp(asof)
    c = apply_calibration(up.assign(raw_margin=ph - pa, raw_total=ph + pa,
                                    day=(up.date - season_start).dt.days.clip(lower=0)), cal)
    m, t, phome = c.pred_margin.to_numpy(), c.pred_total.to_numpy(), c.p_home.to_numpy()
    ph_cal, pa_cal = (t + m) / 2, (t - m) / 2
    gp = dict(zip(r.teams, r.n_games))
    proj = pd.DataFrame({
        "date": up.date.dt.date, "tipoff": up.tipoff, "game_id": up.game_id,
        "home_id": up.home_id, "away_id": up.away_id,
        "away": up.away_name, "home": up.home_name, "neutral": up.neutral,
        "away_pts": pa_cal.round(1), "home_pts": ph_cal.round(1),
        "home_spread": (-m).round(1), "total": t.round(1), "poss": poss.round(1),
        "home_win_prob": phome.round(3),
        "fair_ml_home": american(phome), "fair_ml_away": american(1 - phome),
        "home_games_played": up.home_id.map(gp).to_numpy(), "away_games_played": up.away_id.map(gp).to_numpy(),
    }).sort_values(["date", "tipoff"])

    rat = r.frame()
    names = pd.concat([games[["home_id", "home_name"]].set_axis(["team_id", "team"], axis=1),
                       games[["away_id", "away_name"]].set_axis(["team_id", "team"], axis=1),
                       up[["home_id", "home_name"]].set_axis(["team_id", "team"], axis=1),
                       up[["away_id", "away_name"]].set_axis(["team_id", "team"], axis=1)]).drop_duplicates("team_id", keep="last")
    rat = rat.merge(names, on="team_id", how="left")
    rat.insert(0, "rank", np.arange(1, len(rat) + 1))
    rat = rat[["rank", "team", "team_id", "games", "adj_em", "adj_o", "adj_d", "adj_t"]].round(2)

    OUT.mkdir(parents=True, exist_ok=True)
    proj.to_csv(OUT / f"projections_{asof}.csv", index=False)
    rat.to_csv(OUT / f"ratings_{asof}.csv", index=False)
    meta = dict(asof=str(asof), season=cur, d1_games_this_season=int(len(now)),
                league_eff=round(float(r.mu + r.O.mean() + r.D.mean()), 2),
                league_poss=round(float(r.mu_t + 2 * r.T.mean()), 2),
                hca_points_conf=round(float(2 * r.hca * (r.mu_t + 2 * r.T.mean()) / 100 * cal["bm"][0]), 2),
                hca_points_nonconf=round(float(2 * (r.hca + r.hca_nc) * (r.mu_t + 2 * r.T.mean()) / 100 * cal["bm"][0]), 2))
    (OUT / f"meta_{asof}.json").write_text(json.dumps(meta, indent=1))
    return proj, rat, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(datetime.now(ZoneInfo("America/New_York")).date()))  # slate date in ET
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    proj, rat, meta = main(date.fromisoformat(a.date), a.days, refresh=not a.no_refresh)
    pd.set_option("display.width", 220)
    print(meta)
    print(rat.head(25).to_string(index=False))
    print(proj.drop(columns=["game_id", "tipoff"]).to_string(index=False))
