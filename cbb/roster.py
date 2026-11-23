"""Roster-continuity features from player box scores (leak-free: only games before each date).

For each (season, date, team) we know which players have logged minutes so far and where each of them
played last season. That lets the preseason prior follow the players: a team that returns its rotation
keeps more of last year's rating; a team rebuilt through the portal inherits its transfers' old teams.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CONF_ENV = "ROSTER_CONF_FEATURE"     # set to 1/true/yes/on to turn on RosterPrior's conference feature


def player_minutes(season):
    p = pd.read_parquet(DATA / f"player_box_{season}.parquet",
                        columns=["game_id", "game_date", "athlete_id", "team_id", "minutes"])
    p = p.dropna(subset=["athlete_id", "minutes"])
    p = p[p.minutes > 0]
    p["date"] = pd.to_datetime(p.game_date)
    p["team_id"] = p.team_id.astype(int)
    p["athlete_id"] = p.athlete_id.astype(int)
    return p[["date", "team_id", "athlete_id", "minutes"]]


def last_season_map(season):
    """Each athlete's primary team last season and share of available minutes (0-1)."""
    p = player_minutes(season)
    tm = p.groupby(["athlete_id", "team_id"]).minutes.sum().reset_index()
    team_tot = p.groupby("team_id").minutes.sum()
    tm["pct"] = (tm.minutes / (team_tot.reindex(tm.team_id).to_numpy() / 5)).clip(0, 1)
    tm = tm.sort_values("minutes").drop_duplicates("athlete_id", keep="last")
    return tm.set_index("athlete_id")[["team_id", "pct"]].rename(columns={"team_id": "src"})


def composition(season, dates):
    """For each date: team, source team (last season), share of this season's minutes so far, avg prev pct.

    src == -1 means the player has no D-I minutes last season (freshman / JUCO / international / redshirt).
    """
    cur = player_minutes(season)
    prev = last_season_map(season - 1)
    cur = cur.join(prev, on="athlete_id")
    cur["src"] = cur.src.fillna(-1).astype(int)
    cur["pct"] = cur.pct.fillna(0.0)
    cur = cur.sort_values("date")
    out = []
    for d in dates:
        c = cur[cur.date < d]
        if c.empty:
            continue
        a = c.groupby(["team_id", "athlete_id", "src"], as_index=False).agg(minutes=("minutes", "sum"), pct=("pct", "first"))
        a["share"] = a.minutes / a.groupby("team_id").minutes.transform("sum")
        a["share_pct"] = a.share * a.pct
        g = a.groupby(["team_id", "src"], as_index=False).agg(share=("share", "sum"), share_pct=("share_pct", "sum"))
        g["date"] = d
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["team_id", "src", "share", "share_pct", "date"])


FEATS = ["carry", "ret", "ret_w", "trn", "trn_w", "new"]


def _centered(r):
    return {"O": pd.Series(r.O - r.O.mean(), index=r.teams),
            "D": pd.Series(r.D - r.D.mean(), index=r.teams),
            "T": pd.Series(r.T - r.T.mean(), index=r.teams)}


def features(comp_d, last, comp=("O", "D", "T")):
    """Roster features for one date's composition rows, using last season's (centered) ratings."""
    lc = _centered(last)
    out = {}
    c = comp_d
    is_ret = c.src == c.team_id
    is_trn = (~is_ret) & (c.src >= 0) & c.src.isin(lc["O"].index)
    is_new = ~(is_ret | is_trn)
    for k in comp:
        v = c.src.map(lc[k]).fillna(0.0)
        f = pd.DataFrame({
            "team_id": c.team_id,
            "ret": np.where(is_ret, c.share * v, 0), "ret_w": np.where(is_ret, c.share_pct * v, 0),
            "trn": np.where(is_trn, c.share * v, 0), "trn_w": np.where(is_trn, c.share_pct * v, 0),
            "new": np.where(is_new, c.share, 0)}).groupby("team_id").sum()
        f["carry"] = lc[k].reindex(f.index).fillna(np.nan)
        out[k] = f[FEATS]
    return out


def conf_avg(last, conf):
    """comp -> Series(team -> conference mean of last season's centered rating), over the teams in `conf`.

    `conf` is the CURRENT season's team -> conference map (model.season_conf), so this is the same
    conference average that model.make_prior mixes in with weight w_conf.
    """
    lc = _centered(last)
    out = {}
    for k, s in lc.items():
        df = pd.DataFrame({"c": conf, "v": s.reindex(conf.index)})
        out[k] = df.groupby("c").v.transform("mean")
    return out


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class RosterPrior:
    """Linear map from roster features -> season-final rating, refit each season on prior seasons only.

    conf_feature (EXP-008): add a "conf" regressor = the current-season conference average of last
    season's centered ratings (the quantity make_prior shrinks toward with w_conf). Teams without a
    conference fall back to their own carry-over rating. Conference membership comes from the season's
    schedule (known preseason, the same source walk_forward feeds make_prior); ratings are last season's
    finals; roster minutes are games before the date, so the feature is leak-free.

    conf_feature=None (the default) reads the ROSTER_CONF_FEATURE environment variable (1/true/yes/on
    turns it on; unset or anything else is off), so e.g.
        ROSTER_CONF_FEATURE=1 python experiments/scoreboard.py <label> --roster
    selects it without code changes. Off reproduces the original RosterPrior exactly.

    games: optional games frame for the conference map; loaded lazily from data/games.parquet if needed.
    """

    def __init__(self, first_fit_season=2012, snap_every=4, snap_until_day=60, conf_feature=None, games=None):
        self.first = first_fit_season
        self.snap_every, self.snap_until = snap_every, snap_until_day
        self.conf_feature = _env_flag(CONF_ENV) if conf_feature is None else bool(conf_feature)
        self.feats = FEATS + ["conf"] if self.conf_feature else list(FEATS)
        self.coef = {}          # season -> {comp: (coef array, intercept)}
        self.fit_seasons = {}   # season -> training seasons used by fit_for (all < season)
        self.comp_cache = {}
        self.train_rows = {}    # season -> {comp: (X, y)}
        self._games = games
        self._conf = {}

    def comp(self, s):
        if s not in self.comp_cache:
            p = DATA / f"roster_comp_{s}.parquet"
            self.comp_cache[s] = pd.read_parquet(p) if p.exists() else None
        return self.comp_cache[s]

    def season_conf(self, s):
        """Team -> conference for season s, exactly as walk_forward builds it for make_prior."""
        if s not in self._conf:
            from model import season_conf
            if self._games is None:
                self._games = pd.read_parquet(DATA / "games.parquet")
            g = self._games
            self._conf[s] = season_conf(g[(g.season == s) & g.d1])
        return self._conf[s]

    def _features(self, s, comp_d, last):
        f = features(comp_d, last)
        if not self.conf_feature:
            return f
        ca = conf_avg(last, self.season_conf(s))
        for k in f:
            # teams without a conference (shouldn't happen for D-I) fall back to their own carry-over
            f[k] = f[k].assign(conf=ca[k].reindex(f[k].index).fillna(f[k]["carry"]))[self.feats]
        return f

    def add_training_season(self, s, last, final):
        """Store (features, target) snapshots for season s once its final ratings are known."""
        c = self.comp(s)
        if c is None or last is None:
            return
        dates = sorted(c.date.unique())
        d0 = dates[0]
        snaps = [d for i, d in enumerate(dates) if i % self.snap_every == 0 and (d - d0).days <= self.snap_until]
        fc = _centered(final)
        rows = {k: [] for k in "ODT"}
        for d in snaps:
            f = self._features(s, c[c.date == d], last)
            for k in "ODT":
                x = f[k].dropna()
                y = fc[k].reindex(x.index)
                ok = y.notna()
                rows[k].append((x[ok].to_numpy(), y[ok].to_numpy()))
        self.train_rows[s] = {k: (np.vstack([a for a, _ in v]), np.concatenate([b for _, b in v]))
                              for k, v in rows.items() if v}

    def fit_for(self, s):
        """Fit coefficients using every stored season < s."""
        prev = sorted(t for t in self.train_rows if self.first <= t < s)
        if len(prev) < 2:
            return None
        self.fit_seasons[s] = prev
        res = {}
        for k in "ODT":
            X = np.vstack([self.train_rows[t][k][0] for t in prev])
            y = np.concatenate([self.train_rows[t][k][1] for t in prev])
            A = np.c_[X, np.ones(len(X))]
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
            res[k] = beta
        self.coef[s] = res
        return res

    def prior_at(self, s, d, last, teams):
        """Return dict comp -> Series(team -> prior deviation) for teams with roster data before date d."""
        c = self.comp(s)
        if c is None or last is None or s not in self.coef:
            return None
        cd = c[c.date == d]
        if cd.empty:
            # latest available snapshot strictly before d
            prior_dates = c.date[c.date < d]
            if prior_dates.empty:
                return None
            cd = c[c.date == prior_dates.max()]
        f = self._features(s, cd, last)
        out = {}
        for k in "ODT":
            x = f[k].dropna()
            beta = self.coef[s][k]
            out[k] = pd.Series(x.to_numpy() @ beta[:-1] + beta[-1], index=x.index)
        return out


def build_cache(seasons):
    games = pd.read_parquet(DATA / "games.parquet")
    for s in seasons:
        dates = sorted(games[games.season == s].date.unique())
        comp = composition(s, dates)
        comp.to_parquet(DATA / f"roster_comp_{s}.parquet")
        print(s, len(comp), flush=True)


if __name__ == "__main__":
    import sys
    build_cache([int(x) for x in sys.argv[1:]] or range(2011, 2028))
