"""Opponent-adjusted efficiency + tempo ratings, refit daily with only prior information.

Efficiency model (one row per team-game):
    pts_per_100(off i vs def j) = mu + O_i + D_j + h * loc_i + e      loc = +1 home, -1 away, 0 neutral
Tempo model (one row per game):
    possessions = mu_t + T_home + T_away + e

Each day's fit is a weighted ridge regression whose penalty pulls every team toward a preseason
prior = r * (last season's final rating). Recency weights decay with a half-life in days.

The O/D penalty is a 2x2 block per team: strength lam_eff on quality q = (O - D)/2 (drives margins) and
lam_style on style s = (O + D)/2 (drives totals only; margins are invariant except under a RosterPrior override,
whose O/D regressions see lam_style through past ratings). lam_style = None (or = lam_eff) is the plain
diagonal ridge lam_eff * ((O - O0)^2 + (D - D0)^2). Without a roster override, margin predictions do not depend on lam_style.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve


@dataclass
class Params:
    half_life: float = 60.0      # days; recency weighting of games inside a season
    lam_eff: float = 6.0         # prior strength for O/D, in pseudo-games (on quality (O-D)/2 if lam_style is set)
    lam_style: float | None = None  # prior strength on style (O+D)/2 (totals only); None -> lam_eff (plain ridge)
    lam_tempo: float = 6.0       # prior strength for tempo, in pseudo-games
    r_eff: float = 0.65          # carry-over of last season's O/D deviation into this season's prior
    r_tempo: float = 0.75        # carry-over of last season's tempo deviation
    w_conf: float = 0.3          # share of the O/D prior taken from the team's (current) conference average
    new_team_eff: float = -8.0   # prior net rating (per side, pts/100) for teams with no prior-season rating
    blowout_cap: float = 999.0   # cap on per-game efficiency differential (pts/100)
    hca_prior: float = 1.4       # per-side home edge, pts/100 (margin edge ~ 2 * hca * poss / 100)
    hca_nc_prior: float = 0.0    # extra per-side home edge in NON-conference games (buy games, travel)
    lam_hca: float = 2000.0
    lam_mu: float = 50.0


@dataclass
class Ratings:
    teams: np.ndarray
    mu: float
    hca: float
    O: np.ndarray
    D: np.ndarray
    mu_t: float
    T: np.ndarray
    n_games: np.ndarray = field(default=None)
    hca_nc: float = 0.0

    def frame(self):
        df = pd.DataFrame({"team_id": self.teams, "O": self.O, "D": self.D, "T": self.T,
                           "games": self.n_games})
        # vs an average opponent on a neutral floor
        df["adj_o"] = self.mu + df.O + self.D.mean()
        df["adj_d"] = self.mu + df.D + self.O.mean()
        df["adj_em"] = df.adj_o - df.adj_d
        df["adj_t"] = self.mu_t + df["T"] + self.T.mean()
        return df.sort_values("adj_em", ascending=False)


class SeasonModel:
    """Holds one season's D-I games and solves the ratings for any cutoff date."""

    def __init__(self, games, teams, prior, p: Params):
        self.p = p
        self.teams = np.asarray(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        self.n = n
        g = games.sort_values("date").reset_index(drop=True)
        self.g = g
        hi = g.home_id.map(self.idx).to_numpy()
        ai = g.away_id.map(self.idx).to_numpy()
        loc = np.where(g.neutral.to_numpy(), 0.0, 1.0)
        nc = np.where(g.conf_game.to_numpy().astype(bool), 0.0, 1.0)
        m = len(g)
        self.day = (g.date - g.date.min()).dt.days.to_numpy()
        self.d0 = g.date.min()

        # efficiency targets (pts / 100 poss), optional blowout cap on the differential
        yh = 100 * g.home_pts.to_numpy() / g.poss.to_numpy()
        ya = 100 * g.away_pts.to_numpy() / g.poss.to_numpy()
        mid, diff = (yh + ya) / 2, np.clip(yh - ya, -p.blowout_cap, p.blowout_cap)
        yh, ya = mid + diff / 2, mid - diff / 2

        # columns: 0 mu, 1 hca, 2 hca_nc, 3..3+n O, 3+n..3+2n D
        rows = np.r_[np.arange(m), np.arange(m) + m]
        off = np.r_[hi, ai]
        dfn = np.r_[ai, hi]
        L = np.r_[loc, -loc]
        LN = L * np.r_[nc, nc]
        r_ = np.repeat(rows, 5)
        c_ = np.c_[np.zeros(2 * m, int), np.ones(2 * m, int), np.full(2 * m, 2), 3 + off, 3 + n + dfn].ravel()
        v_ = np.c_[np.ones(2 * m), L, LN, np.ones(2 * m), np.ones(2 * m)].ravel()
        self.X = sp.csr_matrix((v_, (r_, c_)), shape=(2 * m, 3 + 2 * n))
        self.y = np.r_[yh, ya]
        self.row_day = np.r_[self.day, self.day]

        # tempo: columns 0 mu_t, 1..n T
        r2 = np.repeat(np.arange(m), 3)
        c2 = np.c_[np.zeros(m, int), 1 + hi, 1 + ai].ravel()
        self.Xt = sp.csr_matrix((np.ones(3 * m), (r2, c2)), shape=(m, 1 + n))
        self.yt = g.poss.to_numpy()

        # priors
        self.prior = prior
        self.b0 = np.r_[prior["mu"], prior["hca"], prior["hca_nc"], prior["O"], prior["D"]]
        # O/D penalty, a 2x2 block per team: lam_eff on (O-D)/2, lam_style on (O+D)/2.
        #   P[O_i,O_i] = P[D_i,D_i] = (lam_eff + lam_style)/2,  P[O_i,D_i] = P[D_i,O_i] = (lam_style - lam_eff)/2
        # self.pen is P's diagonal; self.couple = (O cols, D cols, off-diagonal value), None when it is 0.
        lam_s = p.lam_eff if p.lam_style is None else p.lam_style
        a, c = (p.lam_eff + lam_s) / 2, (lam_s - p.lam_eff) / 2
        self.pen = np.r_[p.lam_mu, p.lam_hca, p.lam_hca, np.full(n, a), np.full(n, a)]
        self.couple = (3 + np.arange(n), 3 + n + np.arange(n), c) if c != 0 else None
        self.b0t = np.r_[prior["mu_t"], prior["T"]]
        self.pent = np.r_[p.lam_mu, np.full(n, p.lam_tempo)]
        self.hi, self.ai = hi, ai

    def solve(self, date, override=None):
        """Ratings using only games strictly before `date`.

        override: optional dict comp -> Series(team_id -> prior deviation) replacing the preseason prior
        """
        p = self.p
        b0, b0t = self.b0, self.b0t
        if override:
            b0, b0t = self.b0.copy(), self.b0t.copy()
            n = self.n
            for comp, ser in override.items():
                ser = ser[ser.index.isin(self.idx)]
                ii = np.array([self.idx[t] for t in ser.index], dtype=int)
                if comp == "O":
                    b0[3 + ii] = ser.to_numpy()
                elif comp == "D":
                    b0[3 + n + ii] = ser.to_numpy()
                elif comp == "T":
                    b0t[1 + ii] = ser.to_numpy()
        cut = (pd.Timestamp(date) - self.d0).days
        sel = self.row_day < cut
        selt = self.day < cut
        if p.half_life and p.half_life < 1e6:
            w = 0.5 ** ((cut - self.row_day[sel]) / p.half_life)
            wt = 0.5 ** ((cut - self.day[selt]) / p.half_life)
        else:
            w, wt = np.ones(sel.sum()), np.ones(selt.sum())
        b = self._ridge(self.X[sel], self.y[sel], w, b0, self.pen, self.couple)
        bt = self._ridge(self.Xt[selt], self.yt[selt], wt, b0t, self.pent)
        n = self.n
        ng = np.bincount(np.r_[self.hi[selt], self.ai[selt]], minlength=n)
        return Ratings(self.teams, b[0], b[1], b[3:3 + n], b[3 + n:], bt[0], bt[1:], ng, hca_nc=b[2])

    @staticmethod
    def _ridge(X, y, w, b0, pen, couple=None):
        """argmin sum w (y - Xb)^2 + (b - b0)' P (b - b0), P = diag(pen) + symmetric pairs from `couple`.

        couple: optional (i, j, c) adding c at P[i, j] and P[j, i] (index arrays i, j; scalar c).
        """
        if X.shape[0] == 0:
            return b0.copy()
        Xw = X.multiply(w[:, None]).tocsr()
        A = (X.T @ Xw).toarray()
        A[np.diag_indices_from(A)] += pen
        rhs = Xw.T @ y + pen * b0
        if couple is not None:
            i, j, c = couple
            A[i, j] += c
            A[j, i] += c
            rhs[i] += c * b0[j]
            rhs[j] += c * b0[i]
        return cho_solve(cho_factor(A), rhs)


def predict(r: Ratings, idx, home_ids, away_ids, neutral, conf_game):
    """Raw (uncalibrated) expected possessions, home/away points from a Ratings snapshot."""
    hi = np.array([idx[t] for t in home_ids], dtype=int)
    ai = np.array([idx[t] for t in away_ids], dtype=int)
    loc = np.where(np.asarray(neutral), 0.0, 1.0)
    h = r.hca + r.hca_nc * np.where(np.asarray(conf_game).astype(bool), 0.0, 1.0)
    poss = r.mu_t + r.T[hi] + r.T[ai]
    eh = r.mu + r.O[hi] + r.D[ai] + h * loc
    ea = r.mu + r.O[ai] + r.D[hi] - h * loc
    return poss, eh * poss / 100, ea * poss / 100


def make_prior(last: Ratings | None, teams, p: Params, conf=None, default_mu=103.0, default_t=68.0):
    """Preseason prior: last season's final ratings, pulled toward the team's conference average.

    conf: optional Series team_id -> conference id for the CURRENT season (handles realignment).
    """
    n = len(teams)
    if last is None:
        return dict(mu=default_mu, hca=p.hca_prior, hca_nc=p.hca_nc_prior, O=np.zeros(n), D=np.zeros(n),
                    mu_t=default_t, T=np.zeros(n))
    lidx = {t: i for i, t in enumerate(last.teams)}
    O, D, T = np.zeros(n), np.zeros(n), np.zeros(n)
    # center last season's ratings so the prior is a pure deviation from average
    lo, ld, lt = last.O - last.O.mean(), last.D - last.D.mean(), last.T - last.T.mean()
    has = np.array([t in lidx for t in teams])
    jj = np.array([lidx.get(t, 0) for t in teams])
    own_o, own_d = np.where(has, lo[jj], np.nan), np.where(has, ld[jj], np.nan)
    if conf is not None and p.w_conf > 0:
        cf = pd.Series(conf).reindex(teams).to_numpy()
        df = pd.DataFrame({"c": cf, "o": own_o, "d": own_d})
        co = df.groupby("c").o.transform("mean").to_numpy()
        cd = df.groupby("c").d.transform("mean").to_numpy()
        co, cd = np.where(np.isnan(co), own_o, co), np.where(np.isnan(cd), own_d, cd)
    else:
        co, cd = own_o, own_d
    for k, t in enumerate(teams):
        if has[k]:
            j = jj[k]
            O[k] = p.r_eff * ((1 - p.w_conf) * own_o[k] + p.w_conf * co[k])
            D[k] = p.r_eff * ((1 - p.w_conf) * own_d[k] + p.w_conf * cd[k])
            T[k] = p.r_tempo * lt[j]
        else:
            O[k], D[k] = p.new_team_eff / 2, -p.new_team_eff / 2
    return dict(mu=last.mu + last.O.mean() + last.D.mean(), hca=last.hca, hca_nc=last.hca_nc, O=O, D=D,
                mu_t=last.mu_t + 2 * last.T.mean(), T=T)


def season_conf(gs):
    c = pd.concat([gs[["home_id", "home_conf"]].set_axis(["t", "c"], axis=1),
                   gs[["away_id", "away_conf"]].set_axis(["t", "c"], axis=1)]).dropna()
    return c.drop_duplicates("t", keep="last").set_index("t").c


def season_teams(games_season):
    d1 = games_season[games_season.d1]
    return np.unique(np.r_[d1.home_id.to_numpy(), d1.away_id.to_numpy()])


def walk_forward(games, seasons, p: Params, extra_teams=None, keep_final=False, roster=None, start=None):
    """Daily out-of-sample raw predictions for every D-I vs D-I game in `seasons`.

    games must include the season before seasons[0] (used only to build the first prior).
    Returns (pred DataFrame, dict season -> final Ratings)
    """
    out, finals = [], {}
    last = None
    first = seasons[0]
    all_seasons = sorted(set(games.season.unique()) & set(range(start or first - 2, max(seasons) + 1)))
    for s in all_seasons:
        gs = games[(games.season == s) & games.d1]
        teams = season_teams(gs)
        if extra_teams is not None and s in extra_teams:
            teams = np.union1d(teams, extra_teams[s])
        sm = SeasonModel(gs, teams, make_prior(last, teams, p, conf=season_conf(gs)), p)
        if roster is not None:
            roster.fit_for(s)
        ov = None
        if s in seasons:
            for d, gd in gs.groupby("date"):
                ov = roster.prior_at(s, d, last, teams) if roster is not None else None
                r = sm.solve(d, ov)
                poss, ph, pa = predict(r, sm.idx, gd.home_id, gd.away_id, gd.neutral, gd.conf_game)
                gh = np.array([r.n_games[sm.idx[t]] for t in gd.home_id])
                ga = np.array([r.n_games[sm.idx[t]] for t in gd.away_id])
                out.append(pd.DataFrame({
                    "game_id": gd.game_id.to_numpy(), "raw_poss": poss, "raw_home": ph, "raw_away": pa,
                    "gp_home": gh, "gp_away": ga}))
        end = gs.date.max() + pd.Timedelta(days=1)
        if roster is not None:
            ov = roster.prior_at(s, end, last, teams)
        final = sm.solve(end, ov)
        if roster is not None:
            roster.add_training_season(s, last, final)
        last = final
        finals[s] = (last, sm) if keep_final else last
    pred = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    pred = pred.merge(games, on="game_id")
    pred["raw_margin"] = pred.raw_home - pred.raw_away
    pred["raw_total"] = pred.raw_home + pred.raw_away
    pred["margin"] = pred.home_pts - pred.away_pts
    pred["total"] = pred.home_pts + pred.away_pts
    return pred, finals
