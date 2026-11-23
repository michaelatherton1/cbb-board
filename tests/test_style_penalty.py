"""EXP-006: separate ridge penalty on the 'style' direction (O+D)/2 of the O/D ratings.

- lam_style None / == lam_eff reproduces the plain diagonal ridge exactly (synthetic league).
- Any lam_style solves the block-penalised objective (checked against an independent augmented least squares).
- Margins do not depend on lam_style: synthetic league, and a real-season walk_forward (needs local data).
"""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cbb"))
import model  # noqa: E402
from model import Params, SeasonModel, predict  # noqa: E402


def synthetic_league(n_teams=12, n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    teams = np.arange(100, 100 + n_teams)
    q = rng.normal(0, 6, n_teams)
    s = rng.normal(0, 3, n_teams)
    t = rng.normal(0, 3, n_teams)
    rows = []
    d0 = pd.Timestamp("2019-11-05")
    gid = 0
    for day in range(n_days):
        order = rng.permutation(n_teams)
        for k in range(0, n_teams - 1, 2):
            h, a = order[k], order[k + 1]
            neutral = rng.random() < 0.2
            poss = 68 + t[h] + t[a] + rng.normal(0, 4)
            loc = 0.0 if neutral else 1.0
            eh = 103 + (q[h] + s[h]) + (s[a] - q[a]) + 1.4 * loc + rng.normal(0, 10)
            ea = 103 + (q[a] + s[a]) + (s[h] - q[h]) - 1.4 * loc + rng.normal(0, 10)
            rows.append(dict(game_id=gid, date=d0 + pd.Timedelta(days=day), home_id=teams[h], away_id=teams[a],
                             neutral=neutral, conf_game=bool(rng.random() < 0.5), poss=poss,
                             home_pts=eh * poss / 100, away_pts=ea * poss / 100))
            gid += 1
    games = pd.DataFrame(rows)
    prior = dict(mu=102.0, hca=1.4, hca_nc=0.0, O=rng.normal(0, 3, n_teams), D=rng.normal(0, 3, n_teams),
                 mu_t=67.0, T=rng.normal(0, 2, n_teams))
    return games, teams, prior


def old_solve(sm, date, override=None):
    """The pre-EXP-006 efficiency solve: plain diagonal ridge with lam_eff on every O and D."""
    p, n = sm.p, sm.n
    b0 = sm.b0.copy()
    if override:
        for comp, ser in override.items():
            ii = np.array([sm.idx[t] for t in ser.index], dtype=int)
            if comp == "O":
                b0[3 + ii] = ser.to_numpy()
            elif comp == "D":
                b0[3 + n + ii] = ser.to_numpy()
    pen = np.r_[p.lam_mu, p.lam_hca, p.lam_hca, np.full(n, p.lam_eff), np.full(n, p.lam_eff)]
    cut = (pd.Timestamp(date) - sm.d0).days
    sel = sm.row_day < cut
    X, y = sm.X[sel], sm.y[sel]
    w = 0.5 ** ((cut - sm.row_day[sel]) / p.half_life)
    Xw = X.multiply(w[:, None]).tocsr()
    A = (X.T @ Xw).toarray()
    A[np.diag_indices_from(A)] += pen
    return cho_solve(cho_factor(A), Xw.T @ y + pen * b0)


def override_for(teams):
    ix = pd.Index(teams[:5])
    return {"O": pd.Series(np.linspace(-4, 4, 5), index=ix), "D": pd.Series(np.linspace(3, -3, 5), index=ix)}


BASE = Params(half_life=30.0, lam_eff=4.0, lam_tempo=2.0, lam_hca=500.0, lam_mu=50.0)
DATES = [pd.Timestamp("2019-11-05"), pd.Timestamp("2019-11-09"), pd.Timestamp("2019-11-25"),
         pd.Timestamp("2019-12-20")]


@pytest.mark.parametrize("lam_style", [None, 4.0])
@pytest.mark.parametrize("use_override", [False, True])
def test_default_reproduces_plain_ridge_exactly(lam_style, use_override):
    games, teams, prior = synthetic_league()
    sm = SeasonModel(games, teams, prior, replace(BASE, lam_style=lam_style))
    assert sm.couple is None
    ov = override_for(teams) if use_override else None
    for d in DATES[1:]:
        r = sm.solve(d, ov)
        b = old_solve(sm, d, ov)
        n = sm.n
        assert np.array_equal(np.r_[r.mu, r.hca, r.hca_nc, r.O, r.D], np.r_[b[:3], b[3:3 + n], b[3 + n:]])


@pytest.mark.parametrize("lam_style", [1.0, 12.0, 30.0])
@pytest.mark.parametrize("use_override", [False, True])
def test_block_penalty_matches_augmented_least_squares(lam_style, use_override):
    """lam_eff * sum ((dO - dD)^2)/2 + lam_style * sum ((dO + dD)^2)/2, d = deviation from the prior center."""
    games, teams, prior = synthetic_league()
    p = replace(BASE, lam_style=lam_style)
    sm = SeasonModel(games, teams, prior, p)
    ov = override_for(teams) if use_override else None
    d = DATES[2]
    r = sm.solve(d, ov)
    n = sm.n
    b0 = sm.b0.copy()
    if ov:
        ii = np.array([sm.idx[t] for t in ov["O"].index])
        b0[3 + ii], b0[3 + n + ii] = ov["O"].to_numpy(), ov["D"].to_numpy()
    cut = (d - sm.d0).days
    sel = sm.row_day < cut
    sw = np.sqrt(0.5 ** ((cut - sm.row_day[sel]) / p.half_life))
    k = 3 + 2 * n
    rows, rhs = [sm.X[sel].toarray() * sw[:, None]], [sm.y[sel] * sw]
    for j, lam in enumerate([p.lam_mu, p.lam_hca, p.lam_hca]):
        e = np.zeros(k)
        e[j] = np.sqrt(lam)
        rows.append(e[None])
        rhs.append([np.sqrt(lam) * b0[j]])
    for i in range(n):
        for lam, sgn in [(p.lam_eff, -1.0), (lam_style, 1.0)]:
            e = np.zeros(k)
            e[3 + i], e[3 + n + i] = 1.0, sgn
            e *= np.sqrt(lam / 2)
            rows.append(e[None])
            rhs.append([e @ b0])
    want = np.linalg.lstsq(np.vstack(rows), np.concatenate([np.ravel(x) for x in rhs]), rcond=None)[0]
    got = np.r_[r.mu, r.hca, r.hca_nc, r.O, r.D]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-8)


def test_no_games_returns_prior_center():
    games, teams, prior = synthetic_league()
    sm = SeasonModel(games, teams, prior, replace(BASE, lam_style=12.0))
    r = sm.solve(DATES[0])
    np.testing.assert_array_equal(r.O, prior["O"])
    np.testing.assert_array_equal(r.D, prior["D"])


@pytest.mark.parametrize("use_override", [False, True])
def test_synthetic_margins_do_not_depend_on_lam_style(use_override):
    games, teams, prior = synthetic_league()
    ov = override_for(teams) if use_override else None
    up = games[games.date >= DATES[2]].head(30)
    out = {}
    for ls in (None, 4.0, 12.0, 30.0):
        sm = SeasonModel(games, teams, prior, replace(BASE, lam_style=ls))
        r = sm.solve(DATES[2], ov)
        _, ph, pa = predict(r, sm.idx, up.home_id, up.away_id, up.neutral, up.conf_game)
        out[ls] = (ph - pa, ph + pa, r.O - r.D)
    for ls in (4.0, 12.0, 30.0):
        np.testing.assert_allclose(out[ls][0], out[None][0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(out[ls][2], out[None][2], rtol=0, atol=1e-9)
    assert np.abs(out[30.0][1] - out[None][1]).max() > 1e-3


# ---- real season (needs downloaded data) -------------------------------------------------------------------
DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def real_wf():
    f = DATA / "games.parquet"
    if not f.exists():
        pytest.skip("needs downloaded data: data/games.parquet missing")
    import backtest
    games = pd.read_parquet(f)
    games = games[(games.season >= 2017) & (games.season <= 2019)]
    p = backtest.load_params()
    out = {}
    for ls in (4, 12, 30):
        pred, _ = model.walk_forward(games, [2019], replace(p, lam_style=ls), start=2017)
        out[ls] = pred.set_index("game_id").sort_index()
    return out


def test_real_season_margins_do_not_depend_on_lam_style(real_wf):
    ref = real_wf[4]
    assert len(ref) > 4000
    for ls in (12, 30):
        got = real_wf[ls]
        assert got.index.equals(ref.index)
        np.testing.assert_allclose(got.raw_margin.to_numpy(), ref.raw_margin.to_numpy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(got.raw_poss.to_numpy(), ref.raw_poss.to_numpy(), rtol=0, atol=1e-9)
        assert np.abs(got.raw_total - ref.raw_total).mean() > 0.05
