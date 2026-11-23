"""EXP-008: RosterPrior's optional conference-average feature.

Synthetic data only (no data/*.parquet needed): checks the option is off by default, that off reproduces
the original features, that the "conf" feature is the current-season conference mean of last season's
centered ratings, and that the fit for season s only uses training seasons < s.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cbb"))
import roster as R  # noqa: E402
from model import Ratings  # noqa: E402

TEAMS = np.arange(1, 9)
CONF = {t: ("A" if t <= 4 else "B") for t in TEAMS}
SEASONS = range(2012, 2018)


def _ratings(rng):
    n = len(TEAMS)
    return Ratings(teams=TEAMS.copy(), mu=103.0, hca=1.4, O=rng.normal(0, 5, n), D=rng.normal(0, 5, n),
                   mu_t=68.0, T=rng.normal(0, 2, n), n_games=np.full(n, 30))


def _comp(season, rng, n_dates=10):
    """Per-date composition rows: returners, one transfer from another team, and newcomers per team."""
    d0 = pd.Timestamp(f"{season - 1}-11-10")
    rows = []
    for i in range(n_dates):
        for t in TEAMS:
            w = rng.dirichlet([3, 1, 1])
            src_trn = TEAMS[(t % len(TEAMS))]            # the next team over
            for src, sh in zip((t, src_trn, -1), w):
                rows.append(dict(team_id=t, src=src, share=sh, share_pct=sh * rng.uniform(0.3, 1.0),
                                 date=d0 + pd.Timedelta(days=i)))
    return pd.DataFrame(rows)


def _games(seasons):
    rows = []
    for s in seasons:
        for h, a in [(1, 2), (3, 4), (5, 6), (7, 8), (1, 5)]:
            rows.append(dict(season=s, d1=True, home_id=h, away_id=a, home_conf=CONF[h], away_conf=CONF[a]))
    return pd.DataFrame(rows)


@pytest.fixture
def world():
    rng = np.random.default_rng(8)
    comps = {s: _comp(s, rng) for s in SEASONS}
    ratings = {s: _ratings(rng) for s in range(2011, 2018)}
    return comps, ratings, _games(SEASONS)


def _prior(world, **kw):
    comps, _, games = world
    r = R.RosterPrior(games=games, **kw)
    r.comp_cache.update(comps)
    return r


def test_conf_feature_off_by_default(monkeypatch):
    monkeypatch.delenv(R.CONF_ENV, raising=False)
    r = R.RosterPrior()
    assert r.conf_feature is False and r.feats == R.FEATS
    monkeypatch.setenv(R.CONF_ENV, "0")
    assert R.RosterPrior().conf_feature is False
    monkeypatch.setenv(R.CONF_ENV, "1")
    assert R.RosterPrior().conf_feature is True and R.RosterPrior().feats == R.FEATS + ["conf"]
    assert R.RosterPrior(conf_feature=False).conf_feature is False     # explicit argument wins


def test_off_reproduces_original_features(world):
    comps, ratings, _ = world
    cd = comps[2014][comps[2014].date == comps[2014].date.min()]
    got, want = _prior(world, conf_feature=False)._features(2014, cd, ratings[2013]), R.features(cd, ratings[2013])
    for k in "ODT":
        pd.testing.assert_frame_equal(got[k], want[k])


def test_conf_feature_is_conference_mean_of_last_season(world):
    comps, ratings, _ = world
    last = ratings[2013]
    cd = comps[2014][comps[2014].date == comps[2014].date.min()]
    f = _prior(world, conf_feature=True)._features(2014, cd, last)
    for k in "OD":
        v = pd.Series(getattr(last, k) - getattr(last, k).mean(), index=last.teams)
        want = v.groupby(pd.Series(CONF)).transform("mean")
        assert list(f[k].columns) == R.FEATS + ["conf"]
        np.testing.assert_allclose(f[k]["conf"].to_numpy(), want.reindex(f[k].index).to_numpy())
        pd.testing.assert_frame_equal(f[k][R.FEATS], R.features(cd, last)[k])


def test_fit_uses_only_earlier_seasons(world):
    _, ratings, _ = world
    r = _prior(world, conf_feature=True)
    for s in SEASONS:                                   # every season stored, including s and later
        r.add_training_season(s, ratings[s - 1], ratings[s])
    beta = r.fit_for(2015)
    assert r.fit_seasons[2015] == [2012, 2013, 2014]
    assert all(len(beta[k]) == len(R.FEATS) + 2 for k in "ODT")    # 7 features + intercept

    # identical fit from a prior that never saw 2015+: later seasons cannot leak into the 2015 coefficients
    r2 = _prior(world, conf_feature=True)
    for s in (2012, 2013, 2014):
        r2.add_training_season(s, ratings[s - 1], ratings[s])
    beta2 = r2.fit_for(2015)
    for k in "ODT":
        np.testing.assert_allclose(beta[k], beta2[k])

    # too little history -> no fit
    assert r.fit_for(2013) is None and 2013 not in r.fit_seasons


def test_prior_at_uses_latest_snapshot_before_date(world):
    comps, ratings, _ = world
    r = _prior(world, conf_feature=True)
    for s in (2012, 2013, 2014):
        r.add_training_season(s, ratings[s - 1], ratings[s])
    r.fit_for(2015)
    c = comps[2015]
    dates = sorted(c.date.unique())
    gap = dates[3] + pd.Timedelta(hours=12)              # no snapshot on this date -> use dates[3], not later
    got = r.prior_at(2015, gap, ratings[2014], TEAMS)
    f = r._features(2015, c[c.date == dates[3]], ratings[2014])
    for k in "ODT":
        x = f[k].dropna()
        np.testing.assert_allclose(got[k].to_numpy(), x.to_numpy() @ r.coef[2015][k][:-1] + r.coef[2015][k][-1])
    assert r.prior_at(2015, dates[0] - pd.Timedelta(days=1), ratings[2014], TEAMS) is None
