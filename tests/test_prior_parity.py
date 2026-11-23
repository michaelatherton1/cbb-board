"""The live model (daily.py) must use the same preseason prior as the backtest (model.walk_forward).

Replays daily's production path (daily.season_model, no downloads) on past dates of a finished season and
checks its raw margins against walk_forward's out-of-sample raw margins for the same games. Needs the
downloaded data/*.parquet files; skipped on a fresh clone without them.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cbb"))
import prep  # noqa: E402

SEASON = 2020
START = max(2010, SEASON - 15)          # daily.main's prior chain; the backtest's chain also starts in 2010
DAYS = (0, 3, 7, 14, 30)
NEEDED = [prep.DATA / "games.parquet"] + [prep.DATA / f"{kind}_{y}.parquet"
                                          for y in range(START, SEASON + 1) for kind in ("team_box", "sched")]


@pytest.fixture(scope="module")
def replay():
    missing = [f.name for f in NEEDED if not f.exists()]
    if missing:
        pytest.skip(f"needs downloaded data: {len(missing)} data/*.parquet files missing (e.g. {missing[0]})")
    import backtest
    from model import walk_forward
    p = backtest.load_params()
    # backtest side: walk_forward on data/games.parquet, as backtest.run does
    ref_games = pd.read_parquet(prep.DATA / "games.parquet")
    ref, _ = walk_forward(ref_games[ref_games.season <= SEASON], [SEASON], p, start=START)
    # production side: daily.main's inputs, built from the raw season files
    games = prep.load_all(range(START, SEASON + 1))
    _, finals = walk_forward(games[games.season < SEASON], [SEASON - 1], p, start=START)
    return dict(p=p, ref=ref.set_index("game_id"), games=games, last=finals[SEASON - 1], conf=prep.conf_map(SEASON))


@pytest.mark.parametrize("day", DAYS)
def test_daily_prior_matches_walk_forward(replay, day):
    import daily
    from model import predict
    games, p = replay["games"], replay["p"]
    gs = games[(games.season == SEASON) & games.d1]
    asof = gs.date.min() + pd.Timedelta(days=day)
    up = gs[gs.date == asof]                     # that day's games stand in for daily's upcoming slate
    assert len(up), f"no D-I games on day {day} of {SEASON}"
    now, sm = daily.season_model(games, SEASON, asof.date(), up, replay["last"], p, replay["conf"])
    assert (now.date < asof).all()
    r = sm.solve(asof)
    _, ph, pa = predict(r, sm.idx, up.home_id, up.away_id, up.neutral, up.conf_game)
    want = replay["ref"].loc[up.game_id.to_numpy(), "raw_margin"].to_numpy()
    np.testing.assert_allclose(ph - pa, want, rtol=0, atol=1e-9)
    want_t = replay["ref"].loc[up.game_id.to_numpy(), "raw_total"].to_numpy()   # totals too (EXP-006 lam_style)
    np.testing.assert_allclose(ph + pa, want_t, rtol=0, atol=1e-9)
