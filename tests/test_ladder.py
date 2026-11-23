"""Offline tests for cbb/ladder.py on small synthetic boards (no network, no data files)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cbb"))
import numpy as np
import pandas as pd
import pytest
import ladder
from venues import american_to_prob

CTX = dict(day=40, neutral=False)


def test_pmf_is_a_distribution_and_spreads_never_tie():
    for c in (-12.3, 0.0, 3.5, 25.0):
        n, p = ladder.make_dist("spread", c, total=145, **CTX).pmf()
        assert abs(p.sum() - 1) < 1e-12 and (p >= 0).all()
        assert p[n == 0].sum() == 0                     # a basketball game can't end tied
    n, p = ladder.make_dist("total", 150, abs_spread=6, **CTX).pmf()
    assert abs(p.sum() - 1) < 1e-12


@pytest.mark.parametrize("c", [-9.0, -2.5, 0.0, 4.0, 17.5])
@pytest.mark.parametrize("h", [-20.5, -6.5, -1.5, 1.5, 6.5, 20.5])
def test_complementary_rungs_sum_to_one(c, h):
    d = ladder.make_dist("spread", c, total=145, **CTX)
    # home at handicap h and away at -h are the two sides of one Kalshi market
    assert abs(ladder.fair_prob("spread", "home", h, dist=d) + ladder.fair_prob("spread", "away", -h, dist=d) - 1) < 1e-12
    dt = ladder.make_dist("total", 148.0, abs_spread=abs(c), **CTX)
    line = 148.0 + h
    assert abs(ladder.fair_prob("total", "over", line, dist=dt) + ladder.fair_prob("total", "under", line, dist=dt) - 1) < 1e-12


@pytest.mark.parametrize("c", [1.5, 6.0, 14.5])
@pytest.mark.parametrize("h", [-25.5, -8.5, -0.5, 3.5, 12.5])
def test_mirror_symmetry_home_away(c, h):
    # the same game seen from the other bench: home favored by c == away favored by c
    p_home = ladder.fair_prob("spread", "home", h, center=c, total=145, **CTX)
    p_away = ladder.fair_prob("spread", "away", h, center=-c, total=145, **CTX)
    assert abs(p_home - p_away) < 1e-9


def test_center_rungs_are_near_even_and_pair_sums():
    # total centered on a whole number: Over/Under the half points on either side of it
    c = 150.0
    d = ladder.make_dist("total", c, abs_spread=5, **CTX)
    over_lo = ladder.fair_prob("total", "over", c - 0.5, dist=d)
    under_hi = ladder.fair_prob("total", "under", c + 0.5, dist=d)
    assert 0.5 < over_lo < 0.53 and 0.5 < under_hi < 0.53
    p_eq = d.pmf()[1][d.pmf()[0] == c].sum()                # both win if the total lands exactly on c
    assert abs(over_lo + under_hi - 1 - p_eq) < 1e-12 and 0 < p_eq < 0.03
    # spread: the center is defined so that a 50/50 book line at -6.5 prices -6.5 at 0.5
    cs = ladder.implied_center("spread", "home", -6.5, 0.5, total=145, **CTX)
    assert abs(cs - 6.5) < 1.0
    assert abs(ladder.fair_prob("spread", "home", -6.5, center=cs, total=145, **CTX) - 0.5) < 1e-3
    # pick'em game: 'wins by over 0.5' on each side covers every outcome (no ties)
    d0 = ladder.make_dist("spread", 0.0, total=145, **CTX)
    assert abs(ladder.fair_prob("spread", "home", -0.5, dist=d0) + ladder.fair_prob("spread", "away", -0.5, dist=d0) - 1) < 1e-12


def test_monotone_ladder():
    d = ladder.make_dist("spread", 9.0, total=150, **CTX)
    hs = np.arange(-30.5, 31, 1.0)
    ph = [ladder.fair_prob("spread", "home", h, dist=d) for h in hs]   # more points given -> more likely to cover
    assert all(np.diff(ph) >= 0) and ph[0] < 0.05 and ph[-1] > 0.95
    dt = ladder.make_dist("total", 145.0, abs_spread=9, **CTX)
    po = [ladder.fair_prob("total", "over", L, dist=dt) for L in np.arange(110.5, 181, 1.0)]
    assert all(np.diff(po) <= 0)


def test_total_split_tail_thinner_below():
    # deep unders are rarer than deep overs at the same distance from the line
    d = ladder.make_dist("total", 145.0, abs_spread=5, **CTX)
    assert ladder.fair_prob("total", "under", 125.5, dist=d) < ladder.fair_prob("total", "over", 164.5, dist=d)


def _book_rows(gid, home_line, p_home_am, p_away_am, total, p_over_am, p_under_am, books=ladder.BOOKS):
    rows = []
    for b in books:
        rows += [dict(venue=b, market="spread", side="home", line=home_line, ask=american_to_prob(p_home_am)),
                 dict(venue=b, market="spread", side="away", line=-home_line, ask=american_to_prob(p_away_am)),
                 dict(venue=b, market="total", side="over", line=total, ask=american_to_prob(p_over_am)),
                 dict(venue=b, market="total", side="under", line=total, ask=american_to_prob(p_under_am))]
    return [dict(r, game_id=gid, fee_coef=0.0, bid=np.nan, ask_size=np.nan) for r in rows]


def _board(rows, tip="2026-11-23 20:00:00-05:00"):
    b = pd.DataFrame(rows)
    b["date"] = "2026-11-23"; b["tipoff"] = tip; b["away"] = "Away U"; b["home"] = "Home St"
    b["pick"] = np.where(b.market == "total", b.side.str.title() + " " + b.line.astype(str),
                         np.where(b.side == "home", "Home St", "Away U") + " " + b.line.map(lambda x: f"{x:+g}"))
    b["market_id"] = "x"
    return b


def test_devig_and_off_even_price_moves_center():
    even = _board(_book_rows(1, -6.5, -110, -110, 150.5, -110, -110))
    juiced = _board(_book_rows(1, -6.5, -130, +110, 150.5, +105, -125))
    fe, fj = ladder.fair_center(even, 1, **CTX), ladder.fair_center(juiced, 1, **CTX)
    assert fe["n_books_spread"] == 3 and fe["n_books_total"] == 3
    assert abs(fe["total"] - 150.5) < 0.05
    assert fj["spread"] > fe["spread"] + 0.5           # home -6.5 at -130: home is favored by more than 6.5
    assert fj["total"] < fe["total"] - 0.3             # under juiced: center below 150.5
    # the center reproduces the de-vigged price at the book's line
    q = american_to_prob(-130) / (american_to_prob(-130) + american_to_prob(110))
    p = ladder.fair_prob("spread", "home", -6.5, center=fj["spread"], total=fj["total"], **CTX)
    assert abs(p - q) < 1e-3


def test_consensus_is_median_of_books():
    rows = (_book_rows(1, -5.5, -110, -110, 150.5, -110, -110, books=["DraftKings Sportsbook"])
            + _book_rows(1, -6.5, -110, -110, 151.5, -110, -110, books=["FanDuel Sportsbook"])
            + _book_rows(1, -9.5, -110, -110, 160.5, -110, -110, books=["BetMGM"]))
    fc = ladder.fair_center(_board(rows), 1, **CTX)
    mid = ladder.fair_center(_board(_book_rows(1, -6.5, -110, -110, 151.5, -110, -110)), 1, **CTX)
    assert abs(fc["spread"] - mid["spread"]) < 1e-6 and abs(fc["total"] - mid["total"]) < 1e-6


def test_price_ladder_edge_math_fee_and_tipoff_filter():
    rows = _book_rows(1, -6.5, -110, -110, 150.5, -110, -110)
    kal = [dict(venue="Kalshi", market="spread", side="home", line=-6.5, ask=0.40, bid=0.39, ask_size=100.0),
           dict(venue="Kalshi", market="spread", side="away", line=6.5, ask=0.61, bid=0.60, ask_size=np.nan),
           dict(venue="Kalshi", market="total", side="over", line=160.5, ask=0.30, bid=0.28, ask_size=50.0),
           dict(venue="Kalshi", market="total", side="under", line=160.5, ask=1.00, bid=0.70, ask_size=np.nan)]
    rows += [dict(r, game_id=1, fee_coef=0.07) for r in kal]
    rh = dict(kal[0], venue="Robinhood", game_id=1, fee_coef=0.02 / (0.40 * 0.60))
    rows.append(rh)
    early = [dict(r, game_id=2) for r in _book_rows(2, -3.5, -110, -110, 140.5, -110, -110)]
    early.append(dict(venue="Kalshi", market="spread", side="home", line=-3.5, ask=0.30, bid=0.29, ask_size=10.0,
                      game_id=2, fee_coef=0.07))
    b = pd.concat([_board(rows), _board(early, tip="2026-11-23 12:00:00-05:00")], ignore_index=True)
    lad = ladder.price_ladder(b, now="2026-11-23T18:12Z")
    assert set(lad.game_id) == {1}                     # game 2 tipped at 17:00Z, before the run
    assert not ((lad.market == "total") & (lad.side == "under")).any()   # ask 1.00 = nothing offered
    k = lad[(lad.venue == "Kalshi") & (lad.rung == "Home St -6.5")].iloc[0]
    fee = 0.07 * 0.40 * 0.60
    assert abs(k.fee - round(fee, 4)) < 1e-9 and abs(k.cost - round(0.40 + fee, 4)) < 1e-9
    assert abs(k.fair - 0.5) < 2e-3
    assert abs(k.edge - (k.fair - 0.40 - fee)) < 1e-3 and abs(k.edge_c - 100 * k.edge) < 0.02
    assert abs(k.usd_avail - 40.0) < 1e-9             # 100 contracts at 40c
    a = lad[(lad.venue == "Kalshi") & (lad.side == "away")].iloc[0]
    assert abs(a.fair + k.fair - 1) < 2e-3 and np.isnan(a.usd_avail)
    r = lad[lad.venue == "Robinhood"].iloc[0]
    assert abs(r.fee - 0.02) < 1e-9 and r.edge < k.edge          # same book, flat 2c > Kalshi's 1.68c
    assert k.stretch == pytest.approx(6.5 - k.center, abs=0.01)
