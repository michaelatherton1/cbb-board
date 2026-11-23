"""Price Kalshi's spread and total ladders off the sportsbook consensus (EXP-005).

Fair value here comes from the sportsbooks (DraftKings, FanDuel, BetMGM via TheRundown), not from the model:

1. fair_center(board, game_id): each book's main spread and main total, with the margin removed
   (multiplicative de-vig of its two-way price), turned into an implied center with the distribution below
   (a main line priced off 50/50 moves the center), then the median across books.
2. The distribution around the center, fitted on 2018-2020 closing lines; model chosen by leave-one-season-out
   over 2018-2020 on half-point rungs (scripts: experiments/exp005/part_a*.py):
     spread  integer home margin. Split normal from the favorite's side: one sigma for "favorite falls short of
             the line", another for "favorite beats it", each log-linear in the total, |spread|, days into the
             season and neutral site; times key-number weights on |margin| (a game can't end tied, 1-point
             margins are rarer and 2-7 point margins more common than a Normal says). |spread| in the sigma
             features is capped at 30 (few games beyond).
     total   split normal around the line, sigma log-linear in the same features; x0.934 below the line and
             x1.019 above it (big unders are rarer than a Normal says; overtime fattens the high side).
3. price_ladder(board): every Kalshi spread/total rung (and Robinhood's, which is Kalshi's book plus a flat 2c
   fee): fair probability, cost after the taker fee, edge = fair - cost.

    python cbb/ladder.py data/out/board_2026-11-23.csv --now 2026-11-23T18:12Z
"""
import argparse, math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

BOOKS = ("DraftKings Sportsbook", "FanDuel Sportsbook", "BetMGM")
LADDER_VENUES = ("Kalshi", "Robinhood")

# Spread shape, fitted on 2018-2020 closes with the center defined as in production (the closing line is a
# 50/50 proposition), iterating centers <-> shape. Favorite's perspective.
# features: [1, (total-140)/10, |spread|/10, (|spread|/10)^2, exp(-day/21), neutral]
SPREAD_THETA_FAV_SHORT = np.array([2.4397, 0.0115, -0.1216, 0.0352, 0.1212, 0.0889])   # favorite under the line
SPREAD_THETA_FAV_OVER = np.array([2.3639, 0.0200, -0.0009, 0.0286, 0.0646, -0.0279])   # favorite over the line
# weight on P(|margin| = 0, 1, ..., 12); 1 beyond
KEY_W = np.array([0.0, 0.8565, 1.0927, 1.1861, 1.1862, 1.1293, 1.1058, 1.1020, 1.0648, 1.2099, 1.1380, 1.0436,
                  1.1108])
# features: [1, (total-140)/10, |spread|/10, exp(-day/21), neutral]
TOTAL_THETA = np.array([2.8407, 0.0391, -0.0449, 0.0648, -0.0066])
TOTAL_K_UNDER, TOTAL_K_OVER = 0.9339, 1.0191
EARLY_TAU = 21.0
DEFAULT_TOTAL, DEFAULT_ABS_SPREAD = 141.0, 8.0
MAX_ABS_SPREAD_FEATURE = 30.0   # ~100 games past 30 in 2018-2020: don't let the |spread|^2 term extrapolate


def season_day(d):
    """Days since opening day, taken as the first Monday of November (2026-11-02 for the 2027 season)."""
    d = pd.Timestamp(d).date()
    y = d.year if d.month >= 7 else d.year - 1
    nov1 = date(y, 11, 1)
    opener = nov1 + timedelta(days=(0 - nov1.weekday()) % 7)
    return max((d - opener).days, 0)


def spread_sigmas(center, total=None, day=30, neutral=False):
    """(sigma below the center, sigma above the center) for the home margin."""
    T = DEFAULT_TOTAL if total is None or not np.isfinite(total) else total
    a = min(abs(center), MAX_ABS_SPREAD_FEATURE) / 10
    x = np.array([1.0, (T - 140) / 10, a, a * a, math.exp(-day / EARLY_TAU), float(bool(neutral))])
    short, over = math.exp(x @ SPREAD_THETA_FAV_SHORT), math.exp(x @ SPREAD_THETA_FAV_OVER)
    # home favored (center >= 0): below the center = favorite falls short. Away favored: mirrored.
    return (short, over) if center >= 0 else (over, short)


def total_sigmas(center, abs_spread=None, day=30, neutral=False):
    a = DEFAULT_ABS_SPREAD if abs_spread is None or not np.isfinite(abs_spread) else abs(abs_spread)
    a = min(a, MAX_ABS_SPREAD_FEATURE)
    x = np.array([1.0, (center - 140) / 10, a / 10, math.exp(-day / EARLY_TAU), float(bool(neutral))])
    s = math.exp(x @ TOTAL_THETA)
    return s * TOTAL_K_UNDER, s * TOTAL_K_OVER


@dataclass
class Dist:
    """Integer-valued result (home margin or total points) around a center: split normal, optional key weights."""
    market: str
    center: float
    sd_lo: float
    sd_hi: float
    key_w: np.ndarray = None
    _pmf: tuple = field(default=None, repr=False)

    def _cdf(self, x):
        x = np.asarray(x, float) - self.center
        return np.where(x < 0, norm.cdf(x / self.sd_lo), norm.cdf(x / self.sd_hi))

    def pmf(self):
        if self._pmf is None:
            w = 12 * max(self.sd_lo, self.sd_hi)
            n = np.arange(math.floor(self.center - w), math.ceil(self.center + w) + 1)
            p = self._cdf(n + 0.5) - self._cdf(n - 0.5)
            if self.key_w is not None:
                a = np.abs(n)
                p = p * np.where(a < len(self.key_w), self.key_w[np.minimum(a, len(self.key_w) - 1)], 1.0)
            self._pmf = (n, p / p.sum())
        return self._pmf

    def p_over(self, t):
        """P(result > t)."""
        n, p = self.pmf()
        return float(p[n > t].sum())

    def p_under(self, t):
        """P(result < t)."""
        n, p = self.pmf()
        return float(p[n < t].sum())


def make_dist(market, center, total=None, abs_spread=None, day=30, neutral=False):
    if market == "spread":
        lo, hi = spread_sigmas(center, total, day, neutral)
        return Dist("spread", float(center), lo, hi, KEY_W)
    if market == "total":
        lo, hi = total_sigmas(center, abs_spread, day, neutral)
        return Dist("total", float(center), lo, hi, None)
    raise ValueError(market)


def fair_prob(market, side, line, center=None, dist=None, **ctx):
    """Probability that the rung wins outright (a tie with a whole-number line counts as a loss, as on Kalshi).

    spread: side 'home'/'away', line = that team's handicap (-7.5 = must win by 8+); center = home margin.
    total:  side 'over'/'under', line = total points; center = expected (median) total.
    """
    if dist is None:
        dist = make_dist(market, center, **ctx)
    if market == "spread":
        if side == "home":
            return dist.p_over(-line)            # home margin + line > 0
        if side == "away":
            return dist.p_under(line)            # -home margin + line > 0
    elif market == "total":
        if side == "over":
            return dist.p_over(line)
        if side == "under":
            return dist.p_under(line)
    raise ValueError((market, side))


def _p_win_no_push(market, side, line, dist):
    """Win probability given no push: what a de-vigged sportsbook price means on a whole-number line."""
    win = fair_prob(market, side, line, dist=dist)
    other = {"home": "away", "away": "home", "over": "under", "under": "over"}[side]
    lose = fair_prob(market, other, -line if market == "spread" else line, dist=dist)
    return win / (win + lose) if win + lose > 0 else np.nan


def implied_center(market, side, line, p_win, **ctx):
    """Center at which this distribution gives the de-vigged probability p_win to (side, line)."""
    if not (0 < p_win < 1):
        return np.nan
    # the center that makes the rung a coin flip is about the line itself; search +-30 around it
    c0 = (-line if side == "home" else line) if market == "spread" else line

    def f(c):
        return _p_win_no_push(market, side, line, make_dist(market, c, **ctx)) - p_win

    try:
        return brentq(f, c0 - 30, c0 + 30, xtol=1e-4)
    except ValueError:
        return np.nan


def _book_pairs(g, market):
    """Per book: (line for side A, de-vigged P(side A)), side A = home (spread) or over (total)."""
    a_side, b_side = ("home", "away") if market == "spread" else ("over", "under")
    out = []
    for book, x in g[g.market == market].groupby("venue"):
        a, b = x[x.side == a_side], x[x.side == b_side]
        if len(a) != 1 or len(b) != 1:
            continue
        la, lb = float(a.line.iloc[0]), float(b.line.iloc[0])
        consistent = abs(la + lb) < 1e-6 if market == "spread" else abs(la - lb) < 1e-6
        pa, pb = float(a.ask.iloc[0]), float(b.ask.iloc[0])
        if not consistent or not (0 < pa < 1 and 0 < pb < 1):
            continue
        out.append(dict(book=book, line=la, p_raw=pa, p_raw_other=pb, vig=pa + pb - 1, p=pa / (pa + pb)))
    return out


def fair_center(board, game_id, day=30, neutral=False):
    """Book-consensus centers for one game.

    Each book's main line comes with a two-way price; the margin is removed multiplicatively
    (p_home / (p_home + p_away)). Near even money - where main lines sit - the multiplicative, power and
    additive methods agree to ~0.1c, and multiplicative is the simplest to audit. The de-vigged probability at
    the book's line is converted into the center that gives that probability under the fitted distribution, so
    -6.5 at -130/+110 counts as a bit more than 6.5. Consensus = median across the books that have the market.
    """
    g = board[(board.game_id == game_id) & board.venue.isin(BOOKS)]
    sp, tt = _book_pairs(g, "spread"), _book_pairs(g, "total")
    # context for the sigmas: median raw lines (the other market's center barely moves sigma)
    T0 = float(np.median([b["line"] for b in tt])) if tt else None
    m0 = float(np.median([-b["line"] for b in sp])) if sp else None
    ctx = dict(day=day, neutral=neutral)
    for b in sp:
        b["center"] = implied_center("spread", "home", b["line"], b["p"], total=T0, **ctx)
    for b in tt:
        b["center"] = implied_center("total", "over", b["line"], b["p"], abs_spread=m0, **ctx)
    cs = [b["center"] for b in sp if np.isfinite(b["center"])]
    ct = [b["center"] for b in tt if np.isfinite(b["center"])]
    abbrev = lambda s: {"DraftKings Sportsbook": "DK", "FanDuel Sportsbook": "FD", "BetMGM": "MGM"}.get(s, s)
    return dict(
        spread=float(np.median(cs)) if cs else np.nan, total=float(np.median(ct)) if ct else np.nan,
        n_books_spread=len(cs), n_books_total=len(ct),
        spread_detail=" ".join(f"{abbrev(b['book'])} {b['line']:+g}@{b['p']:.3f}" for b in sp),
        total_detail=" ".join(f"{abbrev(b['book'])} {b['line']:g}@{b['p']:.3f}" for b in tt),
        book_spread_range=(max(cs) - min(cs)) if cs else np.nan,
        book_total_range=(max(ct) - min(ct)) if ct else np.nan)


def _stretch(market, side, line, center):
    """How far past the consensus the rung needs the game to go (>0: long shot side, <0: cushion side)."""
    if market == "spread":
        return (-line - center) if side == "home" else (center - line)
    return (line - center) if side == "over" else (center - line)


def venue_center(rows, market, dist_kw, c0):
    """Center that best fits a venue's ladder mids (least squares in probability), for comparing to the books."""
    r = rows[(rows.bid > 0) & (rows.ask < 1) & ((rows.ask - rows.bid) <= 0.05)]
    r = r[((r.ask + r.bid) / 2).between(0.05, 0.95)]
    if len(r) < 3 or not np.isfinite(c0):
        return np.nan
    mid = ((r.ask + r.bid) / 2).to_numpy()
    sides, lines = r.side.to_numpy(), r.line.to_numpy()

    def loss(c):
        d = make_dist(market, c, **dist_kw)
        return sum((fair_prob(market, s, l, dist=d) - m) ** 2 for s, l, m in zip(sides, lines, mid))

    res = minimize_scalar(loss, bounds=(c0 - 15, c0 + 15), method="bounded", options=dict(xatol=0.01))
    return float(res.x)


def price_ladder(board, now=None, venues=LADDER_VENUES, neutral=None):
    """One row per ladder rung on `venues` with a book-consensus fair probability.

    board   board_<date>.csv as a DataFrame (or a path)
    now     drop games whose tipoff is at or before this time (e.g. the board's run time)
    neutral optional {game_id: bool}; neutral-site games get a slightly wider spread distribution
    """
    b = pd.read_csv(board) if isinstance(board, (str, Path)) else board.copy()
    for c in ("line", "ask", "bid", "fee_coef", "ask_size"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["tip_utc"] = pd.to_datetime(b.tipoff, utc=True)
    if now is not None:
        now = pd.Timestamp(now)
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        b = b[b.tip_utc > now]
    neutral = neutral or {}
    lad = b[b.venue.isin(venues) & b.market.isin(["spread", "total"]) & (b.ask > 0) & (b.ask < 1)]
    rows = []
    for gid, g in lad.groupby("game_id"):
        day = season_day(g.date.iloc[0])
        neu = bool(neutral.get(gid, False))
        fc = fair_center(b, gid, day=day, neutral=neu)
        dists = {}
        if np.isfinite(fc["spread"]):
            dists["spread"] = make_dist("spread", fc["spread"], total=fc["total"] if np.isfinite(fc["total"]) else None,
                                        day=day, neutral=neu)
        if np.isfinite(fc["total"]):
            dists["total"] = make_dist("total", fc["total"],
                                       abs_spread=fc["spread"] if np.isfinite(fc["spread"]) else None,
                                       day=day, neutral=neu)
        for (venue, market), x in g.groupby(["venue", "market"]):
            if market not in dists:
                continue
            d, c = dists[market], fc[market]
            other = "total" if market == "spread" else "spread"
            kw = dict(day=day, neutral=neu)
            kw["total" if market == "spread" else "abs_spread"] = fc[other] if np.isfinite(fc[other]) else None
            vc = venue_center(x, market, kw, c) if venue == "Kalshi" else np.nan
            for r in x.itertuples(index=False):
                fair = fair_prob(market, r.side, r.line, dist=d)
                fee = float(r.fee_coef if pd.notna(r.fee_coef) else 0) * r.ask * (1 - r.ask)
                rows.append(dict(
                    date=r.date, tipoff=r.tipoff, game_id=gid, game=f"{r.away} @ {r.home}", venue=venue,
                    market=market, rung=r.pick, side=r.side, line=r.line, ask=r.ask, bid=r.bid,
                    ask_size=r.ask_size, usd_avail=r.ask_size * r.ask if pd.notna(r.ask_size) else np.nan,
                    fee=round(fee, 4), cost=round(r.ask + fee, 4), fair=round(fair, 4),
                    edge=round(fair - r.ask - fee, 4), edge_c=round(100 * (fair - r.ask - fee), 2),
                    stretch=round(_stretch(market, r.side, r.line, c), 2), center=round(c, 2),
                    venue_center=round(vc, 2) if np.isfinite(vc) else np.nan,
                    center_gap=round(vc - c, 2) if np.isfinite(vc) else np.nan,
                    n_books=fc[f"n_books_{market}"], book_range=round(fc[f"book_{market}_range"], 2),
                    books=fc[f"{market}_detail"], sd_lo=round(d.sd_lo, 2), sd_hi=round(d.sd_hi, 2),
                    market_id=r.market_id))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("edge", ascending=False).reset_index(drop=True)


def _neutral_map(board_path):
    p = Path(board_path)
    proj = p.with_name(p.name.replace("board_", "projections_"))
    if proj.exists():
        pr = pd.read_csv(proj)
        if "neutral" in pr:
            return dict(zip(pr.game_id, pr.neutral.astype(bool)))
    return {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("board")
    ap.add_argument("--now", help="board run time (UTC); games that tipped by then are dropped")
    ap.add_argument("--venues", nargs="*", default=list(LADDER_VENUES))
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--min-usd", type=float, default=10.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    lad = price_ladder(a.board, now=a.now, venues=a.venues, neutral=_neutral_map(a.board))
    if a.out:
        lad.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    k = lad[lad.venue == "Kalshi"]
    print(f"{len(k)} Kalshi rungs in {k.game_id.nunique()} games; edge after fee (cents):")
    print(k.edge_c.describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_string())
    c = k[(k.edge >= a.min_edge) & (k.usd_avail >= a.min_usd)]
    print(f"\nedge >= {100 * a.min_edge:.0f}c with >= ${a.min_usd:.0f} at the ask: {len(c)}")
    print(c[["game", "rung", "ask", "fee", "fair", "edge_c", "ask_size", "usd_avail", "stretch", "center",
             "venue_center", "n_books"]].to_string(index=False))
