"""Daily board: model projections next to live Kalshi / Polymarket US / Novig prices.

    python cbb/board.py --date 2026-11-12 --days 2          # fetch live prices (needs internet)
    python cbb/board.py --quotes data/out/quotes_X.csv       # re-score saved quotes

Outputs (data/out/): quotes_<date>.csv (raw, normalized), board_<date>.csv (every priced outcome with
model probability and edge), edges_<date>.csv (filtered), arbs_<date>.csv, board_<date>.html
"""
import argparse, json, sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm as N
sys.path.insert(0, str(Path(__file__).parent))
import prep
from names import team_aliases, match_games, best_score
import venues

DATA, OUT = prep.DATA, prep.DATA / "out"
MIN_EDGE = 0.03        # 3 cents per $1 contract after estimated fees
PRICE_RANGE = (0.05, 0.95)


def american(p):
    p = np.clip(np.asarray(p, float), 0.0099, 0.9901)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p).round(0)


def load_projections(asof, days):
    p = OUT / f"projections_{asof}.csv"
    if not p.exists():
        import daily
        daily.main(asof, days, refresh=True)
    proj = pd.read_csv(p)
    proj["date"] = pd.to_datetime(proj.date)
    return proj


MANUAL = DATA / "manual_quotes.csv"     # optional: prices typed in from apps without an API


def manual_quotes(path=MANUAL):
    """Rows: venue,date,away,home,market,side,line,ask[,fee_coef]  (side: away|home|over|under; ask in $ or cents)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=venues.COLS)
    m = pd.read_csv(path)
    if m.empty:
        return pd.DataFrame(columns=venues.COLS)
    m["ask"] = pd.to_numeric(m.ask, errors="coerce")
    m.loc[m.ask > 1, "ask"] = m.ask / 100
    out = pd.DataFrame({
        "venue": m.venue, "event_id": m.venue + "|" + m.away + "|" + m.home, "event_name": m.away + " at " + m.home,
        "date": pd.to_datetime(m.date).dt.date, "start_utc": None, "team1": m.away, "team2": m.home,
        "market": m.market.str.lower().replace({"moneyline": "ml"}),
        "side_team": np.where(m.side.str.lower() == "home", m.home, np.where(m.side.str.lower() == "away", m.away, None)),
        "side": np.where(m.side.str.lower().isin(["over", "under"]), m.side.str.lower(), None),
        "line": pd.to_numeric(m.get("line"), errors="coerce"), "ask": m.ask, "bid": None, "ask_size": None,
        "fee_coef": pd.to_numeric(m.get("fee_coef", 0), errors="coerce"), "market_id": None})
    return out[venues.COLS]


def match(quotes, proj, season):
    sched = pd.read_parquet(DATA / f"sched_{season}.parquet")
    aliases = team_aliases(sched)
    games = proj if {"home_id", "away_id"} <= set(proj) else proj.merge(
        prep.upcoming(season, proj.date.min().date(), proj.date.max().date())[["game_id", "home_id", "away_id"]],
        on="game_id")
    ev = quotes.assign(event_key=quotes.venue + "|" + quotes.event_id.astype(str))
    evs = ev.drop_duplicates("event_key")[["event_key", "date", "team1", "team2"]]
    m = match_games(evs, games[["game_id", "home_id", "away_id", "date"]], aliases)
    ev["game_id"] = ev.event_key.map(lambda k: m.get(k, (None,))[0])
    unmatched = ev[ev.game_id.isna()].drop_duplicates("event_key")
    ev = ev.dropna(subset=["game_id"]).copy()
    ev["game_id"] = ev.game_id.astype(int)
    ev = ev.merge(games[["game_id", "home_id", "away_id"]], on="game_id")

    # which side does each team-named outcome belong to?
    def side_of(r):
        if r.market == "total":
            return r.side
        sh = best_score(r.side_team, aliases.get(r.home_id, []))
        sa = best_score(r.side_team, aliases.get(r.away_id, []))
        if max(sh, sa) < 0.72 or abs(sh - sa) < 0.05:
            return None
        return "home" if sh > sa else "away"
    ev["side"] = ev.apply(side_of, axis=1)
    ev = ev.dropna(subset=["side"])
    # Kalshi lists each team twice (YES on its own market, NO on the opponent's): keep the cheaper ask
    ev["line_key"] = pd.to_numeric(ev.line, errors="coerce").fillna(-999)
    ev = (ev.sort_values("ask").drop_duplicates(["game_id", "venue", "market", "side", "line_key"])
          .drop(columns="line_key"))
    return ev, unmatched


def score(q, proj, cal):
    d = q.rename(columns={"date": "venue_date"}).merge(proj[["game_id", "date", "tipoff", "away", "home", "neutral", "home_spread", "total",
                      "home_win_prob", "away_games_played", "home_games_played"]], on="game_id")
    for c in ("line", "ask", "bid", "fee_coef", "ask_size"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    pm = -d.home_spread                      # model home margin
    sd_m, sd_t = cal["resid_sd"], cal["total_sd"]
    prob = np.full(len(d), np.nan)
    ml = d.market == "ml"
    prob[ml & (d.side == "home")] = d.home_win_prob[ml & (d.side == "home")]
    prob[ml & (d.side == "away")] = 1 - d.home_win_prob[ml & (d.side == "away")]
    sp = d.market == "spread"
    mean = np.where(d.side == "home", pm, -pm)
    prob[sp] = N.cdf((mean[sp] + d.line[sp]) / sd_m)
    tt = d.market == "total"
    over = N.cdf((d.total - d.line) / sd_t)
    prob[tt] = np.where(d.side[tt] == "over", over[tt], 1 - over[tt])
    d["model_prob"] = prob
    d["fee"] = (d.fee_coef.fillna(0) * d.ask * (1 - d.ask)).round(4)
    d["edge"] = (d.model_prob - d.ask - d.fee).round(4)       # expected $ per $1-payout contract
    d["roi"] = (d.edge / (d.ask + d.fee)).round(4)
    d["ask_american"] = american(d.ask)
    d["fair_american"] = american(d.model_prob)
    d["pick"] = np.select(
        [d.market == "total", d.side == "home"],
        [d.side.str.title() + " " + d.line.astype(str), d.home], d.away)
    sp_lbl = d.line.map(lambda x: "" if pd.isna(x) else f" {x:+g}")
    d.loc[sp, "pick"] = d.loc[sp, "pick"] + sp_lbl[sp]
    d.loc[ml, "pick"] = d.loc[ml, "pick"] + " ML"
    return d


def _between(lo, hi):
    """Whole numbers strictly between lo and hi, as text ('4', '4–5'), or None."""
    import math
    a, z = math.floor(lo) + 1, math.ceil(hi) - 1
    if z < a:
        return None
    return f"{a}" if a == z else f"{a}–{z}"


def _margin_middle(lo, hi, home, away):
    """Home margins strictly between lo and hi, phrased as who wins by how much."""
    import math
    a, z = math.floor(lo) + 1, math.ceil(hi) - 1
    if z < a:
        return None
    rng = lambda u, v: f"{u}" if u == v else f"{u}–{v}"
    if a > 0:
        return f"both win if {home} wins by {rng(a, z)}"
    if z < 0:
        return f"both win if {away} wins by {rng(-z, -a)}"
    return f"both win if the margin is {rng(a, z)}"


def _cost(r):
    return float(r.ask + r.fee)


def arbitrage(board, max_cost=1.0):
    """Pairs of outcomes that can't both lose, bought for less than $1 in total (fees included).

    ml:     home + away
    spread: home at handicap h + away at handicap a, with h + a >= 0   (a + h > 0 leaves a middle)
    total:  over L1 + under L2, with L2 >= L1                           (L2 > L1 leaves a middle)
    Pushes on whole-number lines refund, so they never lose money.
    """
    rows = []
    b = board[board.ask.notna() & (board.ask > 0) & (board.ask < 1)]
    for (gid, mkt), g in b.groupby(["game_id", "market"]):
        g = g.assign(cost=g.ask + g.fee)
        # cheapest offer for each (side, line) across venues
        best = g.sort_values("cost").drop_duplicates(["side", "line"] if mkt != "ml" else ["side"])
        if mkt == "ml":
            pairs = [(x, y) for _, x in best[best.side == "home"].iterrows() for _, y in best[best.side == "away"].iterrows()]
            keep = lambda x, y: True
            middle = lambda x, y: None
        elif mkt == "spread":
            pairs = [(x, y) for _, x in best[best.side == "home"].iterrows() for _, y in best[best.side == "away"].iterrows()]
            keep = lambda x, y: x.line + y.line >= 0
            middle = lambda x, y: _margin_middle(-x.line, y.line, x.home, x.away)
        else:
            pairs = [(x, y) for _, x in best[best.side == "over"].iterrows() for _, y in best[best.side == "under"].iterrows()]
            keep = lambda x, y: y.line >= x.line
            middle = lambda x, y: (f"both win if total is {_between(x.line, y.line)}"
                                   if _between(x.line, y.line) else None)
        for x, y in pairs:
            if not keep(x, y):
                continue
            c = _cost(x) + _cost(y)
            if c < max_cost:
                rows.append(dict(game_id=gid, date=x.date, tipoff=x.tipoff, away=x.away, home=x.home, market=mkt,
                                 leg1=x.pick, leg1_venue=x.venue, leg1_ask=x.ask, leg1_fee=x.fee,
                                 leg2=y.pick, leg2_venue=y.venue, leg2_ask=y.ask, leg2_fee=y.fee,
                                 total_cost=round(c, 4), locked_profit=round(1 - c, 4),
                                 roi=round((1 - c) / c, 4), middle=middle(x, y),
                                 same_venue=x.venue == y.venue))
    a = pd.DataFrame(rows)
    if a.empty:
        return a
    # one row per game/market/leg pair, best first
    return a.sort_values("locked_profit", ascending=False).reset_index(drop=True)


def best_lines(board):
    """Best-value offer per game, market and side across venues (ranked by model edge, i.e. price vs model)."""
    b = board[board.ask.between(0.01, 0.99)]
    return (b.sort_values("edge", ascending=False)
            .drop_duplicates(["game_id", "market", "side"])
            .sort_values(["tipoff", "game_id", "market", "side"]))


def html(board, edges, arbs, status, asof):
    css = ("body{font-family:-apple-system,system-ui,sans-serif;margin:16px;color:#111}"
           "table{border-collapse:collapse;font-size:13px;width:100%;margin:8px 0 24px}"
           "th,td{padding:4px 6px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}"
           "th{background:#f4f4f4}.pos{color:#0a7d32;font-weight:600}.wrap{overflow-x:auto}"
           "small{color:#666}")
    games = []
    for gid, g in board.groupby("game_id", sort=False):
        r = g.iloc[0]
        line = lambda s: (g[(g.market == "ml") & (g.side == s)].sort_values("ask")
                          .apply(lambda x: f"{x.venue[:5]} {x.ask:.2f}", axis=1).str.cat(sep=" · "))
        games.append(dict(tip_et=str(r.tipoff)[11:16], game=f"{r.away} @ {r.home}" + (" (N)" if r.neutral else ""),
                          model=f"{r.home} {r.home_spread:+.1f}, total {r.total:.1f}, {r.home} {r.home_win_prob:.0%}",
                          away_ml=line("away"), home_ml=line("home")))
    gtab = pd.DataFrame(games)
    cols = ["tipoff", "pick", "venue", "ask", "ask_american", "fee", "model_prob", "fair_american", "edge", "roi"]
    fmt_odds = lambda s: s.map(lambda v: "" if pd.isna(v) else f"{int(v):+d}")
    e = (edges[cols].assign(tipoff=edges.tipoff.astype(str).str[11:16], ask_american=fmt_odds(edges.ask_american),
                            fair_american=fmt_odds(edges.fair_american))
         .rename(columns={"ask_american": "ask odds", "fair_american": "fair odds", "model_prob": "model",
                          "tipoff": "tip (ET)"})
         if len(edges) else pd.DataFrame(columns=cols))
    stat = " | ".join(f"{k}: {v}" for k, v in status.items())
    return (f"<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'><title>CBB board {asof}</title>"
            f"<style>{css}</style><h2>CBB board — {asof}</h2><small>Prices pulled {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. {stat}</small>"
            f"<h3>Edges ≥ {MIN_EDGE:.0%} after est. fees</h3><div class=wrap>{e.to_html(index=False, float_format=lambda x: f'{x:.3f}')}</div>"
            f"<h3>Arbitrage</h3><div class=wrap>{arbs[['away','home','leg1','leg1_venue','leg2','leg2_venue','total_cost','locked_profit','middle']].to_html(index=False) if len(arbs) else '<p>None right now.</p>'}</div>"
            f"<h3>All games (asks to buy, $ per $1 payout)</h3><div class=wrap>{gtab.to_html(index=False) if len(gtab) else ''}</div>"
            "<small>Model edges are model probability minus price; the backtest found no reliable edge vs closing lines, "
            "so treat these as where the model disagrees, not as bets.</small>")


def main(asof, days, quotes_path=None, venues_only=None):
    cal = json.loads((DATA / "calibration.json").read_text())
    proj = load_projections(asof, days)
    season = asof.year + 1 if asof.month >= 7 else asof.year
    if quotes_path:
        quotes, status = pd.read_csv(quotes_path, parse_dates=["date"]), {"source": str(quotes_path)}
        quotes["date"] = quotes.date.dt.date
    else:
        quotes, status = venues.fetch_all(asof, asof + timedelta(days=days - 1), venues_only)
        OUT.mkdir(parents=True, exist_ok=True)
        quotes.to_csv(OUT / f"quotes_{asof}.csv", index=False)
    man = manual_quotes()
    if len(man):
        quotes = pd.concat([quotes, man], ignore_index=True)
        status["Manual entries"] = f"ok: {len(man)} quotes"
    quotes = quotes[pd.to_numeric(quotes.ask, errors="coerce").notna()]
    keep = ["date", "tipoff", "game_id", "away", "home", "venue", "market", "pick", "side", "line", "ask", "bid",
            "ask_american", "fee", "fee_coef", "model_prob", "fair_american", "edge", "roi", "ask_size", "market_id"]
    if quotes.empty:                         # every venue down / unreachable: still build the page
        empty = pd.DataFrame(columns=keep)
        OUT.mkdir(parents=True, exist_ok=True)
        empty.to_csv(OUT / f"board_{asof}.csv", index=False)
        status.update(matched_events=0, unmatched_events=0)
        return empty, empty, pd.DataFrame(), status
    matched, unmatched = match(quotes, proj, season)
    board = score(matched, proj, cal)
    ok = board.ask.between(*PRICE_RANGE)
    edges = board[ok & (board.edge >= MIN_EDGE)].sort_values("edge", ascending=False)
    arbs = arbitrage(board)
    board[keep].to_csv(OUT / f"board_{asof}.csv", index=False)
    edges[keep].to_csv(OUT / f"edges_{asof}.csv", index=False)
    arbs.to_csv(OUT / f"arbs_{asof}.csv", index=False)
    best_lines(board)[keep].to_csv(OUT / f"best_lines_{asof}.csv", index=False)
    page = html(board, edges, arbs, status, asof)
    (OUT / f"board_{asof}.html").write_text(page)
    (OUT / "board_latest.html").write_text(page)
    status["matched_events"] = int(matched.event_id.nunique())
    status["unmatched_events"] = int(len(unmatched))
    if len(unmatched):
        unmatched[["venue", "event_name", "date", "team1", "team2"]].to_csv(OUT / f"unmatched_{asof}.csv", index=False)
    return board, edges, arbs, status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(datetime.now(ZoneInfo("America/New_York")).date()))  # slate date in ET
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--quotes")
    ap.add_argument("--venues", nargs="*")
    a = ap.parse_args()
    board, edges, arbs, status = main(date.fromisoformat(a.date), a.days, a.quotes, a.venues)
    pd.set_option("display.width", 220)
    print(json.dumps(status, indent=1, default=str))
    print(edges[["tipoff", "pick", "venue", "ask", "model_prob", "edge"]].head(30).to_string(index=False))
    print(arbs[["away", "home", "leg1", "leg1_venue", "leg2", "leg2_venue", "total_cost", "middle"]].to_string(index=False)
          if len(arbs) else "no arbitrage")
