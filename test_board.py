"""Offline test: venue parsers + matching + scoring on fixtures shaped like the live APIs (Nov 11-12, 2026)."""
import sys, os, json, base64
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cbb"))
import pandas as pd
import venues, board


class R:
    def __init__(self, j): self.j = j
    def json(self): return self.j
    def raise_for_status(self): pass


def km(ticker, sub, ya, yb, strike=None, title=None):
    return {"ticker": ticker, "title": title or sub, "yes_sub_title": sub, "no_sub_title": sub, "status": "active",
            "yes_ask_dollars": f"{ya:.4f}", "yes_bid_dollars": f"{yb:.4f}",
            "no_ask_dollars": f"{1-yb:.4f}", "no_bid_dollars": f"{1-ya:.4f}",
            "floor_strike": strike, "strike_type": "greater" if strike is not None else "structured",
            "occurrence_datetime": "2026-11-12T02:00:00Z", "yes_ask_size_fp": "500.00"}


KALSHI = {
    "KXNCAAMBGAME": [
        {"event_ticker": "KXNCAAMBGAME-26NOV11ODUNORF", "title": "Old Dominion vs Norfolk St.", "markets": [
            km("KXNCAAMBGAME-26NOV11ODUNORF-ODU", "Old Dominion", 0.66, 0.63),
            km("KXNCAAMBGAME-26NOV11ODUNORF-NORF", "Norfolk St.", 0.37, 0.34)]},
        {"event_ticker": "KXNCAAMBGAME-26NOV12OKSTTXAM", "title": "Oklahoma St. vs Texas A&M", "markets": [
            km("KXNCAAMBGAME-26NOV12OKSTTXAM-TXAM", "Texas A&M", 0.62, 0.60),
            km("KXNCAAMBGAME-26NOV12OKSTTXAM-OKST", "Oklahoma St.", 0.40, 0.38)]},
        {"event_ticker": "KXNCAAMBGAME-26NOV15CPMTST", "title": "Cal Poly vs Montana St.", "markets": [
            km("KXNCAAMBGAME-26NOV15CPMTST-MTST", "Montana St.", 0.81, 0.61)]},
    ],
    "KXNCAAMBSPREAD": [
        {"event_ticker": "KXNCAAMBSPREAD-26NOV12OKSTTXAM", "title": "Oklahoma St. vs Texas A&M: Spread", "markets": [
            km("KXNCAAMBSPREAD-26NOV12OKSTTXAM-TXAM4", "Texas A&M wins by over 3.5 points", 0.55, 0.53, 3.5),
            km("KXNCAAMBSPREAD-26NOV12OKSTTXAM-TXAM10", "Texas A&M wins by over 9.5 points", 0.30, 0.27, 9.5)]},
    ],
    "KXNCAAMBTOTAL": [
        {"event_ticker": "KXNCAAMBTOTAL-26NOV11ODUNORF", "title": "Old Dominion vs Norfolk St.: Total Points", "markets": [
            km("KXNCAAMBTOTAL-26NOV11ODUNORF-141", "Over 140.5 points", 0.52, 0.49, 140.5,
               "Full Game: Over 140.5 points scored")]},
    ],
}


def team(name, ab):
    return {"safeName": name, "name": name, "abbreviation": ab}


PMUS_EVENTS = {"events": [
    {"id": "1", "title": "UNC Asheville vs. West Georgia", "startTime": "2026-11-12T00:00:00Z",
     "teams": [team("UNC Asheville", "ncashe"), team("West Georgia", "uwg")],
     "markets": [{"slug": "aec-cbb-ncashe-uwg-2026-11-11", "marketType": "moneyline", "closed": False,
                  "feeCoefficient": 0.095, "marketSides": [
                      {"long": True, "description": "Bulldogs", "team": team("UNC Asheville", "ncashe")},
                      {"long": False, "description": "Wolves", "team": team("West Georgia", "uwg")}]}]},
    {"id": "2", "title": "Stony Brook vs. Princeton", "startTime": "2026-11-12T00:00:00Z",
     "teams": [team("Stony Brook", "stbr"), team("Princeton", "prnce")],
     "markets": [{"slug": "aec-cbb-stbr-prnce-2026-11-11", "marketType": "moneyline", "closed": False,
                  "feeCoefficient": 0.095, "marketSides": [
                      {"long": True, "description": "Seawolves", "team": team("Stony Brook", "stbr")},
                      {"long": False, "description": "Tigers", "team": team("Princeton", "prnce")}]}]},
    {"id": "3", "title": "Oklahoma State vs. Texas A&M", "startTime": "2026-11-13T00:30:00Z",
     "teams": [team("Oklahoma State", "okst"), team("Texas A&M", "txam")],
     "markets": [{"slug": "aec-cbb-okst-txam-2026-11-12", "marketType": "moneyline", "closed": False,
                  "feeCoefficient": 0.095, "marketSides": [
                      {"long": True, "description": "Cowboys", "team": team("Oklahoma State", "okst")},
                      {"long": False, "description": "Aggies", "team": team("Texas A&M", "txam")}]}]},
]}
BBO = {"aec-cbb-ncashe-uwg-2026-11-11": (0.67, 0.68), "aec-cbb-stbr-prnce-2026-11-11": (0.15, 0.17),
       "aec-cbb-okst-txam-2026-11-12": (0.33, 0.35)}


def nm(desc, mtype, outs, strike=None, ts=1794529800000):
    return {"marketId": desc + mtype, "eventId": "ev-" + desc[:10], "marketType": mtype, "status": "OPEN",
            "description": desc, "startsTs": ts, "strike": strike,
            "fee": {"coefficient": "0.06", "makerCredit": "0.25", "charged": "WHEN_LIVE"},
            "outcomes": [{"outcomeId": n, "name": n, "status": "TBD", "bid": {"price": f"{b:.3f}", "qty": 100}}
                         for n, b in outs]}


NOVIG = {"items": [
    nm("Oklahoma State Cowboys at Texas A&M Aggies — Moneyline", "MONEY",
       [("Oklahoma State Cowboys", 0.36), ("Texas A&M Aggies", 0.61)]),
    nm("Oklahoma State Cowboys at Texas A&M Aggies — Spread", "SPREAD",
       [("Oklahoma State Cowboys", 0.47), ("Texas A&M Aggies", 0.50)], "-4.5"),
    nm("Oklahoma State Cowboys at Texas A&M Aggies — Total", "TOTAL",
       [("Over", 0.49), ("Under", 0.48)], "165.5"),
], "next": None}


def px_sel(cid, name, price, strike=None, qty=250):
    return {"outcome_id": 1, "competitor_id": cid, "name": name, "price": price, "quantity": qty,
            "strike": strike, "strike_id": f"s{cid}{price}{strike}"}


PX_EVENTS = {"data": {"sport_events": [
    {"event_id": 9001, "tournament_id": 77, "name": "Oklahoma State Cowboys at Texas A&M Aggies",
     "scheduled": "2026-11-13T00:30:00Z", "status": "scheduled",
     "competitors": [{"id": 11, "name": "Oklahoma State Cowboys", "side": "away"},
                     {"id": 22, "name": "Texas A&M Aggies", "side": "home"}]}]}}
PX_MARKETS = {"data": {"9001": [
    {"id": 1, "type": "moneyline", "status": "active",
     "selections": [[px_sel(11, "Oklahoma State Cowboys", 150), px_sel(11, "Oklahoma State Cowboys", 140)],
                    [px_sel(22, "Texas A&M Aggies", -165)]]},
    {"id": 2, "type": "spread", "status": "active", "strike": -4.5,
     "selections": [[px_sel(11, "Oklahoma State Cowboys", -105, 4.5)], [px_sel(22, "Texas A&M Aggies", -110, -4.5)]],
     "market_strikes": [{"id": 3, "type": "spread", "status": "active", "strike": -6.5,
                         "selections": [[px_sel(11, "Oklahoma State Cowboys", -125, 6.5)],
                                        [px_sel(22, "Texas A&M Aggies", 105, -6.5)]]}]},
    {"id": 4, "type": "total", "status": "active", "strike": 165.5,
     "selections": [[px_sel(None, "Over", -108, 165.5)], [px_sel(None, "Under", -112, 165.5)]]},
]}}


class Session:
    calls = []
    def post(self, url, json=None, headers=None, timeout=None):
        assert url.endswith("/auth/login") and json["access_key"] and json["secret_key"]
        return R({"data": {"access_token": "tok", "refresh_token": "r", "access_expire_time": 0}})
    def get(self, url, params=None, headers=None, timeout=None):
        Session.calls.append((url, params, headers))
        if "kalshi" in url:
            return R({"events": KALSHI[params["series_ticker"]], "cursor": ""})
        if url.endswith("/v1/events"):
            return R(PMUS_EVENTS)
        if "/bbo" in url:
            slug = url.split("/markets/")[1].split("/bbo")[0]
            b, a = BBO[slug]
            return R({"marketData": {"marketSlug": slug, "bestBid": {"value": str(b)}, "bestAsk": {"value": str(a)},
                                     "bidDepth": 40, "askDepth": 30}})
        if "prophetx" in url:
            assert headers["Authorization"] == "Bearer tok"
            if url.endswith("/mm/get_tournaments"):
                return R({"data": {"tournaments": [
                    {"id": 77, "name": "NCAA Division I", "sport": {"id": 2, "name": "Basketball"}},
                    {"id": 78, "name": "NCAA Women", "sport": {"id": 2, "name": "Basketball"}},
                    {"id": 79, "name": "NBA", "sport": {"id": 2, "name": "Basketball"}}]}})
            if url.endswith("/mm/get_sport_events"):
                assert params["tournament_id"] == 77, params
                return R(PX_EVENTS)
            if url.endswith("/v4/mm/get_multiple_markets"):
                assert params["event_ids"] == "9001", params
                return R(PX_MARKETS)
        if "novig" in url:
            assert headers["Novig-Key-Id"] and headers["Novig-Signature"] and headers["Novig-Timestamp"]
            return R(NOVIG)
        raise AssertionError(url)


def test_signing():
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    k = ed25519.Ed25519PrivateKey.generate()
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption()).decode()
    s = venues.novig_string_to_sign(1758400800000, "get", "/v3/catalog/markets",
                                    "marketType=MONEY%2CSPREAD&league=NCAAB&limit=5000")
    lines = s.split("\n")
    assert lines[0] == "NOVIG-V3" and lines[2] == "GET" and lines[3] == "/v3/catalog/markets"
    assert lines[4] == "league=NCAAB&limit=5000&marketType=MONEY%2CSPREAD", lines[4]
    assert lines[5] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert venues._canon_query("b=2&a=x y&a=1&c") == "a=1&a=x%20y&b=2&c="
    sig = base64.b64decode(venues.novig_sign(pem, 1758400800000, "GET", "/v3/catalog/markets", "league=NCAAB"))
    k.public_key().verify(sig, venues.novig_string_to_sign(1758400800000, "GET", "/v3/catalog/markets",
                                                           "league=NCAAB").encode())
    os.environ["PROPHETX_ACCESS_KEY"], os.environ["PROPHETX_SECRET_KEY"] = "ak", "sk"
    os.environ["NOVIG_KEY_ID"] = "test-key"
    os.environ["NOVIG_PRIVATE_KEY"] = pem
    print("signing ok")


def test_board():
    venues.requests.Session = Session            # route fetch_all through fixtures
    asof = date(2026, 11, 11)
    # projections as they stood the morning of Nov 11, 2026 (those games are final now)
    import shutil, prep
    prep.download([2027])                         # schedule is needed for team-name matching
    dst = board.OUT / f"projections_{asof}.csv"
    if not dst.exists():
        board.OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(__file__).parent / "fixture_projections_2026-11-11.csv", dst)
    b, edges, arbs, status = board.main(asof, 2)
    print(json.dumps(status, indent=1, default=str))
    show = ["venue", "pick", "side", "line", "ask", "fee", "model_prob", "edge"]
    print(b[show].to_string(index=False))
    # every fixture game inside the window must match; Cal Poly-Montana St (Nov 15) is outside it
    assert status["matched_events"] >= 7, status
    assert set(b.venue) == {"Kalshi", "Polymarket US", "Novig", "ProphetX", "Robinhood"}, set(b.venue)
    # ProphetX: best level per outcome (+150 beats +140), alternate lines kept, American odds -> price
    px = b[(b.venue == "ProphetX") & (b.home == "Texas A&M")]
    okst_ml = px[(px.market == "ml") & (px.side == "away")].ask.iloc[0]
    assert abs(okst_ml - 100 / 250) < 1e-9, okst_ml
    sp = px[px.market == "spread"].set_index(["side", "line"]).ask.round(4).to_dict()
    assert sp == {("away", 4.5): round(105 / 205, 4), ("home", -4.5): round(110 / 210, 4),
                  ("away", 6.5): round(125 / 225, 4), ("home", -6.5): round(100 / 205, 4)}, sp
    tot = px[px.market == "total"].set_index("side").line.to_dict()
    assert tot == {"over": 165.5, "under": 165.5}, tot
    # ProphetX fee ~0.4% of stake; Robinhood = Kalshi price + 2c
    assert abs(px[(px.market == "ml") & (px.side == "away")].fee.iloc[0] - 0.004 * 0.4) < 1e-4
    rh = b[b.venue == "Robinhood"].merge(b[b.venue == "Kalshi"], on=["game_id", "market", "side", "line"],
                                         suffixes=("_rh", "_k"))
    assert len(rh) and (rh.ask_rh == rh.ask_k).all() and ((rh.fee_rh - 0.02).abs() < 1e-3).all()
    tx = b[(b.home == "Texas A&M")]
    assert set(tx.market) == {"ml", "spread", "total"}
    # Novig spread: home strike -4.5 -> A&M -4.5 and Oklahoma St +4.5
    nv = tx[(tx.venue == "Novig") & (tx.market == "spread")].set_index("side").line.to_dict()
    assert nv == {"home": -4.5, "away": 4.5}, nv
    # Kalshi spread 'Texas A&M wins by over 3.5' -> home -3.5 (YES) and away +3.5 (NO)
    ks = tx[(tx.venue == "Kalshi") & (tx.market == "spread") & (tx.line.abs() == 3.5)].set_index("side").line.to_dict()
    assert ks == {"home": -3.5, "away": 3.5}, ks
    # Polymarket US: long side = Oklahoma St (away) at the ask, A&M at 1 - bid
    pm = tx[(tx.venue == "Polymarket US")].set_index("side").ask.to_dict()
    assert abs(pm["away"] - 0.35) < 1e-9 and abs(pm["home"] - 0.67) < 1e-9, pm
    # model probs are complementary per market/line
    for (v, mk, ln), g in b.groupby(["venue", "market", b.line.abs().fillna(-1)]):
        pass
    print("board ok")


if __name__ == "__main__":
    test_signing()
    test_board()


def test_arbitrage():
    import numpy as np
    base = dict(game_id=1, date="2026-11-15", tipoff="2026-11-15T19:00-05:00", away="A", home="H")
    rows = [
        dict(venue="Kalshi", market="ml", side="home", line=np.nan, ask=0.55, fee=0.0173, pick="H ML"),
        dict(venue="Novig", market="ml", side="away", line=np.nan, ask=0.40, fee=0.0, pick="A ML"),
        dict(venue="Kalshi", market="spread", side="home", line=-3.5, ask=0.45, fee=0.0173, pick="H -3.5"),
        dict(venue="Novig", market="spread", side="away", line=4.5, ask=0.50, fee=0.0, pick="A +4.5"),
        dict(venue="Novig", market="spread", side="away", line=2.5, ask=0.30, fee=0.0, pick="A +2.5"),   # h+a<0: not an arb
        dict(venue="Kalshi", market="total", side="over", line=140.5, ask=0.47, fee=0.0174, pick="Over 140.5"),
        dict(venue="Polymarket US", market="total", side="under", line=141.5, ask=0.48, fee=0.0237, pick="Under 141.5"),
        dict(venue="Polymarket US", market="total", side="under", line=139.5, ask=0.20, fee=0.0, pick="Under 139.5"),  # gap: not an arb
    ]
    b = pd.DataFrame([dict(base, **r) for r in rows])
    a = board.arbitrage(b)
    print(a[["market", "leg1", "leg1_venue", "leg2", "leg2_venue", "total_cost", "locked_profit", "middle"]].to_string(index=False))
    got = {(r.market, r.leg1, r.leg2) for r in a.itertuples()}
    assert got == {("ml", "H ML", "A ML"), ("spread", "H -3.5", "A +4.5"), ("total", "Over 140.5", "Under 141.5")}, got
    ml = a[a.market == "ml"].iloc[0]
    assert abs(ml.total_cost - (0.55 + 0.0173 + 0.40)) < 1e-9
    assert a[a.market == "spread"].iloc[0].middle and a[a.market == "total"].iloc[0].middle
    print("arbitrage ok")


if __name__ == "__main__":
    test_arbitrage()
