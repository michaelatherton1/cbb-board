"""TheRundown sportsbook feed: parsing (documented v2 schema), caching and the free-plan budget guard.
Payloads here are made up (fictional prices) in the shape of docs.therundown.io's v2 events example."""
import importlib, json, os, sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cbb"))


def _price(p, main=True):
    return {"price": p, "is_main_line": main, "updated_at": "2026-11-28T15:00:00Z"}


def _event(eid, away, home, status="STATUS_SCHEDULED", same_sign_spread=False):
    (aid, an, am), (hid, hn, hm) = away, home
    spread_away = "-4.5" if same_sign_spread else "+4.5"
    return {
        "event_id": eid, "sport_id": 5, "event_date": "2026-11-29T00:30:00Z",
        "score": {"event_status": status},
        "teams": [{"team_id": aid, "name": an, "mascot": am, "is_away": True, "is_home": False},
                  {"team_id": hid, "name": hn, "mascot": hm, "is_away": False, "is_home": True}],
        "markets": [
            {"market_id": 1, "period_id": 0, "name": "moneyline", "participants": [
                {"id": aid, "type": "TYPE_TEAM", "name": f"{an} {am}",
                 "lines": [{"value": "", "prices": {"19": _price(170), "23": _price(165), "22": _price(0.0001)}}]},
                {"id": hid, "type": "TYPE_TEAM", "name": f"{hn} {hm}",
                 "lines": [{"value": "", "prices": {"19": _price(-200), "23": _price(-195), "22": _price(-210)}}]}]},
            {"market_id": 2, "period_id": 0, "name": "handicap", "participants": [
                {"id": aid, "type": "TYPE_TEAM", "name": f"{an} {am}",
                 "lines": [{"value": spread_away, "prices": {"19": _price(-110), "23": _price(-108)}}]},
                {"id": hid, "type": "TYPE_TEAM", "name": f"{hn} {hm}",
                 "lines": [{"value": "-4.5", "prices": {"19": _price(-110), "23": _price(-112)}}]}]},
            {"market_id": 3, "period_id": 0, "name": "totals", "participants": [
                {"id": 900, "type": "TYPE_RESULT", "name": "Over",
                 "lines": [{"value": "141.5", "prices": {"19": _price(-105), "22": _price(-110)}},
                           {"value": "139.5", "prices": {"19": _price(-125, main=False)}}]},
                {"id": 901, "type": "TYPE_RESULT", "name": "Under",
                 "lines": [{"value": "141.5", "prices": {"19": _price(-115), "22": _price(-110)}}]}]},
            {"market_id": 1, "period_id": 1, "name": "1st half moneyline", "participants": []},
        ]}


PAYLOAD = {"meta": {}, "events": [
    _event("E1", (1, "North Ridge", "Hawks"), (2, "Lakeview State", "Lakers")),
    _event("E2", (3, "Pine Bluff Tech", "Owls"), (4, "Riverbend", "Rams"), same_sign_spread=True),
    _event("E3", (5, "Cedar Mills", "Cats"), (6, "Stonegate", "Stags"), status="STATUS_FINAL"),
]}


class _Resp:
    def __init__(self, body, headers, status=200):
        self._b, self.headers, self.status_code = body, headers, status

    def json(self):
        return self._b

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}", response=self)


class _Session:
    def __init__(self, cost=60, monthly_left=199000):
        self.calls, self.cost, self.left = [], cost, monthly_left

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params), headers.get("X-TheRundown-Key")))
        self.left -= self.cost
        return _Resp(PAYLOAD, {"X-Datapoints": str(self.cost), "X-Datapoints-Remaining": "19000",
                               "X-Datapoints-Monthly-Remaining": str(self.left), "X-Bookmakers": "betmgm,fanduel,draftkings",
                               "X-Data-Delay-Seconds": "300", "X-Tier": "free"})


def _venues(tmp_path, monkeypatch):
    monkeypatch.setenv("CBB_OUT", str(tmp_path))
    monkeypatch.setenv("THERUNDOWN_API_KEY", "test-key")
    import venues
    return importlib.reload(venues)


def test_parse(tmp_path, monkeypatch):
    v = _venues(tmp_path, monkeypatch)
    rows = v.parse_therundown(PAYLOAD)
    import pandas as pd
    df = pd.DataFrame(rows)
    assert set(df.event_id) == {"E1", "E2"}                       # final game dropped
    assert set(df.venue) == {"DraftKings Sportsbook", "FanDuel Sportsbook", "BetMGM"}
    e1 = df[df.event_id == "E1"]
    # off-board BetMGM moneyline on the away team is skipped; home -210 kept
    assert len(e1[(e1.venue == "BetMGM") & (e1.market == "ml")]) == 1
    dk_ml = e1[(e1.venue == "DraftKings Sportsbook") & (e1.market == "ml")].set_index("side_team").ask
    assert abs(dk_ml["North Ridge"] - 100 / 270) < 1e-9 and abs(dk_ml["Lakeview State"] - 200 / 300) < 1e-9
    sp = e1[(e1.venue == "FanDuel Sportsbook") & (e1.market == "spread")].set_index("side_team").line
    assert sp["North Ridge"] == 4.5 and sp["Lakeview State"] == -4.5
    # alternate total (is_main_line False) dropped; over/under at 141.5
    tot = e1[(e1.venue == "DraftKings Sportsbook") & (e1.market == "total")]
    assert set(tot.line) == {141.5} and set(tot.side) == {"over", "under"}
    # same number on both teams -> read as the home spread: away +4.5
    sp2 = df[(df.event_id == "E2") & (df.venue == "DraftKings Sportsbook") & (df.market == "spread")].set_index("side_team").line
    assert sp2["Pine Bluff Tech"] == 4.5 and sp2["Riverbend"] == -4.5
    assert (df.fee_coef == 0).all() and str(df.date.iloc[0]) == "2026-11-28"   # 7:30pm ET tip


def test_cache_and_budget(tmp_path, monkeypatch):
    v = _venues(tmp_path, monkeypatch)
    s = _Session()
    d = date(2026, 11, 28)
    t0 = datetime(2026, 11, 28, 15, 17, tzinfo=timezone.utc)            # 10:17 am ET
    q, info = v.fetch_therundown(d, d, s, now=t0)
    assert len(s.calls) == 1 and len(q) > 0
    url, params, key = s.calls[0]
    assert url.endswith("/sports/5/events/2026-11-28") and key == "test-key"
    assert params["affiliate_ids"] == "19,23,22" and params["main_line"] == "true" and params["offset"] == "300"
    assert "as of 10:17 am ET" in info
    # an hour later: not due, served from cache with no API call
    q2, _ = v.fetch_therundown(d, d, s, now=t0.replace(hour=16))
    assert len(s.calls) == 1 and len(q2) == len(q)
    # three hours later: due again
    v.fetch_therundown(d, d, s, now=t0.replace(hour=18))
    assert len(s.calls) == 2
    # tomorrow is only pulled from 5pm ET
    d2 = date(2026, 11, 29)
    v.fetch_therundown(d, d2, s, now=t0.replace(hour=19))              # 2:17 pm ET: neither due
    assert len(s.calls) == 2
    v.fetch_therundown(d, d2, s, now=t0.replace(hour=22))              # 5:17 pm ET: today (4h old) + tomorrow
    assert len(s.calls) == 4 and s.calls[-1][0].endswith("2026-11-29")
    # budget: 30 points left for the month and 2 days to go -> 15/day, less than a ~69-point pull: paused
    st = json.loads((tmp_path / "therundown_state.json").read_text())
    st["usage"]["X-Datapoints-Monthly-Remaining"] = "30"
    (tmp_path / "therundown_state.json").write_text(json.dumps(st))
    q3, info3 = v.fetch_therundown(d, d, s, now=datetime(2026, 11, 29, 1, 17, tzinfo=timezone.utc))
    assert len(s.calls) == 4 and "paused" in info3 and len(q3) > 0    # cached quotes still served


def test_board_keeps_books_out_of_arbs(tmp_path, monkeypatch):
    v = _venues(tmp_path, monkeypatch)
    import pandas as pd
    import board
    importlib.reload(board)
    b = pd.DataFrame([
        dict(game_id=1, date="2026-11-28", tipoff=None, away="A", home="H", venue="DraftKings Sportsbook", market="ml",
             pick="H", side="home", line=None, ask=0.40, fee=0.0),
        dict(game_id=1, date="2026-11-28", tipoff=None, away="A", home="H", venue="Kalshi", market="ml",
             pick="A", side="away", line=None, ask=0.50, fee=0.01),
    ])
    assert len(board.arbitrage(b)) == 1                                # would be an "arb" with the book in
    assert len(board.arbitrage(b[~b.venue.isin(v.REFERENCE_VENUES)])) == 0
