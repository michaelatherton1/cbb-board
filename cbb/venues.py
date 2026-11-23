"""Live prices for men's college basketball from Kalshi, Polymarket US and Novig.

Every quote is normalized to one row = "buy this outcome": its ask (cost of a $1-payout contract), bid,
and the venue's taker-fee coefficient. Outcomes are named by team text as the venue shows it; board.py
matches them to ESPN games.

    market   side_team / side         line
    ml       team that must win       -
    spread   team taking the handicap  handicap for that team (-7.5 = must win by 8+)
    total    'over' / 'under'         total points

Endpoints (public unless noted):
  Kalshi        GET api.elections.kalshi.com/trade-api/v2/events  series KXNCAAMBGAME / SPREAD / TOTAL
  Polymarket US GET gateway.polymarket.us/v1/events?seriesId=7 (cbb) + /v1/markets/{slug}/bbo
  Novig         GET api.novig.com/v3/catalog/markets?league=NCAAB   (signed: needs NOVIG_KEY_ID + NOVIG_PRIVATE_KEY)
  ProphetX      GET cash.api.prophetx.co/partner/v4/{mm|affiliate}/get_multiple_markets  (needs keys, see below)
  Robinhood     no API: derived from Kalshi's book (Robinhood routes to it) with Robinhood's per-contract fees
"""
import base64, hashlib, os, re, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, unquote
import pandas as pd
import requests

UA = {"User-Agent": "cbb-model/1.0"}
COLS = ["venue", "event_id", "event_name", "date", "start_utc", "team1", "team2", "market", "side_team",
        "side", "line", "ask", "bid", "ask_size", "fee_coef", "market_id"]
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _num(x):
    if isinstance(x, dict):
        x = x.get("value", x.get("price"))
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _row(**kw):
    r = dict.fromkeys(COLS)
    r.update(kw)
    return r


def _ok(ask):
    return ask is not None and 0.0 < ask < 1.0


def split_matchup(title):
    """'Cal Poly vs Montana St.: Spread' -> ('Cal Poly', 'Montana St.')."""
    t = re.split(r"\s*[:—–]\s*(?:spread|total|moneyline|winner|game|full game).*$", title, flags=re.I)[0]
    t = re.sub(r"\s*[—–-]\s*(moneyline|spread|total|point spread|total points).*$", "", t, flags=re.I)
    parts = re.split(r"\s+(?:vs\.?|v\.?|at|@)\s+", t, maxsplit=1, flags=re.I)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (None, None)


# ----------------------------------------------------------------------------------------- Kalshi
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = {"ml": "KXNCAAMBGAME", "spread": "KXNCAAMBSPREAD", "total": "KXNCAAMBTOTAL"}
KALSHI_FEE = 0.07          # taker fee = round up(0.07 x C x P x (1-P)) per Kalshi's fee schedule


def _kalshi_date(event_ticker):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", event_ticker or "")
    if not m:
        return None
    return datetime(2000 + int(m.group(1)), MONTHS[m.group(2)], int(m.group(3))).date()


def _kalshi_px(m, side):
    v = _num(m.get(f"{side}_dollars"))
    if v is None and m.get(side) is not None:          # older integer-cents fields
        v = _num(m.get(side))
        v = v / 100 if v is not None else None
    return v


def kalshi_events(series, min_close_ts, session=requests):
    events, cursor = [], None
    for _ in range(50):
        params = {"series_ticker": series, "with_nested_markets": "true", "limit": 200,
                  "min_close_ts": int(min_close_ts)}
        if cursor:
            params["cursor"] = cursor
        r = session.get(f"{KALSHI}/events", params=params, headers=UA, timeout=30)
        r.raise_for_status()
        j = r.json()
        events += j.get("events") or []
        cursor = j.get("cursor")
        if not cursor or not j.get("events"):
            break
    return events


def parse_kalshi(events, market):
    rows = []
    for ev in events:
        t1, t2 = split_matchup(ev.get("title", ""))
        if not t1:
            continue
        d = _kalshi_date(ev.get("event_ticker"))
        for m in ev.get("markets") or []:
            if m.get("status") not in (None, "active", "open"):
                continue
            ya, yb = _kalshi_px(m, "yes_ask"), _kalshi_px(m, "yes_bid")
            na, nb = _kalshi_px(m, "no_ask"), _kalshi_px(m, "no_bid")
            if na is None and yb is not None:
                na = 1 - yb
            if nb is None and ya is not None:
                nb = 1 - ya
            size = _num(m.get("yes_ask_size_fp"))
            base = dict(venue="Kalshi", event_id=ev.get("event_ticker"), event_name=ev.get("title"), date=d,
                        start_utc=m.get("occurrence_datetime"), team1=t1, team2=t2, market=market,
                        fee_coef=KALSHI_FEE, market_id=m.get("ticker"))
            sub = (m.get("yes_sub_title") or m.get("title") or "").strip()
            if market == "ml":
                team = re.sub(r"\s+wins?$", "", sub, flags=re.I)
                other = t2 if _same(team, t1) else t1 if _same(team, t2) else None
                rows.append(_row(**base, side_team=team, line=None, ask=ya, bid=yb, ask_size=size))
                if other:
                    rows.append(_row(**base, side_team=other, line=None, ask=na, bid=nb))
            elif market == "spread":
                mt = re.match(r"(.+?)\s+wins by (?:over|more than)\s+([\d.]+)", sub, flags=re.I)
                k = _num(m.get("floor_strike")) or (float(mt.group(2)) if mt else None)
                if not mt or k is None:
                    continue
                team = mt.group(1).strip()
                other = t2 if _same(team, t1) else t1 if _same(team, t2) else None
                rows.append(_row(**base, side_team=team, line=-k, ask=ya, bid=yb, ask_size=size))
                if other:
                    rows.append(_row(**base, side_team=other, line=+k, ask=na, bid=nb))
            elif market == "total":
                k = _num(m.get("floor_strike"))
                if k is None or (m.get("strike_type") not in (None, "greater")):
                    continue
                rows.append(_row(**base, side="over", line=k, ask=ya, bid=yb, ask_size=size))
                rows.append(_row(**base, side="under", line=k, ask=na, bid=nb))
    return rows


def fetch_kalshi(start, end, session=requests):
    min_close = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp()
    rows = []
    for market, series in KALSHI_SERIES.items():
        rows += parse_kalshi(kalshi_events(series, min_close, session), market)
    df = pd.DataFrame(rows, columns=COLS)
    return df[df.date.between(start - timedelta(days=1), end + timedelta(days=1))]


def _same(a, b):
    from names import norm
    return bool(a) and bool(b) and norm(a) == norm(b)


# ----------------------------------------------------------------------------------- Polymarket US
PMUS = "https://gateway.polymarket.us"
PMUS_CBB_SERIES = 7        # /v1/sports -> {"sport": "cbb", "series": "7"}


def pmus_events(start, end, session=requests):
    out, offset = [], 0
    smin = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    smax = datetime.combine(end + timedelta(days=1), datetime.min.time().replace(hour=12),
                            tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for _ in range(40):
        params = {"seriesId": PMUS_CBB_SERIES, "active": "true", "closed": "false", "limit": 100,
                  "offset": offset, "startTimeMin": smin, "startTimeMax": smax}
        r = session.get(f"{PMUS}/v1/events", params=params, headers=UA, timeout=30)
        r.raise_for_status()
        evs = r.json().get("events") or []
        out += evs
        if len(evs) < 100:
            break
        offset += 100
    return out


def pmus_bbo(slug, session=requests):
    r = session.get(f"{PMUS}/v1/markets/{slug}/bbo", headers=UA, timeout=30)
    r.raise_for_status()
    return r.json().get("marketData") or {}


def _team_name(side):
    t = side.get("team") or {}
    return t.get("safeName") or t.get("name") or side.get("description")


def parse_pmus(events, bbo):
    """bbo: dict slug -> marketData. Moneyline only (spread/total markets are not in the public event feed)."""
    rows = []
    for ev in events:
        teams = [t.get("safeName") or t.get("name") for t in ev.get("teams") or []]
        t1, t2 = (teams + [None, None])[:2] if len(teams) >= 2 else split_matchup(ev.get("title", ""))
        start = ev.get("startTime")
        d = pd.Timestamp(start).tz_convert("America/New_York").date() if start else None
        for m in ev.get("markets") or []:
            if m.get("closed") or (m.get("marketType") or "").lower() != "moneyline":
                continue
            sides = m.get("marketSides") or []
            longs = [s for s in sides if s.get("long")]
            shorts = [s for s in sides if not s.get("long")]
            q = bbo.get(m.get("slug"))
            if not longs or not shorts or not q:
                continue
            bid, ask = _num(q.get("bestBid")), _num(q.get("bestAsk"))
            base = dict(venue="Polymarket US", event_id=str(ev.get("id")), event_name=ev.get("title"), date=d,
                        start_utc=start, team1=t1, team2=t2, market="ml",
                        fee_coef=_num(m.get("feeCoefficient")), market_id=m.get("slug"))
            rows.append(_row(**base, side_team=_team_name(longs[0]), ask=ask, bid=bid,
                             ask_size=_num(q.get("askDepth"))))
            rows.append(_row(**base, side_team=_team_name(shorts[0]),
                             ask=(1 - bid) if bid is not None else None,
                             bid=(1 - ask) if ask is not None else None, ask_size=_num(q.get("bidDepth"))))
    return rows


def fetch_pmus(start, end, session=requests):
    evs = pmus_events(start, end, session)
    slugs = [m["slug"] for e in evs for m in e.get("markets") or []
             if (m.get("marketType") or "").lower() == "moneyline" and not m.get("closed") and m.get("slug")]
    bbo = {}
    with ThreadPoolExecutor(8) as ex:
        for slug, res in zip(slugs, ex.map(lambda s: _safe(pmus_bbo, s, session), slugs)):
            if res:
                bbo[slug] = res
    return pd.DataFrame(parse_pmus(evs, bbo), columns=COLS)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


# ------------------------------------------------------------------------------------------- Novig
NOVIG = os.environ.get("NOVIG_BASE_URL", "https://api.novig.com")
NOVIG_LEAGUE = "NCAAB"     # from GET /v3/public/types/leagues


def _canon_query(raw):
    if not raw:
        return ""
    enc = lambda s: quote(s, safe="-._~")
    pairs = []
    for part in raw.split("&"):
        k, _, v = part.partition("=")
        pairs.append((enc(unquote(k)), enc(unquote(v))))
    pairs.sort()
    return "&".join(f"{k}={v}" for k, v in pairs)


def novig_string_to_sign(ts_ms, method, path, raw_query, body=b""):
    return "\n".join(["NOVIG-V3", str(ts_ms), method.upper(), path, _canon_query(raw_query),
                      hashlib.sha256(body).hexdigest()])


def novig_sign(private_pem, ts_ms, method, path, raw_query, body=b""):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519
    key = serialization.load_pem_private_key(private_pem.encode() if isinstance(private_pem, str)
                                             else private_pem, password=None)
    msg = novig_string_to_sign(ts_ms, method, path, raw_query, body).encode()
    if isinstance(key, ed25519.Ed25519PrivateKey):
        sig = key.sign(msg)
    else:  # P-256, DER-encoded ECDSA/SHA-256
        sig = key.sign(msg, ec.ECDSA(hashes.SHA256()))
    return base64.b64encode(sig).decode()


def novig_get(path, params, key_id, private_pem, session=requests):
    raw = "&".join(f"{quote(str(k), safe='')}={quote(str(v), safe='')}" for k, v in params.items())
    ts = int(time.time() * 1000)
    headers = dict(UA, **{"Novig-Key-Id": key_id, "Novig-Timestamp": str(ts),
                          "Novig-Signature": novig_sign(private_pem, ts, "GET", path, raw)})
    r = session.get(f"{NOVIG}{path}?{raw}", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_novig(items):
    rows = []
    for m in items:
        if m.get("status") != "OPEN":
            continue
        desc = m.get("description") or ""
        away, home = split_matchup(desc)          # Novig: "Away at Home — Moneyline"
        if not home:
            continue
        mt = m.get("marketType")
        start = m.get("startsTs")
        start_ts = pd.Timestamp(start, unit="ms", tz="UTC") if start else None
        fee = m.get("fee") or {}
        coef = 0.0 if fee.get("charged") == "WHEN_LIVE" else _num(fee.get("coefficient"))
        outs = m.get("outcomes") or []
        if len(outs) != 2:
            continue
        bids = [_num((o.get("bid") or {}).get("price")) for o in outs]
        qtys = [_num((o.get("bid") or {}).get("qty")) for o in outs]
        base = dict(venue="Novig", event_id=m.get("eventId"), event_name=desc,
                    date=start_ts.tz_convert("America/New_York").date() if start_ts is not None else None,
                    start_utc=str(start_ts) if start_ts is not None else None, team1=away, team2=home,
                    fee_coef=coef, market_id=m.get("marketId"))
        k = _num(m.get("strike"))
        for i, o in enumerate(outs):
            j = 1 - i
            ask = (1 - bids[j]) if bids[j] is not None else None   # buy i = cross the other side's bid
            name = (o.get("name") or "").strip()
            if mt == "MONEY":
                rows.append(_row(**base, market="ml", side_team=name, ask=ask, bid=bids[i], ask_size=qtys[j]))
            elif mt == "SPREAD" and k is not None:
                # strike is the HOME side's handicap
                is_home = _same(name, home) or (not _same(name, away) and name.lower() in home.lower())
                rows.append(_row(**base, market="spread", side_team=name, line=k if is_home else -k,
                                 ask=ask, bid=bids[i], ask_size=qtys[j]))
            elif mt == "TOTAL" and k is not None:
                side = "over" if name.lower().startswith("o") else "under" if name.lower().startswith("u") else None
                if side:
                    rows.append(_row(**base, market="total", side=side, line=k, ask=ask, bid=bids[i],
                                     ask_size=qtys[j]))
    return rows


def fetch_novig(start, end, session=requests):
    key_id, pem = os.environ.get("NOVIG_KEY_ID"), os.environ.get("NOVIG_PRIVATE_KEY")
    if not key_id or not pem:
        raise RuntimeError("NOVIG_KEY_ID / NOVIG_PRIVATE_KEY not set")
    t0 = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc) - timedelta(hours=6)
    t1 = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=12)
    items, after = [], None
    for _ in range(20):
        params = {"league": NOVIG_LEAGUE, "marketType": "MONEY,SPREAD,TOTAL",
                  "startsAfter": int(t0.timestamp() * 1000), "startsBefore": int(t1.timestamp() * 1000),
                  "limit": 5000}
        if after:
            params["after"] = after
        j = novig_get("/v3/catalog/markets", params, key_id, pem, session)
        items += j.get("items") or []
        after = j.get("next")
        if not after or not j.get("items"):
            break
    return pd.DataFrame(parse_novig(items), columns=COLS)


# ----------------------------------------------------------------------------------------- ProphetX
# Two ways in (set either in the environment / GitHub secrets):
#   PROPHETX_ACCESS_KEY + PROPHETX_SECRET_KEY  Trading API keys, self-serve: prophetx.co -> Settings -> Token
#   PROPHETX_AFFILIATE_KEY                     Market Data API key, issued by ProphetX on request
PROPHETX = os.environ.get("PROPHETX_BASE_URL", "https://cash.api.prophetx.co/partner")
PROPHETX_FEE_OF_STAKE = 0.004   # assumption: ~0.4% of stake (ProphetX fee doc example: $0.04 on a $10 fill)


def american_to_prob(a):
    a = _num(a)
    if a is None or a == 0:
        return None
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def _flat(x):
    """selections come as a list, or a list of lists (grouped by side) -> flat list of dicts."""
    out = []
    for item in x or []:
        if isinstance(item, list):
            out += _flat(item)
        elif isinstance(item, dict):
            out.append(item)
    return out


class ProphetX:
    def __init__(self, session=requests):
        self.s = session
        self.aff = os.environ.get("PROPHETX_AFFILIATE_KEY")
        self.ak, self.sk = os.environ.get("PROPHETX_ACCESS_KEY"), os.environ.get("PROPHETX_SECRET_KEY")
        if not self.aff and not (self.ak and self.sk):
            raise RuntimeError("PROPHETX keys not set")
        self.token = None
        self.ns = "affiliate" if self.aff else "mm"

    def _headers(self):
        if self.aff:
            return dict(UA, Authorization=self.aff)
        if not self.token:
            r = self.s.post(f"{PROPHETX}/auth/login", json={"access_key": self.ak, "secret_key": self.sk},
                            headers=UA, timeout=30)
            r.raise_for_status()
            self.token = (r.json().get("data") or {}).get("access_token")
        return dict(UA, Authorization=f"Bearer {self.token}")

    def get(self, path, params=None):
        r = self.s.get(f"{PROPHETX}{path}", params=params, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json().get("data")

    def tournaments(self):
        ids = os.environ.get("PROPHETX_TOURNAMENT_IDS")
        if ids:
            return [int(x) for x in ids.split(",") if x.strip()]
        ts = (self.get(f"/{self.ns}/get_tournaments", {"has_active_events": "true"}) or {}).get("tournaments") or []
        keep = []
        for t in ts:
            sport = ((t.get("sport") or {}).get("name") or "").lower()
            name = (t.get("name") or "").lower()
            if "basketball" in sport and ("ncaa" in name or "college" in name) and "women" not in name:
                keep.append(int(t["id"]))
        return keep

    def events(self, tid):
        return (self.get(f"/{self.ns}/get_sport_events", {"tournament_id": tid}) or {}).get("sport_events") or []

    def markets(self, event_ids):
        out = {}
        for i in range(0, len(event_ids), 50):
            chunk = event_ids[i:i + 50]
            # Trading API wants event_ids=1,2,3 ; Market Data API takes the parameter repeated
            params = {"event_ids": chunk} if self.aff else {"event_ids": ",".join(map(str, chunk))}
            data = self.get(f"/v4/{self.ns}/get_multiple_markets", params) or {}
            if isinstance(data, list):                 # some responses are a flat list of markets
                for m in data:
                    out.setdefault(int(m.get("event_id", 0)), []).append(m)
            else:
                for k, v in data.items():
                    out.setdefault(int(k), []).extend(v or [])
        return out


def parse_prophetx(events, markets):
    rows = []
    for ev in events:
        comps = ev.get("competitors") or []
        side_of = {str(c.get("id")): c.get("side") for c in comps}
        name_of = {c.get("side"): c.get("display_name") or c.get("name") for c in comps}
        away, home = name_of.get("away"), name_of.get("home")
        if not home or not away:
            continue
        start = ev.get("scheduled")
        d = pd.Timestamp(start).tz_convert("America/New_York").date() if start else None
        base = dict(venue="ProphetX", event_id=str(ev.get("event_id")), event_name=ev.get("name"), date=d,
                    start_utc=start, team1=away, team2=home)
        mk = markets.get(int(ev.get("event_id")), [])
        expanded = []
        for m in mk:
            expanded.append(m)
            expanded += m.get("market_strikes") or []          # alternate lines
        best = {}
        for m in expanded:
            mtype = (m.get("type") or "").lower()
            market = {"moneyline": "ml", "spread": "spread", "total": "total"}.get(mtype)
            if not market or (m.get("status") not in (None, "active", "open")):
                continue
            for sel in _flat(m.get("selections")):
                p = _num(sel.get("adjusted_price")) or _num(sel.get("price"))
                ask = american_to_prob(p)
                qty = _num(sel.get("quantity"))
                if ask is None or not (qty is None or qty > 0):
                    continue
                strike = _num(sel.get("strike"))
                if strike is None:
                    strike = _num(m.get("strike"))
                name = (sel.get("display_name") or sel.get("name") or "").strip()
                if market == "total":
                    side = "over" if name.lower().startswith("o") else "under" if name.lower().startswith("u") else None
                    if side is None or strike is None:
                        continue
                    team, key = None, ("total", side, abs(strike))
                else:
                    sd = side_of.get(str(sel.get("competitor_id")))
                    team = {"home": home, "away": away}.get(sd) or name
                    if market == "spread" and strike is None:
                        continue
                    key = (market, team, strike if market == "spread" else None)
                fee_included = sel.get("adjusted_price") is not None
                # several price levels can share an outcome: keep the best one for a buyer
                if key not in best or ask < best[key]["ask"]:
                    best[key] = _row(**base, market=market, side_team=team,
                                     side=key[1] if market == "total" else None,
                                     line=abs(strike) if market == "total" else strike, ask=ask, bid=None,
                                     ask_size=qty, market_id=sel.get("strike_id") or m.get("id"),
                                     fee_coef=0.0 if fee_included else PROPHETX_FEE_OF_STAKE / (1 - ask))
        rows += list(best.values())
    return rows


def fetch_prophetx(start, end, session=requests):
    px = ProphetX(session)
    lo = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    hi = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=12)
    evs = []
    for tid in px.tournaments():
        for e in px.events(tid):
            t = pd.Timestamp(e.get("scheduled")) if e.get("scheduled") else None
            if t is not None and lo <= t.to_pydatetime() <= hi and e.get("status") in (None, "scheduled", "not_started"):
                evs.append(e)
    mk = px.markets([int(e["event_id"]) for e in evs]) if evs else {}
    return pd.DataFrame(parse_prophetx(evs, mk), columns=COLS)


# ---------------------------------------------------------------------------------------- Robinhood
# Robinhood has no API. Its college basketball contracts trade on Kalshi's order book (Robinhood is a broker),
# so the price is Kalshi's; only the fee differs: up to $0.01 Robinhood commission + $0.01 exchange fee.
ROBINHOOD_FEE = 0.02
ROBINHOOD_FROM_KALSHI = os.environ.get("ROBINHOOD_FROM_KALSHI", "1") == "1"


def robinhood_from_kalshi(kalshi):
    if kalshi is None or kalshi.empty:
        return pd.DataFrame(columns=COLS)
    rh = kalshi.copy()
    ask = pd.to_numeric(rh.ask, errors="coerce")
    rh = rh[ask.between(0.01, 0.99)].copy()
    a = pd.to_numeric(rh.ask)
    rh["venue"] = "Robinhood"
    rh["event_id"] = "RH|" + rh.event_id.astype(str)
    rh["fee_coef"] = ROBINHOOD_FEE / (a * (1 - a))       # = flat 2¢ per contract
    return rh[COLS]


# -------------------------------------------------------------------------------------------- all
FETCHERS = {"Kalshi": fetch_kalshi, "Polymarket US": fetch_pmus, "Novig": fetch_novig, "ProphetX": fetch_prophetx}


def _reason(e):
    t, msg = type(e).__name__, str(e)
    if "NOVIG_KEY_ID" in msg or "PROPHETX keys" in msg:
        return "no API key set"
    if t in ("ProxyError", "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "SSLError"):
        return "couldn't connect"
    if t == "HTTPError":
        return msg.split(" for url")[0][:60]
    return f"{t}: {msg[:80]}"


def fetch_all(start, end, venues=None):
    frames, status = [], {}
    s = requests.Session()
    for name, fn in FETCHERS.items():
        if venues and name not in venues:
            continue
        try:
            df = fn(start, end, s)
            frames.append(df)
            status[name] = f"ok: {len(df)} quotes"
            if name == "Kalshi" and ROBINHOOD_FROM_KALSHI and (not venues or "Robinhood" in venues):
                rh = robinhood_from_kalshi(df)
                frames.append(rh)
                status["Robinhood"] = f"ok: {len(rh)} quotes (Kalshi book + 2¢ fee)"
        except Exception as e:  # one venue down shouldn't sink the board
            status[name] = "skipped: " + _reason(e)
            if name == "Kalshi" and ROBINHOOD_FROM_KALSHI:
                status["Robinhood"] = "skipped: uses Kalshi's book, which didn't load"
    q = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    return q, status
