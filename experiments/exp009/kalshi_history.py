"""EXP-009a: collect Kalshi's history for college basketball TOTALS markets (series KXNCAAMBTOTAL).

Data collection only: no pricing or profitability analysis here.

Runs in GitHub Actions (.github/workflows/exp009-kalshi-history.yml) because the dev workspace cannot reach Kalshi.
Public Trade API v2 endpoints, no auth; base URL and User-Agent come from cbb/venues.py.

Output folder (default exp009_out/):
  series.json              GET /series/{series}?include_volume=true (whole JSON: fee_type, fee_multiplier, ...)
  series_fee_changes.json  GET /series/fee_changes?series_ticker=...&show_historical=true
  cutoff.json              GET /historical/cutoff
  events.parquet           one row per event (GET /events, plus GET /events?tickers=... for events only seen via markets)
  markets.parquet          one row per market: live GET /markets + archived GET /historical/markets (column `source`)
  trades.parquet           GET /markets/trades and/or GET /historical/trades per market (deduplicated on trade_id)
  candles_1h.parquet       period_interval=60 over each market's life
  candles_1m.parquet       period_interval=1 around tip (see "Tip time" below)
  rules_sample.json        rules text for a spread of events + every distinct rules_secondary / early_close_condition
  summary.json             counts by event date / season, time ranges, request + error counts, field units used

Units: prices are dollars (0-1). Where the API offers `<x>_dollars` that field is used; otherwise legacy integer cents
are divided by 100 and historical dollar strings are parsed. Contract counts use `<x>_fp` when present.
summary.json["field_units"] records which source field was seen for each column.

Tip time: Kalshi has no tip-off field. cbb/venues.py uses the market's `occurrence_datetime` as start_utc, but for open
markets that is tip + 3h (49/50 games on 2026-11-23 vs ESPN). So events.parquet keeps `start_utc_venues`
(= occurrence_datetime, as venues.py derives it) and `tip_est_utc` (= start_utc_venues - 3h). The 1-minute window is
[start_utc_venues - 7h, start_utc_venues + 15m], i.e. it covers tip-4h .. tip+15m whether occurrence_datetime is the
tip or anything up to tip + 3h (settled markets may carry the recorded time instead). Trim with tip_est_utc.

Phases run in strict priority under a wall-clock budget (--budget-min): meta -> markets -> trades -> candles_1h ->
candles_1m. Within a phase, work is ordered nearest-the-middle rung first, newest event first. If the estimated
request count does not fit the budget, 1-minute candles are restricted to rungs within --near-middle-pts of each
event's middle rung (median strike of its ladder). Rows are flushed to part files as they arrive, so a killed job
still leaves readable data; parts are merged at the end.

Self-test (offline, mocked API):  python experiments/exp009/kalshi_history.py --self-test
"""
from __future__ import annotations

import argparse, json, math, random, shutil, sys, tempfile, threading, time, traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cbb"))
import venues  # noqa: E402  (KALSHI base URL, UA, _kalshi_date)

SERIES = "KXNCAAMBTOTAL"
PHASES = ("meta", "markets", "trades", "candles_1h", "candles_1m")
BATCH_MAX_TICKERS = 100          # GET /markets/candlesticks: up to 100 tickers ...
BATCH_MAX_CANDLES = 10_000       # ... and up to 10,000 candles per call
HIST_MAX_PERIODS = 4_000         # per-market requests are chunked to at most this many periods
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


def log(msg):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ units
FIELD_UNITS: dict[str, set] = defaultdict(set)


def note(field, tag):
    FIELD_UNITS[field].add(tag)


def _f(x):
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def price(d, key, ctx="market"):
    """Dollar price: prefer `<key>_dollars`; else `<key>` (int = legacy cents -> /100, str/float = dollars)."""
    if not d:
        return None
    v = d.get(f"{key}_dollars")
    if v is not None:
        note(f"{ctx}.{key}", f"{key}_dollars")
        return _f(v)
    v = d.get(key)
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        note(f"{ctx}.{key}", f"{key} (int cents /100)")
        return v / 100
    note(f"{ctx}.{key}", f"{key} (dollars)")
    return _f(v)


def qty(d, key, ctx="market"):
    """Contract count: prefer `<key>_fp` (fixed-point string), else `<key>`."""
    if not d:
        return None
    for k in (f"{key}_fp", key):
        if d.get(k) is not None:
            note(f"{ctx}.{key}", k)
            return _f(d.get(k))
    return None


def iso_ts(s):
    """ISO-8601 string -> unix seconds (int) or None."""
    if not s:
        return None
    try:
        return int(pd.Timestamp(s).timestamp())
    except (ValueError, TypeError):
        return None


def utc(ts):
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def season_of(d):
    """CBB season label used in this repo: 2025-26 -> 2026."""
    return None if d is None or pd.isna(d) else (d.year + 1 if d.month >= 8 else d.year)


# ------------------------------------------------------------------------------------------------ HTTP
class Client:
    """Rate-limited GET with retries/backoff (429, 5xx, network errors). Never raises; returns None on failure."""

    def __init__(self, base, session=None, rps=9.0, max_retries=6, sleep=time.sleep, clock=time.monotonic):
        self.base, self.s = base.rstrip("/"), session or requests.Session()
        self.gap, self.max_retries, self.sleep, self.clock = 1.0 / rps, max_retries, sleep, clock
        self._lock, self._next = threading.Lock(), 0.0
        self.n_requests = self.n_retries = 0
        self.errors, self.err_counts = [], Counter()

    def _slot(self):
        with self._lock:
            now = self.clock()
            t = max(now, self._next)
            self._next = t + self.gap
            self.n_requests += 1
        if t > now:
            self.sleep(t - now)

    def _backoff(self, delay):
        with self._lock:            # every worker waits after a 429/5xx, not just this one
            self._next = max(self._next, self.clock() + delay)
            self.n_retries += 1
        self.sleep(delay)

    def error(self, kind, path, params, status, msg):
        with self._lock:
            self.err_counts[kind] += 1
            if len(self.errors) < 300:
                self.errors.append(dict(kind=kind, path=path, status=status, msg=str(msg)[:300],
                                        params={k: (str(v)[:120]) for k, v in (params or {}).items()}))

    def get(self, path, params=None):
        status, msg = None, None
        for attempt in range(self.max_retries + 1):
            self._slot()
            retry_after = 0.0
            try:
                r = self.s.get(self.base + path, params=params, headers=venues.UA, timeout=45)
            except requests.RequestException as e:
                status, msg = None, f"{type(e).__name__}: {e}"
            else:
                status = r.status_code
                if status == 200:
                    try:
                        return r.json()
                    except ValueError as e:
                        msg = f"bad json: {e}"
                elif status in RETRY_STATUS:
                    msg = (r.text or "")[:200]
                    retry_after = _f((r.headers or {}).get("Retry-After")) or 0.0
                else:
                    self.error(f"http_{status}", path, params, status, (r.text or "")[:300])
                    return None
            if attempt < self.max_retries:
                delay = max(retry_after, min(60.0, 0.75 * 2 ** attempt)) * (1 + 0.25 * random.random())
                self._backoff(delay)
        self.error(f"gave_up_{status}", path, params, status, msg)
        return None

    def paginate(self, path, params, key, max_pages=100_000):
        """Cursor pagination. Returns (items, complete)."""
        out, cursor, seen = [], None, set()
        for _ in range(max_pages):
            p = dict(params)
            if cursor:
                p["cursor"] = cursor
            j = self.get(path, p)
            if j is None:
                return out, False
            page = j.get(key) or []
            out += page
            cursor = j.get("cursor")
            if not cursor or not page or cursor in seen:
                return out, True
            seen.add(cursor)
        return out, False


# ------------------------------------------------------------------------------------------------ sinks
CANDLE_PX = [f"{s}_{k}" for s in ("yes_bid", "yes_ask", "price") for k in ("open", "high", "low", "close")] + \
            ["price_mean", "price_previous"]
CANDLE_SCHEMA = pa.schema([("ticker", pa.string()), ("event_ticker", pa.string()), ("source", pa.string()),
                           ("period_min", pa.int16()), ("end_period_ts", pa.int64()),
                           ("end_period_utc", pa.timestamp("s", tz="UTC"))] +
                          [(c, pa.float64()) for c in CANDLE_PX] +
                          [("volume", pa.float64()), ("open_interest", pa.float64())])
TRADE_SCHEMA = pa.schema([("trade_id", pa.string()), ("ticker", pa.string()), ("event_ticker", pa.string()),
                          ("source", pa.string()), ("created_time", pa.timestamp("us", tz="UTC")),
                          ("yes_price", pa.float64()), ("no_price", pa.float64()), ("count", pa.float64()),
                          ("taker_side", pa.string()), ("taker_outcome_side", pa.string()),
                          ("taker_book_side", pa.string()), ("is_block_trade", pa.bool_())])


def _prep_candles(df):
    for c in CANDLE_PX + ["volume", "open_interest"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cents = df.pop("_cents").astype(bool)
    if cents.any():
        df.loc[cents, CANDLE_PX] = df.loc[cents, CANDLE_PX] / 100
    df["end_period_ts"] = pd.to_numeric(df["end_period_ts"], errors="coerce").astype("int64")
    df["end_period_utc"] = pd.to_datetime(df["end_period_ts"], unit="s", utc=True).astype("datetime64[s, UTC]")
    df["period_min"] = df["period_min"].astype("int16")
    return df


def _prep_trades(df):
    for c in ("yes_price", "no_price", "count"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cents = df.pop("_cents").astype(bool)
    if cents.any():
        df.loc[cents, ["yes_price", "no_price"]] = df.loc[cents, ["yes_price", "no_price"]] / 100
    df["created_time"] = pd.to_datetime(df["created_time"], utc=True, errors="coerce", format="ISO8601") \
        .astype("datetime64[us, UTC]")
    df["is_block_trade"] = df["is_block_trade"].astype("boolean")
    return df


class PartSink:
    """Rows -> parquet part files (each part is valid on its own), merged into one file by close()."""

    def __init__(self, out_dir, name, schema, prep, flush_rows=250_000):
        self.dir = Path(out_dir) / f"_parts_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name, self.schema, self.prep, self.flush_rows = name, schema, prep, flush_rows
        self.buf, self.n, self.parts, self.lock = [], 0, 0, threading.Lock()

    def add(self, rows):
        if not rows:
            return
        with self.lock:
            self.buf.extend(rows)
            self.n += len(rows)
            if len(self.buf) >= self.flush_rows:
                self._flush()

    def _flush(self):
        if not self.buf:
            return
        df = self.prep(pd.DataFrame(self.buf))
        tbl = pa.Table.from_pandas(df[self.schema.names], schema=self.schema, preserve_index=False)
        pq.write_table(tbl, self.dir / f"part-{self.parts:05d}.parquet")
        self.parts += 1
        self.buf = []

    def close(self, out_path):
        with self.lock:
            self._flush()
        parts = sorted(self.dir.glob("part-*.parquet"))
        with pq.ParquetWriter(out_path, self.schema) as w:
            for p in parts:
                w.write_table(pq.read_table(p, schema=self.schema))
            if not parts:
                w.write_table(self.schema.empty_table())
        shutil.rmtree(self.dir, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ rows
def candle_rows(cands, ticker, event_ticker, source, period):
    if not cands:
        return []
    c0 = cands[0]
    sub = {**(c0.get("yes_bid") or {}), **(c0.get("price") or {})}
    sfx = "_dollars" if any(k.endswith("_dollars") for k in sub) else ""
    cents = sfx == "" and isinstance((c0.get("yes_bid") or {}).get("close"), int)
    vk = "volume_fp" if "volume_fp" in c0 else "volume"
    ok = "open_interest_fp" if "open_interest_fp" in c0 else "open_interest"
    note(f"candles.{source}.prices", f"open{sfx}" + (" (int cents /100)" if cents else ""))
    note(f"candles.{source}.volume", vk)
    note(f"candles.{source}.open_interest", ok)
    o, h, l, c = "open" + sfx, "high" + sfx, "low" + sfx, "close" + sfx
    out = []
    for x in cands:
        yb, ya, p = x.get("yes_bid") or {}, x.get("yes_ask") or {}, x.get("price") or {}
        out.append({"ticker": ticker, "event_ticker": event_ticker, "source": source, "period_min": period,
                    "end_period_ts": x.get("end_period_ts"), "_cents": cents,
                    "yes_bid_open": yb.get(o), "yes_bid_high": yb.get(h), "yes_bid_low": yb.get(l),
                    "yes_bid_close": yb.get(c),
                    "yes_ask_open": ya.get(o), "yes_ask_high": ya.get(h), "yes_ask_low": ya.get(l),
                    "yes_ask_close": ya.get(c),
                    "price_open": p.get(o), "price_high": p.get(h), "price_low": p.get(l), "price_close": p.get(c),
                    "price_mean": p.get("mean" + sfx), "price_previous": p.get("previous" + sfx),
                    "volume": x.get(vk), "open_interest": x.get(ok)})
    return out


def trade_rows(trades, event_ticker, source):
    out = []
    for t in trades:
        dollars = t.get("yes_price_dollars") is not None
        cents = not dollars and isinstance(t.get("yes_price"), int)
        note(f"trades.{source}.price", "yes_price_dollars" if dollars else
             ("yes_price (int cents /100)" if cents else "yes_price (dollars)"))
        ck = "count_fp" if t.get("count_fp") is not None else "count"
        note(f"trades.{source}.count", ck)
        out.append({"trade_id": t.get("trade_id"), "ticker": t.get("ticker"), "event_ticker": event_ticker,
                    "source": source, "created_time": t.get("created_time"), "_cents": cents,
                    "yes_price": t.get("yes_price_dollars") if dollars else t.get("yes_price"),
                    "no_price": t.get("no_price_dollars") if dollars else t.get("no_price"),
                    "count": t.get(ck), "taker_side": t.get("taker_side"),
                    "taker_outcome_side": t.get("taker_outcome_side"), "taker_book_side": t.get("taker_book_side"),
                    "is_block_trade": t.get("is_block_trade")})
    return out


MARKET_TIMES = ["created_time", "updated_time", "open_time", "close_time", "expected_expiration_time",
                "expiration_time", "latest_expiration_time", "occurrence_datetime", "settlement_ts",
                "fee_waiver_expiration_time"]
MARKET_TEXT = ["ticker", "event_ticker", "market_type", "title", "subtitle", "yes_sub_title", "no_sub_title",
               "strike_type", "status", "result", "early_close_condition", "expiration_value",
               "settlement_bounds_type", "mve_collection_ticker"]


def market_row(m, source):
    r = {k: m.get(k) for k in MARKET_TEXT}
    r["source"] = source
    for k in MARKET_TIMES:
        r[k] = m.get(k)
    for k in ("floor_strike", "cap_strike", "settlement_timer_seconds"):
        r[k] = _f(m.get(k))
    r["functional_strike"] = None if m.get("functional_strike") is None else str(m.get("functional_strike"))
    r["can_close_early"] = m.get("can_close_early")
    r["is_provisional"] = m.get("is_provisional")
    for k in ("yes_bid", "yes_ask", "no_bid", "no_ask", "last_price", "previous_yes_bid", "previous_yes_ask",
              "previous_price", "settlement_value", "notional_value", "settlement_floor"):
        r[k] = price(m, k)
    for k in ("volume", "volume_24h", "open_interest", "yes_bid_size", "yes_ask_size"):
        r[k] = qty(m, k)
    pls = m.get("price_level_structure")
    r["price_level_structure"] = None if pls is None else json.dumps(pls)
    raw = {k: v for k, v in m.items() if k not in ("rules_primary", "rules_secondary")}
    r["raw_json"] = json.dumps(raw, sort_keys=True)
    return r


# ------------------------------------------------------------------------------------------------ runner
class Runner:
    def __init__(self, cfg, client, clock=time.monotonic, now=None):
        self.cfg, self.c, self.clock = cfg, client, clock
        self.t0 = clock()
        self.deadline = self.t0 + cfg.budget_min * 60
        self.now = int(now if now is not None else time.time())
        self.out = Path(cfg.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.summary = {"config": {k: v for k, v in vars(cfg).items() if k != "self_test"},
                        "started_utc": utc(self.now), "phases": {}}
        self.mk = None            # markets DataFrame
        self.ev = None            # events DataFrame
        self.stat = defaultdict(dict)   # ticker -> {n_trades, trades_status, ...}
        self.seen_trades = set()
        self.dupes = 0
        self.batch_splits = 0

    # ---------------------------------------------------------------- helpers
    def left(self):
        return self.deadline - self.clock()

    def dump(self, name, obj):
        (self.out / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))

    def set_stat(self, ticker, **kw):
        with self.lock:
            self.stat[ticker].update(kw)

    def add_stat(self, ticker, key, n):
        with self.lock:
            self.stat[ticker][key] = self.stat[ticker].get(key, 0) + n

    def run_tasks(self, name, items, fn):
        """Run fn(item) on a thread pool in priority order; stop starting new items at the deadline."""
        n, done, started, last = len(items), 0, 0, self.clock()
        t_start = self.clock()
        it = iter(items)
        futs, stop = set(), False
        with ThreadPoolExecutor(self.cfg.workers) as ex:
            while True:
                while not stop and len(futs) < self.cfg.workers * 3:
                    if self.left() <= 0:
                        stop = True
                        break
                    try:
                        item = next(it)
                    except StopIteration:
                        break
                    futs.add(ex.submit(self._safe, name, fn, item))
                    started += 1
                if not futs:
                    break
                fin, futs = wait(futs, timeout=5, return_when=FIRST_COMPLETED)
                done += len(fin)
                if self.clock() - last > 30 or (not futs and done == n):
                    el = self.clock() - t_start
                    rate = done / el if el > 0 else 0
                    eta = (n - done) / rate / 60 if rate else float("nan")
                    log(f"{name}: {done}/{n} done, {self.c.n_requests} requests total, "
                        f"{rate:.2f} items/s, ETA {eta:.1f} min, budget left {self.left() / 60:.1f} min")
                    last = self.clock()
        return {"items": n, "done": done, "not_started": n - started, "complete": started == n}

    def _safe(self, name, fn, item):
        try:
            fn(item)
        except Exception as e:  # a parsing bug in one item must not kill the run
            self.c.error(f"exception_{name}", name, {"item": str(item)[:200]}, None,
                         f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-400:]}")

    # ---------------------------------------------------------------- phase: meta
    def phase_meta(self):
        S = self.cfg.series
        series = self.c.get(f"/series/{S}", {"include_volume": "true"})
        self.dump("series.json", series if series is not None else {"error": "fetch failed"})
        cutoff = self.c.get("/historical/cutoff")
        self.dump("cutoff.json", cutoff if cutoff is not None else {"error": "fetch failed"})
        fees = self.c.get("/series/fee_changes", {"series_ticker": S, "show_historical": "true"})
        self.dump("series_fee_changes.json", fees if fees is not None else {"error": "fetch failed"})
        self.cutoff = cutoff or {}
        s = (series or {}).get("series") or {}
        self.summary["series_fee"] = {k: s.get(k) for k in ("fee_type", "fee_multiplier")}
        self.summary["series_fee"]["fee_changes"] = (fees or {}).get("series_fee_change_arr")
        self.summary["cutoff"] = cutoff
        self.summary["phases"]["meta"] = {"series_ok": series is not None, "cutoff_ok": cutoff is not None,
                                          "fee_changes_ok": fees is not None}
        log(f"meta: series fee {self.summary['series_fee']}, cutoff {cutoff}")

    # ---------------------------------------------------------------- phase: markets
    def phase_markets(self):
        S, cfg = self.cfg.series, self.cfg
        evs, ev_ok = self.c.paginate("/events", {"series_ticker": S, "limit": 200}, "events")
        log(f"markets: {len(evs)} events from /events (complete={ev_ok})")
        live, live_ok = self.c.paginate("/markets", {"series_ticker": S, "limit": 1000}, "markets")
        log(f"markets: {len(live)} live markets (complete={live_ok})")
        hist, hist_ok = self.c.paginate("/historical/markets", {"series_ticker": S, "limit": 1000}, "markets")
        log(f"markets: {len(hist)} historical markets (complete={hist_ok})")

        rows, raw_by_ticker, overlap = {}, {}, 0
        for src, ms in (("live", live), ("historical", hist)):
            for m in ms:
                t = m.get("ticker")
                if not t:
                    continue
                if t in rows:
                    overlap += 1
                    continue
                rows[t] = market_row(m, src)
                raw_by_ticker[t] = m
        mk = pd.DataFrame(list(rows.values()))
        if mk.empty:
            mk = pd.DataFrame(columns=["ticker", "event_ticker", "source", "floor_strike", "cap_strike"] +
                              MARKET_TIMES)
        mk["event_date"] = mk["event_ticker"].map(venues._kalshi_date)
        if cfg.min_event_date:
            mk = mk[mk.event_date >= pd.Timestamp(cfg.min_event_date).date()]
        if cfg.max_event_date:
            mk = mk[mk.event_date <= pd.Timestamp(cfg.max_event_date).date()]
        if cfg.max_markets:
            mk = mk.sort_values("event_date", ascending=False).head(cfg.max_markets)
        for k in MARKET_TIMES:
            mk[k + "_ts"] = pd.to_numeric(mk[k].map(iso_ts), errors="coerce")

        # ladder geometry: middle = median strike of the event's ladder
        fl, cp = pd.to_numeric(mk["floor_strike"]), pd.to_numeric(mk["cap_strike"])
        strike = fl.where(fl.notna(), cp)
        mk["strike"] = strike.where(~(fl.notna() & cp.notna()), (fl + cp) / 2)     # 'between' rungs: midpoint
        mid = mk.groupby("event_ticker")["strike"].agg(lambda s: float(pd.Series(s.dropna().unique()).median())
                                                       if s.notna().any() else math.nan)
        mk["middle_strike"] = mk["event_ticker"].map(mid)
        mk["dist_from_middle"] = (mk["strike"] - mk["middle_strike"]).abs()
        mk["rung_rank"] = mk.groupby("event_ticker")["dist_from_middle"].rank(method="first").fillna(999) - 1
        mk["n_rungs_event"] = mk.groupby("event_ticker")["ticker"].transform("count")

        # events: /events + /events?tickers= for event tickers only seen through markets
        ev_by = {e.get("event_ticker"): e for e in evs if e.get("event_ticker")}
        missing = sorted(set(mk["event_ticker"].dropna()) - set(ev_by))
        got_missing = 0
        for i in range(0, len(missing), 100):
            chunk = missing[i:i + 100]
            more, _ = self.c.paginate("/events", {"tickers": ",".join(chunk), "limit": 200}, "events")
            for e in more:
                if e.get("event_ticker") in chunk:
                    ev_by[e["event_ticker"]] = e
                    got_missing += 1
        self.ev = self._events_frame(mk, ev_by, raw_by_ticker, set(e.get("event_ticker") for e in evs))
        # start time as venues.py derives it, carried onto markets
        mk = mk.merge(self.ev[["event_ticker", "start_utc_venues", "tip_est_utc"]], on="event_ticker", how="left")
        mk["start_ts"] = pd.to_numeric(mk["start_utc_venues"].map(iso_ts), errors="coerce")
        self.mk = mk.reset_index(drop=True)
        self.summary["phases"]["markets"] = {
            "events_endpoint": len(evs), "events_endpoint_complete": ev_ok,
            "live_markets": len(live), "live_complete": live_ok,
            "historical_markets": len(hist), "historical_complete": hist_ok,
            "overlap_live_and_historical": overlap, "markets_kept": int(len(mk)),
            "events_kept": int(len(self.ev)), "events_only_seen_via_markets": len(missing),
            "of_which_found_by_tickers_lookup": got_missing}
        self.write_markets()
        log(f"markets: kept {len(mk)} markets in {len(self.ev)} events "
            f"({int((mk.source == 'historical').sum())} from historical endpoints)")

    def _events_frame(self, mk, ev_by, raw_by_ticker, from_list):
        rows = []
        for et, g in mk.groupby("event_ticker"):
            e = ev_by.get(et) or {}
            occ = g["occurrence_datetime"].dropna()
            start = occ.mode().iloc[0] if len(occ) else None
            mid_t = g.sort_values("rung_rank")["ticker"].iloc[0]
            raw_mid = raw_by_ticker.get(mid_t, {})
            tip = (pd.Timestamp(start) - pd.Timedelta(hours=self.cfg.tip_offset_h)) if start else None
            rows.append({
                "event_ticker": et, "series_ticker": e.get("series_ticker"), "title": e.get("title"),
                "sub_title": e.get("sub_title"), "category": e.get("category"), "strike_date": e.get("strike_date"),
                "strike_period": e.get("strike_period"), "mutually_exclusive": e.get("mutually_exclusive"),
                "fee_type_override": None if e.get("fee_type_override") is None else str(e.get("fee_type_override")),
                "fee_multiplier_override": _f(e.get("fee_multiplier_override")),
                "event_date": g["event_date"].iloc[0], "season": season_of(g["event_date"].iloc[0]),
                "start_utc_venues": start, "n_distinct_occurrence": int(occ.nunique()),
                "tip_est_utc": None if tip is None else tip.isoformat().replace("+00:00", "Z"),
                "n_markets": int(len(g)), "n_markets_historical": int((g.source == "historical").sum()),
                "min_strike": _f(g["strike"].min()), "max_strike": _f(g["strike"].max()),
                "middle_strike": _f(g["middle_strike"].iloc[0]),
                "first_open_time": g["open_time"].dropna().min() if g["open_time"].notna().any() else None,
                "last_close_time": g["close_time"].dropna().max() if g["close_time"].notna().any() else None,
                "statuses": ",".join(sorted(set(g["status"].dropna().astype(str)))),
                "results": ",".join(sorted(set(g["result"].dropna().astype(str)))),
                "middle_ticker": mid_t,
                "rules_primary": raw_mid.get("rules_primary"), "rules_secondary": raw_mid.get("rules_secondary"),
                "early_close_condition": raw_mid.get("early_close_condition"),
                "in_events_list": et in from_list, "event_found": bool(e),
                "raw_json": json.dumps({k: v for k, v in e.items() if k != "markets"}, sort_keys=True) if e else None})
        ev = pd.DataFrame(rows)
        if not ev.empty:
            for k in ("start_utc_venues", "tip_est_utc", "first_open_time", "last_close_time"):
                ev[k] = ev[k].astype(object)
        return ev

    def write_markets(self):
        mk = self.mk.copy()
        for k, v in (("n_trades", 0), ("trades_status", "not_reached"), ("n_candles_1h", 0),
                     ("h1_status", "not_reached"), ("n_candles_1m", 0), ("m1_status", "not_reached"),
                     ("m1_window_start", None), ("m1_window_end", None)):
            mk[k] = [self.stat.get(t, {}).get(k, v) for t in mk["ticker"]]
        for k in MARKET_TIMES + ["start_utc_venues", "tip_est_utc"]:
            mk[k] = pd.to_datetime(mk[k], utc=True, errors="coerce", format="ISO8601")
        for k in ("m1_window_start", "m1_window_end"):
            mk[k] = pd.to_datetime(mk[k], unit="s", utc=True, errors="coerce")
        mk.drop(columns=[c for c in mk.columns if c.endswith("_ts") and c[:-3] in MARKET_TIMES] + ["start_ts"],
                errors="ignore").to_parquet(self.out / "markets.parquet", index=False)
        ev = self.ev.copy()
        for k in ("start_utc_venues", "tip_est_utc", "first_open_time", "last_close_time"):
            ev[k] = pd.to_datetime(ev[k], utc=True, errors="coerce", format="ISO8601")
        ev.to_parquet(self.out / "events.parquet", index=False)

    # ---------------------------------------------------------------- planning
    def priority(self, df):
        """nearest-the-middle rung first, newest event first."""
        return df.assign(dsort=pd.to_datetime(df["event_date"])).sort_values(
            ["rung_rank", "dsort"], ascending=[True, False]).drop(columns="dsort")

    def windows_1h(self):
        mk = self.mk
        start = mk["open_time_ts"]
        end = pd.concat([mk["close_time_ts"], mk["settlement_ts_ts"]], axis=1).min(axis=1)
        end = end.where(end.notna(), self.now).clip(upper=self.now)
        end = (end // 3600 + 1) * 3600                       # include the candle that contains the close
        start = start.where(start.notna(), end - 3 * 86400)
        start = pd.concat([start, end - self.cfg.h1_max_days * 86400], axis=1).max(axis=1)
        return start, end

    def windows_1m(self):
        mk, cfg = self.mk, self.cfg
        anchor = mk["start_ts"].where(mk["start_ts"].notna(), mk["close_time_ts"])
        ws = anchor - int((cfg.tip_offset_h * 60 + cfg.m1_before_tip_min) * 60)
        we = anchor + int(cfg.m1_after_start_min * 60)
        life_s = mk["open_time_ts"]
        life_e = mk["close_time_ts"].where(mk["close_time_ts"].notna(), self.now).clip(upper=self.now)
        ws = pd.concat([ws, life_s], axis=1).max(axis=1)
        we = pd.concat([we, life_e], axis=1).min(axis=1)
        return ws, we, anchor

    def estimate(self, restrict):
        mk = self.mk
        vol = mk["volume"]
        n_trade_markets = int((vol.isna() | (vol > 0)).sum())
        live = mk[mk.source == "live"]
        hs, he = self.windows_1h()
        per = ((he - hs) / 3600).clip(lower=1)
        h1_live = max(1, math.ceil(float((per[mk.source == "live"]).sum()) / (BATCH_MAX_CANDLES * 0.9))) \
            if len(live) else 0
        h1_live = max(h1_live, live["event_ticker"].nunique() // 4)
        h1_hist = int((mk.source == "historical").sum())
        m1 = mk if not restrict else mk[mk["dist_from_middle"] <= self.cfg.near_middle_pts]
        m1_live = m1[m1.source == "live"]["event_ticker"].nunique()
        m1_hist = int((m1.source == "historical").sum())
        calls = {"trades": int(n_trade_markets * 1.1), "candles_1h": h1_live + h1_hist,
                 "candles_1m": m1_live + m1_hist}
        rps = min(self.cfg.rps, self.cfg.workers * 4.0)
        return calls, {k: v / rps / 60 for k, v in calls.items()}

    # ---------------------------------------------------------------- phase: trades
    def phase_trades(self):
        mk = self.mk
        tc = iso_ts(self.cutoff.get("trades_created_ts")) if self.cutoff else None
        zero = mk["volume"].notna() & (mk["volume"] <= 0)
        for t in mk.loc[zero, "ticker"]:
            self.set_stat(t, trades_status="skipped_zero_volume")
        todo = self.priority(mk[~zero])
        items = []
        for r in todo.itertuples():
            eps = []
            if r.source == "historical":
                eps.append("historical")
                if tc is not None and r.close_time_ts >= tc:          # NaN compares False
                    eps.append("live")
            else:
                eps.append("live")
                if tc is not None and not (r.open_time_ts >= tc):     # opened before the cutoff, or unknown
                    eps.append("historical")
            items.append((r.ticker, r.event_ticker, tuple(eps)))
        self.sink_tr = PartSink(self.out, "trades", TRADE_SCHEMA, _prep_trades)
        res = self.run_tasks("trades", items, self._trades_one)
        self.sink_tr.close(self.out / "trades.parquet")
        res.update(rows=self.sink_tr.n, duplicates_dropped=self.dupes,
                   markets_zero_volume_skipped=int(zero.sum()))
        self.summary["phases"]["trades"] = res

    def _trades_one(self, item):
        ticker, et, eps = item
        n, ok = 0, True
        for ep in eps:
            path = "/historical/trades" if ep == "historical" else "/markets/trades"
            trades, complete = self.c.paginate(path, {"ticker": ticker, "limit": 1000}, "trades")
            ok &= complete
            fresh = []
            with self.lock:
                for t in trades:
                    tid = t.get("trade_id")
                    if tid in self.seen_trades:
                        self.dupes += 1
                        continue
                    self.seen_trades.add(tid)
                    fresh.append(t)
            self.sink_tr.add(trade_rows(fresh, et, ep))
            n += len(fresh)
        self.set_stat(ticker, n_trades=n, trades_status="ok" if ok else "error")

    # ---------------------------------------------------------------- candles (shared)
    def _batch(self, tickers, start, end, period):
        """GET /markets/candlesticks; returns {ticker: [candles]} or None. Splits if the 10k cap is hit."""
        j = self.c.get("/markets/candlesticks", {"market_tickers": ",".join(tickers), "start_ts": int(start),
                                                 "end_ts": int(end), "period_interval": period})
        if j is None:
            return None
        got = {m.get("market_ticker") or m.get("ticker"): (m.get("candlesticks") or [])
               for m in (j.get("markets") or [])}
        if sum(len(v) for v in got.values()) >= self.cfg.batch_max_candles:
            with self.lock:
                self.batch_splits += 1
            if len(tickers) == 1:             # one market alone hits the cap: per-market endpoint, chunked
                cands, ok = self._single(tickers[0], "live", start, end, period)
                return {tickers[0]: cands} if ok else None
            h = len(tickers) // 2
            a, b = self._batch(tickers[:h], start, end, period), self._batch(tickers[h:], start, end, period)
            if a is None or b is None:
                return None
            return {**a, **b}
        return got

    def _single(self, ticker, source, start, end, period):
        """Per-market candles (historical endpoint, or live per-market endpoint), chunked by HIST_MAX_PERIODS."""
        out, ok, s = [], True, int(start)
        step = HIST_MAX_PERIODS * period * 60
        while s < end:
            e = min(int(end), s + step)
            if source == "historical":
                j = self.c.get(f"/historical/markets/{ticker}/candlesticks",
                               {"start_ts": s, "end_ts": e, "period_interval": period})
            else:
                j = self.c.get(f"/series/{self.cfg.series}/markets/{ticker}/candlesticks",
                               {"start_ts": s, "end_ts": e, "period_interval": period})
            if j is None:
                ok = False
            else:
                out += j.get("candlesticks") or []
            s = e + 1
        return out, ok

    def _emit(self, sink, key, ticker, et, source, period, cands, ok, status_key):
        rows = candle_rows(cands, ticker, et, source, period)
        sink.add(rows)
        self.set_stat(ticker, **{key: len(rows), status_key: "ok" if ok else "error"})

    def _live_groups(self, df, ws, we, period):
        """Pack live markets (in priority order of events) into batch calls within the ticker/candle caps."""
        groups = []
        df = df.assign(wstart=ws[df.index], wend=we[df.index], dsort=pd.to_datetime(df["event_date"]))
        ev_order = df.groupby("event_ticker")["dsort"].first().sort_values(ascending=False).index
        by_ev = {et: g.sort_values("rung_rank") for et, g in df.groupby("event_ticker")}
        cap = self.cfg.batch_max_candles * self.cfg.batch_fill
        cur, cs, ce = [], None, None
        for et in ev_order:
            for r in by_ev[et].itertuples():
                s2 = r.wstart if cs is None else min(cs, r.wstart)
                e2 = r.wend if ce is None else max(ce, r.wend)
                n_per = (e2 - s2) / (period * 60) + 2
                if cur and (len(cur) >= BATCH_MAX_TICKERS or (len(cur) + 1) * n_per > cap):
                    groups.append((cur, cs, ce))
                    cur, cs, ce = [], None, None
                    s2, e2 = r.wstart, r.wend
                cur.append((r.ticker, r.event_ticker))
                cs, ce = s2, e2
        if cur:
            groups.append((cur, cs, ce))
        return groups

    def _candle_phase(self, name, period, sel, ws, we, key, status_key):
        self.batch_splits = 0
        mk = self.mk.loc[sel]
        for t in mk.loc[~(we[mk.index] > ws[mk.index]), "ticker"]:
            self.set_stat(t, **{status_key: "no_window"})
        mk = mk[we[mk.index] > ws[mk.index]]
        sink = PartSink(self.out, name, CANDLE_SCHEMA, _prep_candles)
        live = mk[mk.source == "live"]
        hist = self.priority(mk[mk.source == "historical"])
        items = [("batch", g) for g in self._live_groups(live, ws, we, period)] + \
                [("single", (r.ticker, r.event_ticker, int(ws[r.Index]), int(we[r.Index])))
                 for r in hist.itertuples()]

        def one(item):
            kind, payload = item
            if kind == "single":
                t, et, s, e = payload
                cands, ok = self._single(t, "historical", s, e, period)
                self._emit(sink, key, t, et, "historical", period, cands, ok, status_key)
                return
            members, s, e = payload
            got = self._batch([t for t, _ in members], s, e, period)
            for t, et in members:
                if got is None:      # batch failed: per-market live endpoint
                    cands, ok = self._single(t, "live", ws[self._ix[t]], we[self._ix[t]], period)
                else:
                    cands, ok = got.get(t, []), True
                self._emit(sink, key, t, et, "live", period, cands, ok, status_key)

        self._ix = {t: i for i, t in zip(self.mk.index, self.mk["ticker"])}
        res = self.run_tasks(name, items, one)
        sink.close(self.out / f"{name}.parquet")
        res.update(rows=sink.n, markets=int(len(mk)), live_batches=len(items) - len(hist),
                   historical_markets=int(len(hist)), batch_splits=self.batch_splits)
        return res

    def phase_candles_1h(self):
        ws, we = self.windows_1h()
        res = self._candle_phase("candles_1h", 60, self.mk.index, ws, we, "n_candles_1h", "h1_status")
        self.summary["phases"]["candles_1h"] = res

    def phase_candles_1m(self, restrict):
        ws, we, anchor = self.windows_1m()
        sel = self.mk.index
        if restrict:
            far = self.mk["dist_from_middle"].isna() | (self.mk["dist_from_middle"] > self.cfg.near_middle_pts)
            for t in self.mk.loc[far, "ticker"]:
                self.set_stat(t, m1_status="skipped_far_rung")
            sel = self.mk.index[~far]
        for i in sel:
            t = self.mk.at[i, "ticker"]
            if pd.notna(ws[i]) and pd.notna(we[i]):
                self.set_stat(t, m1_window_start=int(ws[i]), m1_window_end=int(we[i]))
        ws2 = ws.fillna(0)
        we2 = we.fillna(0)
        res = self._candle_phase("candles_1m", 1, sel, ws2, we2, "n_candles_1m", "m1_status")
        res.update(restricted_near_middle=bool(restrict), near_middle_pts=self.cfg.near_middle_pts,
                   window=f"[start_utc_venues - {self.cfg.tip_offset_h}h - {self.cfg.m1_before_tip_min}min, "
                          f"start_utc_venues + {self.cfg.m1_after_start_min}min], clamped to market life",
                   markets_without_start_used_close_time=int(self.mk["start_ts"].isna().sum()))
        self.summary["phases"]["candles_1m"] = res

    # ---------------------------------------------------------------- summary
    def finish(self):
        mk = self.mk
        if mk is not None:
            self.write_markets()
            self.write_rules()
        s = self.summary
        s["finished_utc"] = utc(self.now + int(self.clock() - self.t0))
        s["elapsed_min"] = round((self.clock() - self.t0) / 60, 2)
        s["requests"], s["retries"] = self.c.n_requests, self.c.n_retries
        s["errors"] = {"count": sum(self.c.err_counts.values()), "by_kind": dict(self.c.err_counts),
                       "first": self.c.errors[:100]}
        s["field_units"] = {k: sorted(v) for k, v in sorted(FIELD_UNITS.items())}
        if mk is not None and len(mk):
            s.update(self.coverage())
        self.dump("summary.json", s)
        return s

    def coverage(self):
        out = self.out
        m = pd.read_parquet(out / "markets.parquet")
        cnt = {"markets": m.groupby("event_date").size(),
               "events": m.groupby("event_date")["event_ticker"].nunique(),
               "markets_historical": m[m.source == "historical"].groupby("event_date").size(),
               "markets_settled_with_result": m[m.result.isin(["yes", "no"])].groupby("event_date").size(),
               "volume": m.groupby("event_date")["volume"].sum()}
        cov = {}
        for name, col in (("trades", "n_trades"), ("candles_1h", "n_candles_1h"), ("candles_1m", "n_candles_1m")):
            cnt[name] = m.groupby("event_date")[col].sum()
            p = out / f"{name}.parquet"
            if p.exists():
                tcol = "created_time" if name == "trades" else "end_period_utc"
                t = pq.read_table(p, columns=[tcol]).column(0)
                cov[name] = {"rows": len(t), "min": str(pc.min(t).as_py()) if len(t) else None,
                             "max": str(pc.max(t).as_py()) if len(t) else None}
        by_date = pd.DataFrame(cnt).fillna(0).reset_index().rename(columns={"index": "event_date"})
        by_date["event_date"] = by_date["event_date"].astype(str)
        by_date["season"] = pd.to_datetime(by_date["event_date"]).map(season_of)
        by_season = by_date.groupby("season").agg({c: "sum" for c in cnt}).reset_index()
        by_season["first_date"] = by_date.groupby("season")["event_date"].min().values
        by_season["last_date"] = by_date.groupby("season")["event_date"].max().values
        status = {c: m[c].value_counts().to_dict() for c in ("trades_status", "h1_status", "m1_status", "status",
                                                              "result", "source")}
        return {"time_range": {"event_date_min": str(m.event_date.min()), "event_date_max": str(m.event_date.max()),
                               "open_time_min": str(m.open_time.min()), "close_time_max": str(m.close_time.max()),
                               **{f"{k}_rows": v["rows"] for k, v in cov.items()},
                               **{f"{k}_min": v["min"] for k, v in cov.items()},
                               **{f"{k}_max": v["max"] for k, v in cov.items()}},
                "counts_by_season": json.loads(by_season.to_json(orient="records")),
                "counts_by_event_date": json.loads(by_date.to_json(orient="records")),
                "statuses": {k: {str(a): int(b) for a, b in v.items()} for k, v in status.items()}}

    def write_rules(self):
        ev = self.ev.sort_values("event_date") if self.ev is not None and len(self.ev) else None
        if ev is None:
            self.dump("rules_sample.json", {})
            return
        idx = sorted(set(int(round(x)) for x in pd.Series(range(10)) * (len(ev) - 1) / 9)) if len(ev) > 1 else [0]
        sample = [{k: (str(v) if k == "event_date" else v) for k, v in ev.iloc[i][
            ["event_ticker", "event_date", "title", "middle_ticker", "rules_primary", "rules_secondary",
             "early_close_condition"]].items()} for i in idx]
        self.dump("rules_sample.json", {
            "note": "rules of the middle rung for 10 events spread over the date range; plus every distinct "
                    "rules_secondary / early_close_condition with its count of events",
            "sample": sample,
            "distinct_rules_secondary": ev["rules_secondary"].fillna("<none>").value_counts().to_dict(),
            "distinct_early_close_condition": ev["early_close_condition"].fillna("<none>").value_counts().to_dict()})

    # ---------------------------------------------------------------- main
    def run(self):
        phases = set(self.cfg.phases.split(","))
        try:
            self.phase_meta()                       # always: the trades phase needs the cutoff (3 requests)
            self.phase_markets()                    # everything else needs the market list
            calls, mins = self.estimate(restrict=False)
            fits = sum(mins[p] for p in ("trades", "candles_1h", "candles_1m") if p in phases) <= self.left() / 60
            restrict = {"always": True, "never": False}.get(self.cfg.restrict_m1, not fits)
            calls_r, mins_r = self.estimate(restrict=True)
            self.summary["estimate"] = {"calls": calls, "minutes": {k: round(v, 1) for k, v in mins.items()},
                                        "minutes_1m_restricted": round(mins_r["candles_1m"], 1),
                                        "budget_left_min": round(self.left() / 60, 1), "full_pull_fits": fits,
                                        "m1_restricted": restrict}
            log(f"estimate: {self.summary['estimate']}")
            if "trades" in phases:
                self.phase_trades()
                self.write_markets()
            if "candles_1h" in phases:
                self.phase_candles_1h()
                self.write_markets()
            if "candles_1m" in phases:
                self.phase_candles_1m(restrict)
        finally:
            s = self.finish()
        return s


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="exp009_out")
    p.add_argument("--series", default=SERIES)
    p.add_argument("--base", default=venues.KALSHI)
    p.add_argument("--rps", type=float, default=9.0, help="max requests per second (all workers together)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget-min", type=float, default=70.0, help="stop starting new work after this many minutes")
    p.add_argument("--phases", default=",".join(PHASES))
    p.add_argument("--tip-offset-h", type=float, default=3.0, help="tip_est = occurrence_datetime - this")
    p.add_argument("--m1-before-tip-min", type=float, default=240.0)
    p.add_argument("--m1-after-start-min", type=float, default=15.0)
    p.add_argument("--near-middle-pts", type=float, default=15.0)
    p.add_argument("--restrict-m1", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--h1-max-days", type=float, default=30.0)
    p.add_argument("--batch-max-candles", type=int, default=BATCH_MAX_CANDLES)
    p.add_argument("--batch-fill", type=float, default=0.9, help="pack batch calls up to this share of the cap")
    p.add_argument("--min-event-date", default=None)
    p.add_argument("--max-event-date", default=None)
    p.add_argument("--max-markets", type=int, default=0, help="debug: newest N markets only")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv=None):
    cfg = build_parser().parse_args(argv)
    if cfg.self_test:
        return self_test()
    log(f"EXP-009a Kalshi history: series {cfg.series}, base {cfg.base}, out {cfg.out}, budget {cfg.budget_min} min")
    s = Runner(cfg, Client(cfg.base, rps=cfg.rps)).run()
    brief = {k: s.get(k) for k in ("elapsed_min", "requests", "retries", "series_fee", "cutoff", "estimate",
                                   "time_range", "counts_by_season", "statuses")}
    brief["phases"] = s.get("phases")
    brief["errors"] = {k: s["errors"][k] for k in ("count", "by_kind")}
    print(json.dumps(brief, indent=1, default=str), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ self-test
class _Resp:
    def __init__(self, status, payload, headers=None):
        self.status_code, self._p, self.headers = status, payload, headers or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._p


class FakeKalshi:
    """Hand-made API: 2 live events (one settled, one future), 1 archived event (only in historical endpoints),
    cursor pagination everywhere, a 429 and a 500 to retry through, legacy cents fields on archived trades."""

    NOW = int(datetime(2026, 11, 23, tzinfo=timezone.utc).timestamp())

    def __init__(self, cap=BATCH_MAX_CANDLES):
        self.cap, self.calls, self.lock = cap, Counter(), threading.Lock()
        self.failed_once = set()
        iso = lambda t: utc(t)  # noqa: E731
        self.markets, self.events = {}, {}

        def ladder(et, strikes, open_ts, tip_ts, src, status, total_pts=None, occ_offset=3 * 3600, vol0=()):
            ev = {"event_ticker": et, "series_ticker": SERIES, "title": "Team A vs Team B: Total Points",
                  "sub_title": "A vs B", "category": "Sports", "strike_date": iso(tip_ts + 3 * 3600),
                  "mutually_exclusive": False, "fee_type_override": None}
            self.events[et] = (ev, src)
            for k in strikes:
                t = f"{et}-{int(k)}"
                close_ts = tip_ts + 2 * 3600 + 900 if status != "active" else tip_ts + 3 * 3600
                res = "" if status == "active" else ("yes" if total_pts > k else "no")
                m = {"ticker": t, "event_ticker": et, "market_type": "binary", "title": ev["title"],
                     "yes_sub_title": f"Over {k} points scored", "no_sub_title": f"Over {k} points scored",
                     "open_time": iso(open_ts), "close_time": iso(close_ts),
                     "expected_expiration_time": iso(tip_ts + 3 * 3600), "occurrence_datetime":
                         iso(tip_ts + occ_offset), "status": status, "result": res, "strike_type": "greater",
                     "floor_strike": k, "rules_primary": f"If total points in A vs B are above {k}, Yes.",
                     "rules_secondary": "Includes overtime.", "early_close_condition": "closes when final",
                     "can_close_early": True}
                if src == "historical":   # archived markets: legacy cents fields
                    m.update({"yes_bid": 0, "yes_ask": 1, "last_price": 99 if res == "yes" else 1,
                              "volume": 0 if k in vol0 else 120, "open_interest": 50,
                              "settlement_value": 100 if res == "yes" else 0})
                else:
                    m.update({"yes_bid_dollars": "0.4500", "yes_ask_dollars": "0.4700",
                              "last_price_dollars": "0.4600", "volume_fp": "0.00" if k in vol0 else "250.00",
                              "open_interest_fp": "80.00",
                              "settlement_value_dollars": None if res == "" else ("1.0000" if res == "yes" else
                                                                                  "0.0000")})
                    if res:
                        m["settlement_ts"] = iso(close_ts + 600)
                self.markets[t] = (m, src)

        tip1 = int(datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc).timestamp())
        ladder("KXNCAAMBTOTAL-26NOV19AAABBB", [130.5, 135.5, 140.5, 145.5, 150.5], tip1 - 2 * 86400, tip1,
               "live", "finalized", total_pts=142, vol0=(130.5,))
        tip2 = int(datetime(2026, 11, 30, 1, 0, tzinfo=timezone.utc).timestamp())
        ladder("KXNCAAMBTOTAL-26NOV29CCCDDD", [131.5, 136.5, 141.5, 146.5, 151.5], self.NOW - 86400, tip2,
               "live", "active")
        tip3 = int(datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc).timestamp())
        ladder("KXNCAAMBTOTAL-26FEB10EEEFFF", [120.5 + 5 * i for i in range(11)], tip3 - 86400, tip3,
               "historical", "finalized", total_pts=151, occ_offset=2 * 3600)

    # ---- helpers
    @staticmethod
    def _page(items, params, default=100):
        lim = int(params.get("limit", default))
        lim = min(lim, 4)                                 # force multi-page pagination in the test
        start = int(params.get("cursor") or 0)
        nxt = start + lim
        return items[start:nxt], (str(nxt) if nxt < len(items) else "")

    def _candles(self, t, s, e, period, fmt):
        m, _ = self.markets[t]
        life_s, life_e = iso_ts(m["open_time"]), iso_ts(m["close_time"])
        step = period * 60
        out = []
        ts = (max(s, life_s) // step + 1) * step
        while ts <= min(e, life_e + step):
            if fmt == "live":
                c = {"end_period_ts": ts,
                     "yes_bid": {f"{k}_dollars": "0.4400" for k in ("open", "high", "low", "close")},
                     "yes_ask": {f"{k}_dollars": "0.4600" for k in ("open", "high", "low", "close")},
                     "price": {"open_dollars": "0.4500", "high_dollars": "0.4500", "low_dollars": "0.4500",
                               "close_dollars": "0.4500", "mean_dollars": "0.4500", "previous_dollars": "0.4500"},
                     "volume_fp": "3.00", "open_interest_fp": "40.00"}
            else:
                c = {"end_period_ts": ts, "yes_bid": {k: "0.3000" for k in ("open", "high", "low", "close")},
                     "yes_ask": {k: "0.3200" for k in ("open", "high", "low", "close")},
                     "price": {"open": None, "high": None, "low": None, "close": None, "mean": None,
                               "previous": "0.3100"}, "volume": "0.00", "open_interest": "12.00"}
            out.append(c)
            ts += step
        return out

    def _trades(self, t, src):
        m, msrc = self.markets[t]
        base = iso_ts(m["open_time"]) + 3600
        rows = []
        for i in range(5):
            if src == "historical":
                rows.append({"trade_id": f"{t}-h{i}", "ticker": t, "created_time": utc(base + 60 * i),
                             "yes_price": 40 + i, "no_price": 60 - i, "count": 10, "taker_side": "yes"})
            else:
                rows.append({"trade_id": f"{t}-l{i}", "ticker": t, "created_time": utc(base + 60 * i)
                             .replace("Z", ".123456Z"), "yes_price_dollars": f"0.{40 + i}00",
                             "no_price_dollars": f"0.{60 - i}00", "count_fp": "10.00", "taker_side": "no",
                             "is_block_trade": False})
        if src == "live":
            rows.insert(2, dict(rows[1]))                 # a duplicate across pages -> must be dropped
        return rows

    # ---- router
    def get(self, url, params=None, headers=None, timeout=None):
        params = params or {}
        path = url.split("/trade-api/v2", 1)[1]
        with self.lock:
            self.calls[path] += 1
            if path == "/historical/cutoff" and "cutoff" not in self.failed_once:
                self.failed_once.add("cutoff")
                return _Resp(429, {"error": "too many requests"}, {"Retry-After": "1"})
            if path == "/markets/trades" and "trades500" not in self.failed_once:
                self.failed_once.add("trades500")
                return _Resp(500, {"error": "boom"})
        if path == f"/series/{SERIES}":
            return _Resp(200, {"series": {"ticker": SERIES, "fee_type": "quadratic", "fee_multiplier": 1,
                                          "title": "College basketball total points"}})
        if path == "/historical/cutoff":
            return _Resp(200, {"market_settled_ts": "2026-06-01T00:00:00Z", "trades_created_ts":
                               "2026-06-01T00:00:00Z", "orders_updated_ts": "2026-06-01T00:00:00Z"})
        if path == "/series/fee_changes":
            return _Resp(200, {"series_fee_change_arr": []})
        if path == "/events":
            if params.get("tickers"):
                want = params["tickers"].split(",")
                evs = [self.events[t][0] for t in want if t in self.events]
            else:
                evs = [e for e, src in self.events.values() if src == "live"]
            page, cur = self._page(evs, params)
            return _Resp(200, {"events": page, "cursor": cur})
        if path in ("/markets", "/historical/markets"):
            src = "live" if path == "/markets" else "historical"
            ms = [m for m, s in self.markets.values() if s == src and m["event_ticker"].startswith(
                params.get("series_ticker", ""))]
            page, cur = self._page(ms, params)
            return _Resp(200, {"markets": page, "cursor": cur})
        if path in ("/markets/trades", "/historical/trades"):
            t = params["ticker"]
            src = "live" if path == "/markets/trades" else "historical"
            rows = self._trades(t, src) if self.markets[t][1] == src else []
            page, cur = self._page(rows, params)
            return _Resp(200, {"trades": page, "cursor": cur})
        if path == "/markets/candlesticks":
            ts = params["market_tickers"].split(",")
            assert len(ts) <= BATCH_MAX_TICKERS
            if any(self.markets[t][1] != "live" for t in ts):
                return _Resp(400, {"error": "historical market in batch"})
            out, n = [], 0
            for t in ts:
                cs = self._candles(t, int(params["start_ts"]), int(params["end_ts"]),
                                   int(params["period_interval"]), "live")
                cs = cs[:max(0, self.cap - n)]
                n += len(cs)
                out.append({"market_ticker": t, "candlesticks": cs})
            return _Resp(200, {"markets": out})
        if path.startswith(f"/series/{SERIES}/markets/") and path.endswith("/candlesticks"):
            t = path.split("/")[4]
            if self.markets[t][1] != "live":
                return _Resp(404, {"error": "not found"})
            return _Resp(200, {"ticker": t, "candlesticks": self._candles(
                t, int(params["start_ts"]), int(params["end_ts"]), int(params["period_interval"]), "live")})
        if path.startswith("/historical/markets/") and path.endswith("/candlesticks"):
            t = path.split("/")[3]
            if self.markets[t][1] != "historical":
                return _Resp(404, {"error": "not found"})
            return _Resp(200, {"ticker": t, "candlesticks": self._candles(
                t, int(params["start_ts"]), int(params["end_ts"]), int(params["period_interval"]), "hist")})
        return _Resp(404, {"error": f"no route {path}"})


def self_test():
    class FakeClock:
        def __init__(self):
            self.t, self.lock = 1000.0, threading.Lock()

        def __call__(self):
            with self.lock:
                return self.t

        def sleep(self, d):
            with self.lock:
                self.t += max(0.0, d)

    def run(tmp, cap=BATCH_MAX_CANDLES, **over):
        FIELD_UNITS.clear()
        clock = FakeClock()
        fake = FakeKalshi(cap=cap)
        argv = ["--out", str(tmp), "--rps", "50", "--workers", "3"] + sum(
            ([f"--{k.replace('_', '-')}", str(v)] for k, v in over.items()), [])
        cfg = build_parser().parse_args(argv)
        client = Client(cfg.base, session=fake, rps=cfg.rps, sleep=clock.sleep, clock=clock)
        s = Runner(cfg, client, clock=clock, now=FakeKalshi.NOW).run()
        return s, fake

    tmp = Path(tempfile.mkdtemp(prefix="exp009_selftest_"))
    try:
        # 1) full run, 1-min restricted to near-middle rungs, small candle cap to force batch splitting
        s, fake = run(tmp / "a", cap=150, restrict_m1="always", batch_max_candles=150, batch_fill=3.0)
        o = tmp / "a"
        for f in ("series.json", "cutoff.json", "series_fee_changes.json", "summary.json", "rules_sample.json",
                  "events.parquet", "markets.parquet", "trades.parquet", "candles_1h.parquet",
                  "candles_1m.parquet"):
            assert (o / f).exists(), f"missing {f}"
        assert not list(o.glob("_parts_*")), "part folders left behind"
        mk, ev = pd.read_parquet(o / "markets.parquet"), pd.read_parquet(o / "events.parquet")
        tr = pd.read_parquet(o / "trades.parquet")
        h1, m1 = pd.read_parquet(o / "candles_1h.parquet"), pd.read_parquet(o / "candles_1m.parquet")
        assert len(mk) == 21 and mk.ticker.is_unique, len(mk)
        assert (mk.source == "historical").sum() == 11 and (mk.source == "live").sum() == 10
        assert len(ev) == 3 and ev.event_found.all(), ev[["event_ticker", "event_found"]]
        assert ev.set_index("event_ticker").at["KXNCAAMBTOTAL-26FEB10EEEFFF", "in_events_list"] == False  # noqa
        e1 = ev.set_index("event_ticker").loc["KXNCAAMBTOTAL-26NOV19AAABBB"]
        assert e1.middle_strike == 140.5 and str(e1.start_utc_venues) == "2026-11-20 03:00:00+00:00"
        assert str(e1.tip_est_utc) == "2026-11-20 00:00:00+00:00"
        hist_m = mk[mk.source == "historical"].iloc[0]
        assert hist_m.yes_ask == 0.01 and hist_m.settlement_value in (0.0, 1.0), "legacy cents not converted"
        assert mk.loc[mk.ticker == "KXNCAAMBTOTAL-26NOV19AAABBB-130", "trades_status"].item() == \
            "skipped_zero_volume"
        # trades: 4 live markets with volume x 5 + 11 historical x 5 + 5 future-event live markets x 5
        assert tr.trade_id.is_unique and len(tr) == (4 + 5) * 5 + 11 * 5, len(tr)
        assert s["phases"]["trades"]["duplicates_dropped"] == 9, s["phases"]["trades"]
        assert tr.yes_price.between(0, 1).all() and tr.created_time.notna().all()
        assert set(tr[tr.source == "historical"].yes_price.round(2)) == {0.40, 0.41, 0.42, 0.43, 0.44}
        assert (tr["count"] == 10).all()
        # hourly candles inside market lifetimes
        lim = mk.set_index("ticker")
        j = h1.join(lim[["open_time", "close_time"]], on="ticker")
        assert (j.end_period_utc > j.open_time).all()
        assert (j.end_period_utc <= j.close_time + pd.Timedelta(hours=1)).all()
        assert set(h1.ticker) == set(mk.ticker), "every market should have hourly candles in the mock"
        assert h1.groupby(["ticker", "end_period_ts"]).size().max() == 1, "duplicate hourly candles"
        assert set(h1.source) == {"live", "historical"}
        assert h1[h1.source == "live"].yes_bid_close.eq(0.44).all()
        assert h1[h1.source == "historical"].yes_ask_close.eq(0.32).all()
        assert h1[h1.source == "historical"].price_close.isna().all()
        # 1-minute candles: future event has no window; far rungs of the archived event skipped
        m1s = mk.set_index("ticker").m1_status
        assert (m1s[m1s.index.str.contains("CCCDDD")] == "no_window").all(), m1s
        far = mk[(mk.dist_from_middle > 15)]
        assert len(far) == 4 and (far.m1_status == "skipped_far_rung").all()
        assert set(m1.ticker) == set(mk[(mk.dist_from_middle <= 15) & ~mk.ticker.str.contains("CCCDDD")].ticker)
        j = m1.join(lim[["m1_window_start", "m1_window_end", "start_utc_venues"]], on="ticker")
        assert (j.end_period_utc > j.m1_window_start).all() and \
            (j.end_period_utc <= j.m1_window_end + pd.Timedelta(minutes=1)).all()
        assert ((j.start_utc_venues - j.end_period_utc) <= pd.Timedelta(hours=7)).all()
        assert m1.groupby(["ticker", "end_period_ts"]).size().max() == 1, "duplicate 1-min candles"
        # retries + summary
        assert s["retries"] >= 2 and s["errors"]["count"] == 0, s["errors"]
        assert s["phases"]["candles_1h"]["batch_splits"] > 0 and s["phases"]["candles_1m"]["batch_splits"] > 0
        assert s["series_fee"]["fee_type"] == "quadratic"
        assert {r["season"] for r in s["counts_by_season"]} == {2026, 2027}
        assert sum(r["trades"] for r in s["counts_by_event_date"]) == len(tr)
        assert s["field_units"]["trades.historical.price"] == ["yes_price (int cents /100)"]
        rules = json.loads((o / "rules_sample.json").read_text())
        assert rules["distinct_rules_secondary"] == {"Includes overtime.": 3}

        # 2) zero budget: markets still written, every later phase reports not reached
        s2, _ = run(tmp / "b", budget_min=0)
        mk2 = pd.read_parquet(tmp / "b" / "markets.parquet")
        assert len(mk2) == 21 and set(mk2.trades_status) <= {"not_reached", "skipped_zero_volume"}
        assert s2["phases"]["trades"]["complete"] is False and s2["phases"]["candles_1m"]["done"] == 0
        assert pd.read_parquet(tmp / "b" / "candles_1h.parquet").empty

        # 3) phases subset + auto restriction off when everything fits
        s3, fake3 = run(tmp / "c", phases="meta,markets,candles_1h")
        assert "trades" not in s3["phases"] and s3["phases"]["candles_1h"]["complete"]
        assert s3["estimate"]["m1_restricted"] is False
        assert fake3.calls["/markets/trades"] == 0
        print("self-test OK:", json.dumps({"requests": s["requests"], "retries": s["retries"],
                                            "trades": len(tr), "candles_1h": len(h1), "candles_1m": len(m1)}))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
