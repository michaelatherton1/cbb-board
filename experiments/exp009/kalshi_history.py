"""EXP-009a: collect Kalshi's history for college basketball TOTALS markets (series KXNCAAMBTOTAL).

Data collection only: no pricing or profitability analysis here.

Runs in GitHub Actions (.github/workflows/exp009-kalshi-history.yml); the dev workspace cannot reach Kalshi.
Public Trade API v2 endpoints, no auth; base URL and User-Agent come from cbb/venues.py. Two modes:

  --mode plan    meta + every event/market (live GET /markets + archived GET /historical/markets).
                 Writes the internal plan (--plan dir, passed to the shards as an Actions artifact) and the
                 committed meta files under <data-out>/kalshi_totals/meta/.
  --mode shard   --shard k --n-shards N: the events with crc32(event_ticker) % N == k. Trades, then hourly
                 candles, then 1-minute candles, under a wall-clock budget. Writes compact files under
                 <data-out>/kalshi_totals/<season>/shard_<k>/.

Scope (defaults = the lead's EXP-009a spec):
  live markets (this season):      all trades; 1h candles all rungs; 1m candles for rungs within 10 pts of the
                                   event's middle rung (median strike of its ladder)
  historical markets (last season): all trades; 1h candles for rungs within 10 pts of the middle; no 1m candles

Committed layout (every parquet is zstd and < --max-file-mb):
  kalshi_totals/meta/        series.json, series_fee_changes.json, cutoff.json, plan_summary.json,
                             markets.parquet, events.parquet, rules_sample.json, markets_raw_sample.json
  kalshi_totals/<season>/shard_<k>/
                             trades_<YYYY-MM>[_pN].parquet, candles_1h_<YYYY-MM>.parquet,
                             candles_1m_<YYYY-MM>.parquet (YYYY-MM = event month, so a market's rows sit in one file),
                             fetch_status.parquet (per market: counts + ok/error/not_reached/out_of_scope/...),
                             summary.json

Units: prices in dollars (0-1); `<x>_dollars` fields are used when present, legacy integer cents are /100 and
historical dollar strings are parsed. Contract counts use `<x>_fp` when present. summary.json["field_units"] says
which source field was seen.

Tip time: Kalshi has no tip-off field. cbb/venues.py uses the market's `occurrence_datetime` as start_utc; for open
markets that is tip + 3h (49/50 games on 2026-11-23 vs ESPN), and archived markets carry none. events.parquet keeps
`start_utc_venues` (= occurrence_datetime) and `tip_est_utc` (= start_utc_venues - 3h). Because it is not certain
that occurrence_datetime keeps that meaning after a game is final, the 1-minute window is
[start_utc_venues - 7h, start_utc_venues + 15m] clamped to the market's life: it covers tip-4h .. tip+15m whether
occurrence_datetime is the tip or anything up to tip + 3h. Trim with tip_est_utc once the offset is confirmed.

Self-test (offline, mocked API):  python experiments/exp009/kalshi_history.py --self-test
"""
from __future__ import annotations

import argparse, json, math, random, shutil, sys, tempfile, threading, time, traceback, zlib
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
BATCH_MAX_TICKERS = 100          # GET /markets/candlesticks: up to 100 tickers ...
BATCH_MAX_CANDLES = 10_000       # ... and up to 10,000 candles per call
SINGLE_MAX_PERIODS = 4_000       # per-market candle requests are chunked to at most this many periods
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
INF = float("inf")


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
    if not s or (isinstance(s, float) and s != s):
        return None
    try:
        return int(pd.Timestamp(s).timestamp())
    except (ValueError, TypeError):
        return None


def utc(ts):
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def season_of(d):
    """CBB season of a date, as the repo numbers it: 2025-26 -> 2026."""
    if d is None or (not isinstance(d, str) and pd.isna(d)):
        return None
    d = pd.Timestamp(d)
    return d.year + 1 if d.month >= 8 else d.year


def season_label(season):
    return f"{season - 1}-{str(season)[-2:]}"          # 2026 -> '2025-26'


def shard_of(event_ticker, n):
    return zlib.crc32(str(event_ticker).encode()) % n


# ------------------------------------------------------------------------------------------------ HTTP
class Client:
    """Rate-limited GET with retries/backoff (429, 5xx, network errors). Never raises; returns None on failure.
    A 429 also slows every worker down (rate x 2/3, floor 1 req/s); the rate creeps back after successes."""

    def __init__(self, base, session=None, rps=8.0, max_retries=7, sleep=time.sleep, clock=time.monotonic):
        self.base, self.s = base.rstrip("/"), session or requests.Session()
        self.min_gap = self.gap = 1.0 / rps
        self.max_retries, self.sleep, self.clock = max_retries, sleep, clock
        self._lock, self._next, self._ok_streak = threading.Lock(), 0.0, 0
        self.n_requests = self.n_retries = 0
        self.retry_reasons, self.errors, self.err_counts = Counter(), [], Counter()

    def _slot(self):
        with self._lock:
            now = self.clock()
            t = max(now, self._next)
            self._next = t + self.gap
            self.n_requests += 1
        if t > now:
            self.sleep(t - now)

    def _backoff(self, delay, reason):
        with self._lock:            # every worker waits after a 429/5xx, not just this one
            self._next = max(self._next, self.clock() + delay)
            self.n_retries += 1
            self.retry_reasons[reason] += 1
            if reason == "429":
                self.gap = min(1.0, self.gap * 1.5)
                self._ok_streak = 0
        self.sleep(delay)

    def _success(self):
        with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 200 and self.gap > self.min_gap:
                self.gap = max(self.min_gap, self.gap * 0.9)
                self._ok_streak = 0

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
            retry_after, reason = 0.0, None
            try:
                r = self.s.get(self.base + path, params=params, headers=venues.UA, timeout=45)
            except requests.RequestException as e:
                status, msg, reason = None, f"{type(e).__name__}: {e}", type(e).__name__
            else:
                status = r.status_code
                if status == 200:
                    try:
                        j = r.json()
                    except ValueError as e:
                        msg, reason = f"bad json: {e}", "bad_json"
                    else:
                        self._success()
                        return j
                elif status in RETRY_STATUS:
                    msg, reason = (r.text or "")[:200], str(status)
                    retry_after = _f((r.headers or {}).get("Retry-After")) or 0.0
                else:
                    self.error(f"http_{status}", path, params, status, (r.text or "")[:300])
                    return None
            if attempt < self.max_retries:
                delay = max(retry_after, min(60.0, 0.75 * 2 ** attempt)) * (1 + 0.25 * random.random())
                self._backoff(delay, reason)
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


# ------------------------------------------------------------------------------------------------ rows / sinks
OHLC = [f"{s}_{k}" for s in ("yes_bid", "yes_ask", "price") for k in ("open", "high", "low", "close")]
CANDLES_OUT = pa.schema([("ticker", pa.string()), ("event_ticker", pa.string()), ("end_period_ts", pa.int64())] +
                        [(c, pa.float64()) for c in OHLC] +
                        [("volume", pa.float64()), ("open_interest", pa.float64())])
TRADES_OUT = pa.schema([("trade_id", pa.string()), ("ticker", pa.string()), ("event_ticker", pa.string()),
                        ("created_time", pa.timestamp("us", tz="UTC")), ("yes_price", pa.float64()),
                        ("no_price", pa.float64()), ("count", pa.float64()), ("taker_side", pa.string()),
                        ("is_block_trade", pa.bool_())])


def _prep_candles(df):
    for c in OHLC + ["volume", "open_interest"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cents = df.pop("_cents").astype(bool)
    if cents.any():
        df.loc[cents, OHLC] = df.loc[cents, OHLC] / 100
    df["end_period_ts"] = pd.to_numeric(df["end_period_ts"], errors="coerce").astype("int64")
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
    """Rows -> parquet part files in a scratch folder (each valid on its own); read back by table()."""

    def __init__(self, work_dir, name, schema, prep, flush_rows=250_000):
        self.dir = Path(work_dir) / f"_parts_{name}"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.schema, self.prep, self.flush_rows = schema, prep, flush_rows
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

    def table(self):
        with self.lock:
            self._flush()
        parts = sorted(self.dir.glob("part-*.parquet"))
        return pa.concat_tables([pq.read_table(p, schema=self.schema) for p in parts]) if parts \
            else self.schema.empty_table()


def candle_rows(cands, ticker, event_ticker, source):
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
        out.append({"ticker": ticker, "event_ticker": event_ticker, "end_period_ts": x.get("end_period_ts"),
                    "_cents": cents,
                    "yes_bid_open": yb.get(o), "yes_bid_high": yb.get(h), "yes_bid_low": yb.get(l),
                    "yes_bid_close": yb.get(c),
                    "yes_ask_open": ya.get(o), "yes_ask_high": ya.get(h), "yes_ask_low": ya.get(l),
                    "yes_ask_close": ya.get(c),
                    "price_open": p.get(o), "price_high": p.get(h), "price_low": p.get(l), "price_close": p.get(c),
                    "volume": x.get(vk), "open_interest": x.get(ok)})
    return out


def trade_rows(trades, event_ticker, source, extra):
    out = []
    for t in trades:
        dollars = t.get("yes_price_dollars") is not None
        cents = not dollars and isinstance(t.get("yes_price"), int)
        note(f"trades.{source}.price", "yes_price_dollars" if dollars else
             ("yes_price (int cents /100)" if cents else "yes_price (dollars)"))
        ck = "count_fp" if t.get("count_fp") is not None else "count"
        note(f"trades.{source}.count", ck)
        tos = t.get("taker_outcome_side")
        if tos is not None and tos != t.get("taker_side"):
            extra["taker_outcome_side_differs"] += 1
        if t.get("taker_book_side") is not None:
            extra["has_taker_book_side"] += 1
        out.append({"trade_id": t.get("trade_id"), "ticker": t.get("ticker"), "event_ticker": event_ticker,
                    "created_time": t.get("created_time"), "_cents": cents,
                    "yes_price": t.get("yes_price_dollars") if dollars else t.get("yes_price"),
                    "no_price": t.get("no_price_dollars") if dollars else t.get("no_price"),
                    "count": t.get(ck), "taker_side": t.get("taker_side"), "is_block_trade": t.get("is_block_trade")})
    return out


MARKET_TIMES = ["created_time", "updated_time", "open_time", "close_time", "expected_expiration_time",
                "expiration_time", "latest_expiration_time", "occurrence_datetime", "settlement_ts",
                "fee_waiver_expiration_time"]
MARKET_TEXT = ["ticker", "event_ticker", "market_type", "title", "subtitle", "yes_sub_title", "no_sub_title",
               "strike_type", "status", "result", "early_close_condition", "expiration_value",
               "settlement_bounds_type"]


def market_row(m, source):
    r = {k: m.get(k) for k in MARKET_TEXT}
    r["source"] = source
    for k in MARKET_TIMES:
        r[k] = m.get(k)
    for k in ("floor_strike", "cap_strike", "settlement_timer_seconds"):
        r[k] = _f(m.get(k))
    r["can_close_early"] = m.get("can_close_early")
    r["is_provisional"] = m.get("is_provisional")
    for k in ("yes_bid", "yes_ask", "no_bid", "no_ask", "last_price", "previous_yes_bid", "previous_yes_ask",
              "previous_price", "settlement_value", "notional_value"):
        r[k] = price(m, k)
    for k in ("volume", "volume_24h", "open_interest"):
        r[k] = qty(m, k)
    return r


def write_capped(tbl, folder, name, max_mb):
    """Write tbl to folder/name.parquet (zstd); if it is over max_mb split it into name_p1..pN."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    pq.write_table(tbl, path, compression="zstd", compression_level=9)
    size = path.stat().st_size
    if size <= max_mb * 1e6 or tbl.num_rows < 2:
        return [path]
    path.unlink()
    n = math.ceil(size / (max_mb * 1e6 * 0.85))
    step = math.ceil(tbl.num_rows / n)
    out = []
    for i in range(n):
        out += write_capped(tbl.slice(i * step, step), folder, f"{name}_p{i + 1}", max_mb)
    return out


# ------------------------------------------------------------------------------------------------ runner
class Runner:
    def __init__(self, cfg, client, clock=time.monotonic, now=None):
        self.cfg, self.c, self.clock = cfg, client, clock
        self.t0 = clock()
        self.deadline = self.t0 + cfg.budget_min * 60
        self.now = int(now if now is not None else time.time())
        self.data = Path(cfg.data_out) / "kalshi_totals"
        self.plan_dir = Path(cfg.plan)
        self.work = Path(cfg.work)
        self.lock = threading.Lock()
        self.stat = defaultdict(dict)          # ticker -> {n_trades, trades_status, ...}
        self.seen_trades, self.dupes = set(), 0
        self.batch_splits = 0
        self.extra = Counter()
        self.phases = {}

    # ---------------------------------------------------------------- helpers
    def left(self):
        return self.deadline - self.clock()

    @staticmethod
    def dump(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))

    def set_stat(self, ticker, **kw):
        with self.lock:
            self.stat[ticker].update(kw)

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
                if self.clock() - last > 60 or (not futs and done == n):
                    el = self.clock() - t_start
                    rate = done / el if el > 0 else 0
                    eta = (n - done) / rate / 60 if rate else float("nan")
                    log(f"{name}: {done}/{n} done, {self.c.n_requests} requests, {self.c.n_retries} retries "
                        f"{dict(self.c.retry_reasons)}, {rate:.2f} items/s, ETA {eta:.1f} min, "
                        f"budget left {self.left() / 60:.1f} min")
                    last = self.clock()
        return {"items": n, "done": done, "not_started": n - started, "complete": started == n}

    def _safe(self, name, fn, item):
        try:
            fn(item)
        except Exception as e:  # a parsing bug in one item must not kill the run
            self.c.error(f"exception_{name}", name, {"item": str(item)[:200]}, None,
                         f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-400:]}")

    def run_info(self):
        return {"started_utc": utc(self.now), "elapsed_min": round((self.clock() - self.t0) / 60, 2),
                "requests": self.c.n_requests, "retries": self.c.n_retries,
                "retry_reasons": dict(self.c.retry_reasons),
                "errors": {"count": sum(self.c.err_counts.values()), "by_kind": dict(self.c.err_counts),
                           "first": self.c.errors[:50]},
                "field_units": {k: sorted(v) for k, v in sorted(FIELD_UNITS.items())},
                "config": {k: v for k, v in vars(self.cfg).items() if k != "self_test"}}

    # ================================================================ PLAN
    def plan(self):
        S = self.cfg.series
        meta = self.data / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        series = self.c.get(f"/series/{S}", {"include_volume": "true"})
        cutoff = self.c.get("/historical/cutoff")
        fees = self.c.get("/series/fee_changes", {"series_ticker": S, "show_historical": "true"})
        for name, obj in (("series.json", series), ("cutoff.json", cutoff), ("series_fee_changes.json", fees)):
            self.dump(meta / name, obj if obj is not None else {"error": "fetch failed"})
        self.dump(self.plan_dir / "cutoff.json", cutoff or {})
        s = (series or {}).get("series") or {}
        log(f"meta: fee_type {s.get('fee_type')} x{s.get('fee_multiplier')}, cutoff {cutoff}")

        evs, ev_ok = self.c.paginate("/events", {"series_ticker": S, "limit": 200}, "events")
        live, live_ok = self.c.paginate("/markets", {"series_ticker": S, "limit": 1000}, "markets")
        hist, hist_ok = self.c.paginate("/historical/markets", {"series_ticker": S, "limit": 1000}, "markets")
        log(f"markets: {len(evs)} events, {len(live)} live markets, {len(hist)} historical markets "
            f"(complete: {ev_ok}, {live_ok}, {hist_ok})")
        rows, raw, overlap = {}, {}, 0
        for src, ms in (("live", live), ("historical", hist)):
            for m in ms:
                t = m.get("ticker")
                if not t:
                    continue
                if t in rows:
                    overlap += 1
                    continue
                rows[t], raw[t] = market_row(m, src), m
        mk = pd.DataFrame(list(rows.values()))
        mk["event_date"] = mk["event_ticker"].map(venues._kalshi_date)
        if self.cfg.max_markets:
            mk = mk.sort_values("event_date", ascending=False).head(self.cfg.max_markets)
        mk["season"] = mk["event_date"].map(season_of).astype("Int64")
        fl, cp = pd.to_numeric(mk["floor_strike"]), pd.to_numeric(mk["cap_strike"])
        strike = fl.where(fl.notna(), cp)
        mk["strike"] = strike.where(~(fl.notna() & cp.notna()), (fl + cp) / 2)     # 'between' rungs: midpoint
        mid = mk.groupby("event_ticker")["strike"].agg(
            lambda x: float(pd.Series(x.dropna().unique()).median()) if x.notna().any() else math.nan)
        mk["middle_strike"] = mk["event_ticker"].map(mid)
        mk["dist_from_middle"] = (mk["strike"] - mk["middle_strike"]).abs()
        mk["rung_rank"] = (mk.groupby("event_ticker")["dist_from_middle"].rank(method="first").fillna(999) - 1) \
            .astype(int)
        mk["n_rungs_event"] = mk.groupby("event_ticker")["ticker"].transform("count")
        mk["shard"] = mk["event_ticker"].map(lambda e: shard_of(e, self.cfg.n_shards))

        ev_by = {e.get("event_ticker"): e for e in evs if e.get("event_ticker")}
        missing = sorted(set(mk["event_ticker"].dropna()) - set(ev_by))
        for i in range(0, len(missing), 100):
            chunk = missing[i:i + 100]
            more, _ = self.c.paginate("/events", {"tickers": ",".join(chunk), "limit": 200}, "events")
            ev_by.update({e["event_ticker"]: e for e in more if e.get("event_ticker") in chunk})
        ev = self._events(mk, ev_by, raw, {e.get("event_ticker") for e in evs})
        mk = mk.merge(ev[["event_ticker", "start_utc_venues", "tip_est_utc"]], on="event_ticker", how="left")

        # internal plan for the shards (unix-second columns for windows)
        pl = mk.copy()
        for k in ("open_time", "close_time", "settlement_ts"):
            pl[k + "_s"] = pd.to_numeric(pl[k].map(iso_ts), errors="coerce")
        pl["start_s"] = pd.to_numeric(pl["start_utc_venues"].map(iso_ts), errors="coerce")
        pl[["ticker", "event_ticker", "source", "event_date", "season", "volume", "open_time_s", "close_time_s",
            "settlement_ts_s", "start_s", "strike", "dist_from_middle", "rung_rank", "shard"]] \
            .to_parquet(self.plan_dir / "markets_plan.parquet", index=False)

        # committed meta files
        out = mk.copy()
        for k in MARKET_TIMES + ["start_utc_venues", "tip_est_utc"]:
            out[k] = pd.to_datetime(out[k], utc=True, errors="coerce", format="ISO8601")
        out = out.drop(columns=["early_close_condition"])          # one copy per event, in events.parquet
        out.sort_values(["event_date", "event_ticker", "strike"]).to_parquet(
            meta / "markets.parquet", index=False, compression="zstd")
        evo = ev.copy()
        for k in ("start_utc_venues", "tip_est_utc", "first_open_time", "last_close_time"):
            evo[k] = pd.to_datetime(evo[k], utc=True, errors="coerce", format="ISO8601")
        evo.sort_values(["event_date", "event_ticker"]).to_parquet(meta / "events.parquet", index=False,
                                                                     compression="zstd")
        self._rules(ev, meta)
        sample = [raw[t] for t in list(mk[mk.source == "live"].ticker[:10]) + list(mk[mk.source == "historical"]
                                                                                  .ticker[:10])]
        self.dump(meta / "markets_raw_sample.json", sample)

        est = self.estimate(mk_plan=pl)
        summary = {"series_fee": {k: s.get(k) for k in ("fee_type", "fee_multiplier")},
                   "series_fee_changes": (fees or {}).get("series_fee_change_arr"), "cutoff": cutoff,
                   "markets": {"events_endpoint": len(evs), "events_endpoint_complete": ev_ok,
                               "live_markets": len(live), "live_complete": live_ok,
                               "historical_markets": len(hist), "historical_complete": hist_ok,
                               "overlap_live_and_historical": overlap, "markets_kept": int(len(mk)),
                               "events_kept": int(len(ev)), "events_only_seen_via_markets": len(missing)},
                   "by_season": json.loads(
                       mk.groupby("season").agg(events=("event_ticker", "nunique"), markets=("ticker", "size"),
                                                historical=("source", lambda x: int((x == "historical").sum())),
                                                with_result=("result", lambda x: int(x.isin(["yes", "no"]).sum())),
                                                first_date=("event_date", "min"), last_date=("event_date", "max"),
                                                volume=("volume", "sum"))
                       .reset_index().to_json(orient="records", date_format="iso")),
                   "status": {str(k): int(v) for k, v in mk["status"].value_counts().items()},
                   "result": {str(k): int(v) for k, v in mk["result"].value_counts().items()},
                   "shards": est, "run": self.run_info()}
        self.dump(meta / "plan_summary.json", summary)
        log(f"plan: {len(mk)} markets in {len(ev)} events; per-shard estimate {json.dumps(est)}")
        return summary

    def _events(self, mk, ev_by, raw, from_list):
        rows = []
        for et, g in mk.groupby("event_ticker"):
            e = ev_by.get(et) or {}
            occ = g["occurrence_datetime"].dropna()
            start = occ.mode().iloc[0] if len(occ) else None
            mid_t = g.sort_values("rung_rank")["ticker"].iloc[0]
            rm = raw.get(mid_t, {})
            tip = (pd.Timestamp(start) - pd.Timedelta(hours=self.cfg.tip_offset_h)) if start else None
            rows.append({
                "event_ticker": et, "title": e.get("title"), "sub_title": e.get("sub_title"),
                "strike_date": e.get("strike_date"), "event_date": g["event_date"].iloc[0],
                "season": season_of(g["event_date"].iloc[0]), "shard": int(g["shard"].iloc[0]),
                "source": ",".join(sorted(set(g["source"]))),
                "start_utc_venues": start, "n_distinct_occurrence": int(occ.nunique()),
                "tip_est_utc": None if tip is None else tip.isoformat().replace("+00:00", "Z"),
                "n_markets": int(len(g)), "min_strike": _f(g["strike"].min()), "max_strike": _f(g["strike"].max()),
                "middle_strike": _f(g["strike"].median() if g["strike"].notna().any() else None),
                "first_open_time": g["open_time"].dropna().min() if g["open_time"].notna().any() else None,
                "last_close_time": g["close_time"].dropna().max() if g["close_time"].notna().any() else None,
                "statuses": ",".join(sorted(set(g["status"].dropna().astype(str)))),
                "fee_type_override": None if e.get("fee_type_override") is None else str(e.get("fee_type_override")),
                "fee_multiplier_override": _f(e.get("fee_multiplier_override")),
                "middle_ticker": mid_t, "rules_primary": rm.get("rules_primary"),
                "rules_secondary": rm.get("rules_secondary"), "early_close_condition": rm.get("early_close_condition"),
                "in_events_list": et in from_list, "event_found": bool(e)})
        ev = pd.DataFrame(rows)
        ev["middle_strike"] = ev["event_ticker"].map(mk.groupby("event_ticker")["middle_strike"].first())
        return ev

    def _rules(self, ev, meta):
        ev = ev.sort_values("event_date")
        idx = sorted({int(round(i * (len(ev) - 1) / 9)) for i in range(10)}) if len(ev) > 1 else [0]
        sample = [{k: str(v) if k == "event_date" else v for k, v in ev.iloc[i][
            ["event_ticker", "event_date", "title", "middle_ticker", "rules_primary", "rules_secondary",
             "early_close_condition"]].items()} for i in idx]
        self.dump(meta / "rules_sample.json", {
            "note": "rules of the middle rung for 10 events spread over the date range; plus every distinct "
                    "rules_secondary / early_close_condition with its count of events",
            "sample": sample,
            "distinct_rules_secondary": ev["rules_secondary"].fillna("<none>").value_counts().to_dict(),
            "distinct_early_close_condition": ev["early_close_condition"].fillna("<none>").value_counts().to_dict()})

    # ================================================================ SHARD
    def scope(self, mk):
        cfg = self.cfg
        live, hist = mk["source"] == "live", mk["source"] == "historical"
        d = mk["dist_from_middle"].fillna(INF)
        h1 = (live & (d <= cfg.live_h1_pts)) | (hist & (d <= cfg.hist_h1_pts))
        m1 = (live & (d <= cfg.live_m1_pts)) | (hist & (d <= cfg.hist_m1_pts))
        return h1, m1

    def windows_1h(self, mk):
        start = mk["open_time_s"]
        end = pd.concat([mk["close_time_s"], mk["settlement_ts_s"]], axis=1).min(axis=1)
        end = end.where(end.notna(), self.now).clip(upper=self.now)
        end = (end // 3600 + 1) * 3600                       # include the candle that contains the close
        start = start.where(start.notna(), end - 3 * 86400)
        start = pd.concat([start, end - self.cfg.h1_max_days * 86400], axis=1).max(axis=1)
        return start, end

    def windows_1m(self, mk):
        cfg = self.cfg
        ws = mk["start_s"] - int((cfg.tip_offset_h * 60 + cfg.m1_before_tip_min) * 60)
        we = mk["start_s"] + int(cfg.m1_after_start_min * 60)
        life_e = mk["close_time_s"].where(mk["close_time_s"].notna(), self.now).clip(upper=self.now)
        ws = pd.concat([ws, mk["open_time_s"]], axis=1).max(axis=1)
        we = pd.concat([we, life_e], axis=1).min(axis=1)
        return ws.where(mk["start_s"].notna()), we.where(mk["start_s"].notna())

    def estimate(self, mk_plan):
        """Rough requests per shard: trades ~1.1/market with volume; 1h: live batches + 1/historical market;
        1m: 1/live event."""
        h1, m1 = self.scope(mk_plan)
        out = {}
        for k, g in mk_plan.groupby("shard"):
            hv = g["volume"].isna() | (g["volume"] > 0)
            gl = g[h1[g.index] & (g.source == "live")]
            calls = {"trades": int(hv.sum() * 1.1),
                     "candles_1h": int(gl["event_ticker"].nunique() / 5 + 1 +
                                       (h1[g.index] & (g.source == "historical")).sum()),
                     "candles_1m": int(g[m1[g.index]]["event_ticker"].nunique())}
            out[int(k)] = {**calls, "est_minutes_at_6rps": round(sum(calls.values()) / 6 / 60, 1)}
        return out

    def shard(self):
        cfg, k, n = self.cfg, self.cfg.shard, self.cfg.n_shards
        mk = pd.read_parquet(self.plan_dir / "markets_plan.parquet")
        cutoff = json.loads((self.plan_dir / "cutoff.json").read_text())
        mk = mk[mk["event_ticker"].map(lambda e: shard_of(e, n)) == k].reset_index(drop=True)
        self.mk = mk
        log(f"shard {k}/{n}: {len(mk)} markets in {mk.event_ticker.nunique()} events "
            f"({int((mk.source == 'historical').sum())} historical)")
        h1_in, m1_in = self.scope(mk)
        for t in mk.loc[~h1_in, "ticker"]:
            self.set_stat(t, h1_status="out_of_scope")
        for t in mk.loc[~m1_in, "ticker"]:
            self.set_stat(t, m1_status="out_of_scope")
        self.work.mkdir(parents=True, exist_ok=True)
        sinks = {}
        try:
            if "trades" in cfg.phases:
                sinks["trades"] = PartSink(self.work, "trades", TRADES_OUT, _prep_trades)
                self.phase_trades(mk, iso_ts(cutoff.get("trades_created_ts")), sinks["trades"])
            if "candles_1h" in cfg.phases:
                ws, we = self.windows_1h(mk)
                sinks["candles_1h"] = PartSink(self.work, "candles_1h", CANDLES_OUT, _prep_candles)
                self.phases["candles_1h"] = self.phase_candles("candles_1h", 60, mk[h1_in], ws, we,
                                                               sinks["candles_1h"], "n_candles_1h", "h1_status")
            if "candles_1m" in cfg.phases:
                ws, we = self.windows_1m(mk)
                for i in mk.index[m1_in & ws.notna()]:
                    self.set_stat(mk.at[i, "ticker"], m1_window_start=int(ws[i]), m1_window_end=int(we[i]))
                sinks["candles_1m"] = PartSink(self.work, "candles_1m", CANDLES_OUT, _prep_candles)
                res = self.phase_candles("candles_1m", 1, mk[m1_in], ws.fillna(0), we.fillna(0),
                                         sinks["candles_1m"], "n_candles_1m", "m1_status")
                res["window"] = (f"[start_utc_venues - {cfg.tip_offset_h}h - {cfg.m1_before_tip_min}min, "
                                 f"start_utc_venues + {cfg.m1_after_start_min}min] clamped to the market's life")
                self.phases["candles_1m"] = res
        finally:
            self.write_shard(sinks)

    # ---------------------------------------------------------------- trades
    def phase_trades(self, mk, tc, sink):
        zero = mk["volume"].notna() & (mk["volume"] <= 0)
        for t in mk.loc[zero, "ticker"]:
            self.set_stat(t, trades_status="skipped_zero_volume", n_trades=0)
        items = []
        for r in self.priority(mk[~zero]).itertuples():
            if r.source == "historical":
                eps = ("historical",) + (("live",) if tc is not None and r.close_time_s >= tc else ())
            else:
                eps = ("live",) + (("historical",) if tc is not None and not (r.open_time_s >= tc) else ())
            items.append((r.ticker, r.event_ticker, eps))

        def one(item):
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
                    rows = trade_rows(fresh, et, ep, self.extra)
                sink.add(rows)
                n += len(fresh)
            self.set_stat(ticker, n_trades=n, trades_status="ok" if ok else "error")

        res = self.run_tasks("trades", items, one)
        res.update(rows=sink.n, duplicates_dropped=self.dupes, markets_zero_volume_skipped=int(zero.sum()))
        self.phases["trades"] = res

    @staticmethod
    def priority(df):
        """nearest-the-middle rung first, newest event first."""
        return df.assign(dsort=pd.to_datetime(df["event_date"])).sort_values(
            ["rung_rank", "dsort"], ascending=[True, False]).drop(columns="dsort")

    # ---------------------------------------------------------------- candles
    def _batch(self, tickers, start, end, period):
        """GET /markets/candlesticks; returns {ticker: [candles]} or None. Splits if the candle cap is hit."""
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
            return None if a is None or b is None else {**a, **b}
        return got

    def _single(self, ticker, source, start, end, period):
        """Per-market candles (historical endpoint, or the live per-market endpoint), chunked."""
        out, ok, s = [], True, int(start)
        step = SINGLE_MAX_PERIODS * period * 60
        while s < end:
            e = min(int(end), s + step)
            path = (f"/historical/markets/{ticker}/candlesticks" if source == "historical"
                    else f"/series/{self.cfg.series}/markets/{ticker}/candlesticks")
            j = self.c.get(path, {"start_ts": s, "end_ts": e, "period_interval": period})
            if j is None:
                ok = False
            else:
                out += j.get("candlesticks") or []
            s = e + 1
        return out, ok

    def _live_groups(self, df, ws, we, period):
        """Pack live markets (newest event first) into batch calls within the ticker and candle caps."""
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
                    cur, s2, e2 = [], r.wstart, r.wend
                cur.append((r.ticker, r.event_ticker))
                cs, ce = s2, e2
        if cur:
            groups.append((cur, cs, ce))
        return groups

    def phase_candles(self, name, period, mk, ws, we, sink, key, status_key):
        self.batch_splits = 0
        has = we[mk.index] > ws[mk.index]
        for t in mk.loc[~has, "ticker"]:
            self.set_stat(t, **{status_key: "no_window", key: 0})
        mk = mk[has]
        live = mk[mk.source == "live"]
        hist = self.priority(mk[mk.source == "historical"])
        items = [("batch", g) for g in self._live_groups(live, ws, we, period)] + \
                [("single", (r.ticker, r.event_ticker, int(ws[r.Index]), int(we[r.Index])))
                 for r in hist.itertuples()]
        ix = {t: i for i, t in zip(self.mk.index, self.mk["ticker"])}

        def emit(t, et, source, cands, ok):
            rows = candle_rows(cands, t, et, source)
            sink.add(rows)
            self.set_stat(t, **{key: len(rows), status_key: "ok" if ok else "error"})

        def one(item):
            kind, payload = item
            if kind == "single":
                t, et, s, e = payload
                cands, ok = self._single(t, "historical", s, e, period)
                emit(t, et, "historical", cands, ok)
                return
            members, s, e = payload
            got = self._batch([t for t, _ in members], s, e, period)
            for t, et in members:
                if got is None:      # batch failed: per-market live endpoint
                    cands, ok = self._single(t, "live", ws[ix[t]], we[ix[t]], period)
                else:
                    cands, ok = got.get(t, []), True
                emit(t, et, "live", cands, ok)

        res = self.run_tasks(name, items, one)
        res.update(rows=sink.n, markets=int(len(mk)), live_batches=len(items) - len(hist),
                   historical_markets=int(len(hist)), batch_splits=self.batch_splits)
        return res

    # ---------------------------------------------------------------- output
    def write_shard(self, sinks):
        cfg, mk = self.cfg, self.mk
        defaults = {"n_trades": 0, "trades_status": "not_reached", "n_candles_1h": 0, "h1_status": "not_reached",
                    "n_candles_1m": 0, "m1_status": "not_reached", "m1_window_start": None, "m1_window_end": None}
        st = mk[["ticker", "event_ticker", "event_date", "season", "source", "strike", "dist_from_middle",
                 "volume"]].copy()
        for c, v in defaults.items():
            st[c] = [self.stat.get(t, {}).get(c, v) for t in st["ticker"]]
        for c in ("m1_window_start", "m1_window_end"):
            st[c] = pd.to_datetime(pd.to_numeric(st[c], errors="coerce"), unit="s", utc=True)
        month = dict(zip(mk["ticker"], pd.to_datetime(mk["event_date"]).dt.strftime("%Y-%m")))
        season = dict(zip(mk["ticker"], mk["season"]))
        tables = {name: s.table() for name, s in sinks.items()}
        info = self.run_info()
        info["phases"] = self.phases
        info["trades_extra"] = dict(self.extra)
        for se in sorted(mk["season"].dropna().unique()):
            folder = self.data / season_label(int(se)) / f"shard_{cfg.shard}"
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
            files = []
            ss = st[st["season"] == se]
            for name, tbl in tables.items():
                if tbl.num_rows == 0:
                    continue
                tk = tbl.column("ticker").to_pylist()
                keep = pa.array([season.get(t) == se for t in tk])
                sub = tbl.filter(keep)
                mon = [month.get(t) for t in sub.column("ticker").to_pylist()]
                sub = sub.append_column("_m", pa.array(mon, pa.string()))
                sort = [("ticker", "ascending"), ("created_time" if name == "trades" else "end_period_ts",
                                                  "ascending")]
                for m in sorted(set(mon)):
                    part = sub.filter(pc.equal(sub.column("_m"), m)).drop_columns(["_m"]).sort_by(sort)
                    files += write_capped(part, folder, f"{name}_{m}", cfg.max_file_mb)
            ss.to_parquet(folder / "fetch_status.parquet", index=False, compression="zstd")
            files.append(folder / "fetch_status.parquet")
            by_date = ss.groupby("event_date").agg(events=("event_ticker", "nunique"), markets=("ticker", "size"),
                                                   trades=("n_trades", "sum"), candles_1h=("n_candles_1h", "sum"),
                                                   candles_1m=("n_candles_1m", "sum")).reset_index()
            by_date["event_date"] = by_date["event_date"].astype(str)
            rng = {}
            for name, tbl in tables.items():
                if tbl.num_rows:
                    col = tbl.filter(pa.array([season.get(t) == se for t in tbl.column("ticker").to_pylist()])) \
                        .column("created_time" if name == "trades" else "end_period_ts")
                    rng[name] = [str(pc.min(col).as_py()), str(pc.max(col).as_py())] if len(col) else None
            self.dump(folder / "summary.json", {
                "season": season_label(int(se)), "shard": cfg.shard, "n_shards": cfg.n_shards,
                "events": int(ss["event_ticker"].nunique()), "markets": int(len(ss)),
                "rows": {"trades": int(ss["n_trades"].sum()), "candles_1h": int(ss["n_candles_1h"].sum()),
                         "candles_1m": int(ss["n_candles_1m"].sum())},
                "statuses": {c: {str(a): int(b) for a, b in ss[c].value_counts().items()}
                             for c in ("trades_status", "h1_status", "m1_status")},
                "complete": {c: bool(~ss[c].isin(["not_reached", "error"]).any())
                             for c in ("trades_status", "h1_status", "m1_status")},
                "event_date_range": [str(ss["event_date"].min()), str(ss["event_date"].max())],
                "data_time_range": rng,
                "files": [{"name": f.name, "bytes": f.stat().st_size, "rows": pq.ParquetFile(f).metadata.num_rows}
                          for f in files],
                "counts_by_event_date": json.loads(by_date.to_json(orient="records")),
                "run": info})
            log(f"wrote {folder}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")
        shutil.rmtree(self.work, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["plan", "shard"], default="plan")
    p.add_argument("--data-out", default="data_out", help="root of the committed tree (kalshi_totals/...)")
    p.add_argument("--plan", default="plan_out", help="internal plan folder (Actions artifact between jobs)")
    p.add_argument("--work", default="work_tmp", help="scratch folder for part files")
    p.add_argument("--series", default=SERIES)
    p.add_argument("--base", default=venues.KALSHI)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=4)
    p.add_argument("--rps", type=float, default=8.0, help="max requests per second for this process")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--budget-min", type=float, default=290.0, help="stop starting new work after this many minutes")
    p.add_argument("--phases", default="trades,candles_1h,candles_1m")
    p.add_argument("--live-h1-pts", type=float, default=INF)
    p.add_argument("--hist-h1-pts", type=float, default=10.0)
    p.add_argument("--live-m1-pts", type=float, default=10.0)
    p.add_argument("--hist-m1-pts", type=float, default=-1.0, help="-1 = no 1-minute candles for historical")
    p.add_argument("--tip-offset-h", type=float, default=3.0, help="tip_est = occurrence_datetime - this")
    p.add_argument("--m1-before-tip-min", type=float, default=240.0)
    p.add_argument("--m1-after-start-min", type=float, default=15.0)
    p.add_argument("--h1-max-days", type=float, default=30.0)
    p.add_argument("--batch-max-candles", type=int, default=BATCH_MAX_CANDLES)
    p.add_argument("--batch-fill", type=float, default=0.9, help="pack batch calls up to this share of the cap")
    p.add_argument("--max-file-mb", type=float, default=85.0)
    p.add_argument("--max-markets", type=int, default=0, help="debug: newest N markets only (plan mode)")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv=None):
    cfg = build_parser().parse_args(argv)
    if cfg.self_test:
        return self_test()
    if cfg.mode == "shard" and not 0 <= cfg.shard < cfg.n_shards:
        raise SystemExit(f"--shard {cfg.shard} not in [0, {cfg.n_shards})")
    log(f"EXP-009a Kalshi history: mode {cfg.mode}, series {cfg.series}, shard {cfg.shard}/{cfg.n_shards}, "
        f"{cfg.rps} req/s, budget {cfg.budget_min} min")
    r = Runner(cfg, Client(cfg.base, rps=cfg.rps))
    if cfg.mode == "plan":
        r.plan()
    else:
        r.shard()
    print(json.dumps({k: v for k, v in r.run_info().items() if k not in ("config", "field_units")} |
                     {"phases": r.phases}, indent=1, default=str), flush=True)
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
    cursor pagination everywhere, a 429 and a 500 to retry through, legacy cents fields on archived markets."""

    NOW = int(datetime(2026, 11, 23, tzinfo=timezone.utc).timestamp())

    def __init__(self, cap=BATCH_MAX_CANDLES):
        self.cap, self.calls, self.lock = cap, Counter(), threading.Lock()
        self.failed_once = set()
        self.markets, self.events = {}, {}

        def ladder(et, strikes, open_ts, tip_ts, src, status, total_pts=None, vol0=()):
            ev = {"event_ticker": et, "series_ticker": SERIES, "title": "Team A vs Team B: Total Points",
                  "sub_title": "A vs B", "strike_date": utc(tip_ts + 3 * 3600), "fee_type_override": None}
            self.events[et] = (ev, src)
            for k in strikes:
                t = f"{et}-{int(k)}"
                close_ts = tip_ts + 2 * 3600 + 900 if status != "active" else tip_ts + 3 * 3600
                res = "" if status == "active" else ("yes" if total_pts > k else "no")
                m = {"ticker": t, "event_ticker": et, "market_type": "binary", "title": ev["title"],
                     "yes_sub_title": f"Over {k} points scored", "open_time": utc(open_ts),
                     "close_time": utc(close_ts), "expected_expiration_time": utc(tip_ts + 3 * 3600),
                     "status": status, "result": res, "strike_type": "greater", "floor_strike": k,
                     "rules_primary": f"If total points in A vs B are above {k}, Yes.",
                     "rules_secondary": "Includes overtime.", "early_close_condition": "closes when final",
                     "can_close_early": True}
                if src == "historical":   # archived markets: legacy cents fields, no occurrence_datetime
                    m.update({"yes_bid": 0, "yes_ask": 1, "last_price": 99 if res == "yes" else 1,
                              "volume": 0 if k in vol0 else 120, "open_interest": 50,
                              "settlement_value": 100 if res == "yes" else 0})
                else:
                    m.update({"occurrence_datetime": utc(tip_ts + 3 * 3600), "yes_bid_dollars": "0.4500",
                              "yes_ask_dollars": "0.4700", "last_price_dollars": "0.4600",
                              "volume_fp": "0.00" if k in vol0 else "250.00", "open_interest_fp": "80.00",
                              "settlement_value_dollars": None if res == "" else ("1.0000" if res == "yes" else
                                                                                  "0.0000")})
                    if res:
                        m["settlement_ts"] = utc(close_ts + 600)
                self.markets[t] = (m, src)

        tip1 = int(datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc).timestamp())
        ladder("KXNCAAMBTOTAL-26NOV19AAABBB", [128.5, 134.5, 140.5, 146.5, 152.5], tip1 - 2 * 86400, tip1,
               "live", "finalized", total_pts=142, vol0=(128.5,))
        tip2 = int(datetime(2026, 11, 30, 1, 0, tzinfo=timezone.utc).timestamp())
        ladder("KXNCAAMBTOTAL-26NOV29CCDDDD", [131.5, 136.5, 141.5, 146.5, 151.5], self.NOW - 86400, tip2,
               "live", "active")
        tip3 = int(datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc).timestamp())
        ladder("KXNCAAMBTOTAL-26FEB10EEEFFH", [120.5 + 5 * i for i in range(11)], tip3 - 86400, tip3,
               "historical", "finalized", total_pts=151)

    @staticmethod
    def _page(items, params):
        lim = min(int(params.get("limit", 100)), 4)          # force multi-page pagination in the test
        start = int(params.get("cursor") or 0)
        return items[start:start + lim], (str(start + lim) if start + lim < len(items) else "")

    def _candles(self, t, s, e, period, fmt):
        m, _ = self.markets[t]
        life_s, life_e = iso_ts(m["open_time"]), iso_ts(m["close_time"])
        step, out = period * 60, []
        ts = (max(s, life_s) // step + 1) * step
        while ts <= min(e, life_e + step):
            if fmt == "live":
                c = {"end_period_ts": ts,
                     "yes_bid": {f"{k}_dollars": "0.4400" for k in ("open", "high", "low", "close")},
                     "yes_ask": {f"{k}_dollars": "0.4600" for k in ("open", "high", "low", "close")},
                     "price": {f"{k}_dollars": "0.4500" for k in ("open", "high", "low", "close", "mean")},
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
        m, _ = self.markets[t]
        base = iso_ts(m["open_time"]) + 3600
        rows = []
        for i in range(5):
            if src == "historical":
                rows.append({"trade_id": f"{t}-h{i}", "ticker": t, "created_time": utc(base + 60 * i),
                             "yes_price": 40 + i, "no_price": 60 - i, "count": 10, "taker_side": "yes"})
            else:
                rows.append({"trade_id": f"{t}-l{i}", "ticker": t,
                             "created_time": utc(base + 60 * i).replace("Z", ".123456Z"),
                             "yes_price_dollars": f"0.{40 + i}00", "no_price_dollars": f"0.{60 - i}00",
                             "count_fp": "10.00", "taker_side": "no", "is_block_trade": False})
        if src == "live":
            rows.insert(2, dict(rows[1]))                 # a duplicate across pages -> must be dropped
        return rows

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
            return _Resp(200, {"series": {"ticker": SERIES, "fee_type": "quadratic_with_maker_fees",
                                          "fee_multiplier": 1}})
        if path == "/historical/cutoff":
            return _Resp(200, {"market_settled_ts": "2026-06-01T00:00:00Z", "trades_created_ts":
                               "2026-06-01T00:00:00Z"})
        if path == "/series/fee_changes":
            return _Resp(200, {"series_fee_change_arr": []})
        if path == "/events":
            if params.get("tickers"):
                evs = [self.events[t][0] for t in params["tickers"].split(",") if t in self.events]
            else:
                evs = [e for e, src in self.events.values() if src == "live"]
            page, cur = self._page(evs, params)
            return _Resp(200, {"events": page, "cursor": cur})
        if path in ("/markets", "/historical/markets"):
            src = "live" if path == "/markets" else "historical"
            page, cur = self._page([m for m, s in self.markets.values() if s == src], params)
            return _Resp(200, {"markets": page, "cursor": cur})
        if path in ("/markets/trades", "/historical/trades"):
            t, src = params["ticker"], ("live" if path == "/markets/trades" else "historical")
            page, cur = self._page(self._trades(t, src) if self.markets[t][1] == src else [], params)
            return _Resp(200, {"trades": page, "cursor": cur})
        if path == "/markets/candlesticks":
            ts = params["market_tickers"].split(",")
            assert len(ts) <= BATCH_MAX_TICKERS
            if any(self.markets[t][1] != "live" for t in ts):
                return _Resp(400, {"error": "historical market in batch"})
            out, n = [], 0
            for t in ts:
                cs = self._candles(t, int(params["start_ts"]), int(params["end_ts"]),
                                   int(params["period_interval"]), "live")[:max(0, self.cap - n)]
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

    def run(tmp, mode, cap=BATCH_MAX_CANDLES, **over):
        FIELD_UNITS.clear()
        clock, fake = FakeClock(), FakeKalshi(cap=cap)
        argv = ["--mode", mode, "--data-out", str(tmp / "data"), "--plan", str(tmp / "plan"),
                "--work", str(tmp / "work"), "--rps", "50", "--workers", "3"] + sum(
            ([f"--{k.replace('_', '-')}", str(v)] for k, v in over.items()), [])
        cfg = build_parser().parse_args(argv)
        r = Runner(cfg, Client(cfg.base, session=fake, rps=cfg.rps, sleep=clock.sleep, clock=clock),
                   clock=clock, now=FakeKalshi.NOW)
        return (r.plan() if mode == "plan" else r.shard()), r, fake

    tmp = Path(tempfile.mkdtemp(prefix="exp009_selftest_"))
    try:
        # ---- plan
        ps, _, _ = run(tmp, "plan", n_shards=2)
        meta = tmp / "data" / "kalshi_totals" / "meta"
        for f in ("series.json", "cutoff.json", "series_fee_changes.json", "plan_summary.json", "markets.parquet",
                  "events.parquet", "rules_sample.json", "markets_raw_sample.json"):
            assert (meta / f).exists(), f"missing meta/{f}"
        mk, ev = pd.read_parquet(meta / "markets.parquet"), pd.read_parquet(meta / "events.parquet")
        assert len(mk) == 21 and mk.ticker.is_unique and (mk.source == "historical").sum() == 11
        assert len(ev) == 3 and ev.event_found.all() and not ev.set_index("event_ticker").at[
            "KXNCAAMBTOTAL-26FEB10EEEFFH", "in_events_list"]
        e1 = ev.set_index("event_ticker").loc["KXNCAAMBTOTAL-26NOV19AAABBB"]
        assert e1.middle_strike == 140.5 and str(e1.start_utc_venues) == "2026-11-20 03:00:00+00:00"
        assert str(e1.tip_est_utc) == "2026-11-20 00:00:00+00:00"
        hm = mk[mk.source == "historical"].iloc[0]
        assert hm.yes_ask == 0.01 and hm.settlement_value in (0.0, 1.0), "legacy cents not converted"
        assert ps["series_fee"]["fee_type"] == "quadratic_with_maker_fees" and ps["run"]["retries"] >= 1
        assert {r["season"] for r in ps["by_season"]} == {2026, 2027}
        assert json.loads((meta / "rules_sample.json").read_text())["distinct_rules_secondary"] == \
            {"Includes overtime.": 3}
        plan_mk = pd.read_parquet(tmp / "plan" / "markets_plan.parquet")
        assert set(plan_mk.shard) <= {0, 1}

        # ---- both shards (tiny candle cap forces batch splitting; tiny file cap forces file splitting)
        reps = {}
        for k in (0, 1):
            _, r, fake = run(tmp, "shard", cap=150, shard=k, n_shards=2, batch_max_candles=150, batch_fill=3.0,
                             max_file_mb=0.006)
            reps[k] = r
            assert fake.calls.get("/series/KXNCAAMBTOTAL", 0) == 0, "shards must not refetch meta"
        root = tmp / "data" / "kalshi_totals"
        st = pd.concat([pd.read_parquet(p) for p in root.glob("20*/shard_*/fetch_status.parquet")])
        assert len(st) == 21 and st.ticker.is_unique, "every market belongs to exactly one shard"
        assert not list(root.rglob("_parts_*")) and not (tmp / "work").exists()
        tr = pd.concat([pd.read_parquet(p) for p in root.glob("20*/shard_*/trades_*.parquet")])
        h1 = pd.concat([pd.read_parquet(p) for p in root.glob("20*/shard_*/candles_1h_*.parquet")])
        m1 = pd.concat([pd.read_parquet(p) for p in root.glob("20*/shard_*/candles_1m_*.parquet")])
        assert any("_p" in p.name for p in root.glob("20*/shard_*/*.parquet")), "file splitting not exercised"
        assert set(p.parent.parent.name for p in root.glob("20*/shard_*/trades_*.parquet")) == {"2025-26", "2026-27"}
        assert list(tr.columns) == TRADES_OUT.names and list(h1.columns) == CANDLES_OUT.names
        # trades: 4 + 5 live markets with volume, 11 historical; 5 each; live duplicates dropped
        assert tr.trade_id.is_unique and len(tr) == (4 + 5 + 11) * 5, len(tr)
        assert sum(r.dupes for r in reps.values()) == 9
        assert tr.yes_price.between(0, 1).all() and tr.created_time.notna().all() and (tr["count"] == 10).all()
        assert set(tr[tr.ticker.str.contains("FEB10")].yes_price.round(2)) == {0.40, 0.41, 0.42, 0.43, 0.44}
        stx = st.set_index("ticker")
        assert stx.at["KXNCAAMBTOTAL-26NOV19AAABBB-128", "trades_status"] == "skipped_zero_volume"
        # scope: live h1 all rungs; historical h1 only within 10 pts; 1m only live within 10 pts
        hist = stx[stx.source == "historical"]
        assert set(h1.ticker) == set(stx[(stx.source == "live")].index) | set(hist[hist.dist_from_middle <= 10].index)
        assert (hist[hist.dist_from_middle > 10].h1_status == "out_of_scope").all()
        assert (hist.m1_status == "out_of_scope").all()
        live1 = stx[stx.index.str.contains("AAABBB")]
        assert set(m1.ticker) == set(live1[live1.dist_from_middle <= 10].index)
        assert (live1[live1.dist_from_middle > 10].m1_status == "out_of_scope").all()
        assert (stx[stx.index.str.contains("CCDDDD") & (stx.dist_from_middle <= 10)].m1_status == "no_window").all()
        # candles inside market lifetimes, windows respected, no duplicates
        lim = mk.set_index("ticker")
        for c, per in ((h1, 60), (m1, 1)):
            j = c.join(lim[["open_time", "close_time"]], on="ticker")
            t = pd.to_datetime(j.end_period_ts, unit="s", utc=True)
            assert (t > j.open_time).all() and (t <= j.close_time + pd.Timedelta(minutes=per)).all()
            assert not c.duplicated(["ticker", "end_period_ts"]).any()
        j = m1.join(stx[["m1_window_start", "m1_window_end"]], on="ticker")
        t = pd.to_datetime(j.end_period_ts, unit="s", utc=True)
        assert (t > j.m1_window_start).all() and (t <= j.m1_window_end + pd.Timedelta(minutes=1)).all()
        assert h1[h1.ticker.str.contains("FEB10")].yes_ask_close.eq(0.32).all()
        assert h1[h1.ticker.str.contains("FEB10")].price_close.isna().all()
        assert m1.yes_bid_close.eq(0.44).all()
        sums = [json.loads(p.read_text()) for p in root.glob("20*/shard_*/summary.json")]
        assert all(s["complete"] == {"trades_status": True, "h1_status": True, "m1_status": True} for s in sums)
        assert sum(s["rows"]["trades"] for s in sums) == len(tr)
        assert sum(s["run"]["phases"].get("candles_1m", {}).get("batch_splits", 0) for s in sums) > 0
        for s in sums:
            for f in s["files"]:
                assert f["bytes"] < 0.006e6 * 1.5 or f["rows"] == 1 or f["name"] == "fetch_status.parquet"

        # ---- zero budget: files + status still written, nothing fetched, marked incomplete
        shutil.rmtree(tmp / "data" / "kalshi_totals" / "2025-26")
        _, r0, _ = run(tmp, "shard", shard=0, n_shards=1, budget_min=0)
        s0 = json.loads((root / "2025-26" / "shard_0" / "summary.json").read_text())
        assert s0["complete"]["trades_status"] is False and s0["rows"]["trades"] == 0
        assert pd.read_parquet(root / "2025-26" / "shard_0" / "fetch_status.parquet").trades_status \
            .isin(["not_reached", "skipped_zero_volume"]).all()
        print("self-test OK:", json.dumps({"trades": len(tr), "candles_1h": len(h1), "candles_1m": len(m1),
                                            "files": sum(len(s["files"]) for s in sums)}))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
