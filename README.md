# EXP-009a data: Kalshi college basketball totals (KXNCAAMBTOTAL)

This is a data-only orphan branch. Its files are written by `.github/workflows/exp009-kalshi-history.yml` on the
`agents/exp-009-kalshi-totals` branch, using the fetcher `experiments/exp009/kalshi_history.py`.

The source is Kalshi's public Trade API v2. Prices are in dollars (0-1). Times are UTC.

```
kalshi_totals/meta/                 series.json, series_fee_changes.json, cutoff.json, plan_summary.json,
                                    markets.parquet (every market, live + historical), events.parquet (one row per
                                    event, rules text of the middle rung), rules_sample.json, markets_raw_sample.json
kalshi_totals/<season>/shard_<k>/   trades_<YYYY-MM>[_pN].parquet, candles_1h_<YYYY-MM>.parquet,
                                    candles_1m_<YYYY-MM>.parquet, fetch_status.parquet, summary.json
```

- `<season>` is `2025-26` (historical markets) or `2026-27` (live markets). `YYYY-MM` is the event month, so all
  of a market's rows are in one file.
- A shard is the set of events with `crc32(event_ticker) % N == k`.
- `fetch_status.parquet` has one row per market: row counts and an ok / error / not_reached / out_of_scope /
  no_window / skipped_zero_volume status for each data type.
- The tip time is estimated: `events.tip_est_utc = start_utc_venues - 3h`. Kalshi's `occurrence_datetime` is tip + 3h
  on open markets. The 1-minute candles cover `[occurrence - 7h, occurrence + 15m]`, clamped to the market's life.
