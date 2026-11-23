"""Render the dashboard: one self-contained page (docs/cbb/index.html) with today's data embedded.

The page is static, so GitHub Pages can serve it; every run of run.py rewrites it.
"""
import json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import venues

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).parent / "dashboard.html"
SITE = ROOT / "docs" / "cbb"

MANUAL_VENUES = ["DraftKings Predictions", "FanDuel Predicts", "Fanatics Markets", "Underdog", "Crypto.com / OG",
                 "ProphetX", "Robinhood", "Other"]


def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return x


def _records(df, cols):
    return [{c: _clean(v) for c, v in zip(cols, row)} for row in df[cols].itertuples(index=False, name=None)]


def build(proj, board, status, results, ratings, cal, asof, model_asof=None, records=None):
    proj = proj.copy()
    proj["date"] = pd.to_datetime(proj.date).dt.strftime("%Y-%m-%d")
    gcols = ["game_id", "date", "tipoff", "away", "home", "away_id", "home_id", "neutral", "away_pts", "home_pts", "home_spread", "total",
             "home_win_prob", "fair_ml_home", "fair_ml_away", "away_games_played", "home_games_played"]
    gcols = [c for c in gcols if c in proj]
    games = _records(proj, gcols)
    q = []
    if board is not None and len(board):
        b = board.copy()
        b["line"] = pd.to_numeric(b.line, errors="coerce")
        q = _records(b, ["game_id", "venue", "market", "side", "line", "ask", "bid", "fee", "fee_coef", "model_prob",
                         "edge", "ask_size"])
    payload = dict(
        asof=str(asof),
        prices_utc=datetime.now(timezone.utc).isoformat(timespec="minutes"),
        model_asof=model_asof,
        venues=status,
        params=dict(sd_margin=cal["resid_sd"], sd_total=cal["total_sd"], sigma_ml=cal["sigma"],
                    min_edge=0.03, manual_venues=MANUAL_VENUES, reference_venues=venues.REFERENCE_VENUES),
        games=games, quotes=q, records={str(k): v for k, v in (records or {}).items()},
        results=results,
        ratings=_records(ratings, ["rank", "team", "games", "adj_em", "adj_o", "adj_d", "adj_t"]) if ratings is not None else [],
    )
    return payload


def render(payload, out_dir=SITE):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"), default=str).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__CBB_DATA__", data)
    (out_dir / "index.html").write_text(html)
    (out_dir / "data.json").write_text(json.dumps(payload, default=str))
    return out_dir / "index.html"
