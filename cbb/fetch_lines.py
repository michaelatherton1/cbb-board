"""Pull historical opening/closing lines from ESPN game JSON mirrored in sportsdataverse/hoopR-mbb-raw.

Usage: python fetch_lines.py <season> [<season> ...]
Writes data/lines_<season>.parquet with one row per game (first/priority provider).
"""
import sys, json, gzip, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-raw/main/mbb/json/final/{}.json"
sess = requests.Session()
sess.headers["Accept-Encoding"] = "gzip"


def _f(x):
    try:
        return float(str(x).lstrip("ou+").replace("EVEN", "100"))
    except (TypeError, ValueError):
        return None


def parse(gid, j):
    pcs = j.get("pickcenter") or []
    if not pcs:
        return None
    pcs = sorted(pcs, key=lambda p: (p.get("provider") or {}).get("priority", 99))
    with_line = [p for p in pcs if _f(p.get("spread")) is not None or _f(p.get("overUnder")) is not None]
    if not with_line:
        return None
    p = with_line[0]
    h, a = p.get("homeTeamOdds") or {}, p.get("awayTeamOdds") or {}
    row = dict(
        game_id=int(gid),
        provider=(p.get("provider") or {}).get("name"),
        spread_close=_f(p.get("spread")),          # home perspective, negative = home favored
        total_close=_f(p.get("overUnder")),
        ml_home_close=_f(h.get("moneyLine")),
        ml_away_close=_f(a.get("moneyLine")),
        home_team_id=h.get("teamId"),
    )
    ps, tot = p.get("pointSpread") or {}, p.get("total") or {}
    row["spread_open"] = _f(((ps.get("home") or {}).get("open") or {}).get("line"))
    row["total_open"] = _f(((tot.get("over") or {}).get("open") or {}).get("line"))

    # The raw "spread" field is not reliably home-perspective (esp. neutral sites in older consensus data).
    # Rebuild it from the "details" text ("GONZ -34") matched against the box score's home team.
    row["details"] = p.get("details")
    row["home_fav_flag"] = h.get("favorite")
    comps = (((j.get("header") or {}).get("competitions") or [{}])[0].get("competitors") or [])
    abbr = {c.get("homeAway"): ((c.get("team") or {}).get("abbreviation")) for c in comps}
    ids = {c.get("homeAway"): str(c.get("id")) for c in comps}
    row["box_home_id"] = ids.get("home")
    row["spread_home"] = home_spread(row["details"], abbr.get("home"), abbr.get("away"), row["spread_close"],
                                     h.get("favorite"), a.get("favorite"))
    return row


def home_spread(details, home_abbr, away_abbr, raw, home_fav, away_fav):
    if raw is None:
        return None
    mag = abs(raw)
    if details:
        d = details.strip()
        if d.upper() in ("EVEN", "PK", "PICK"):
            return 0.0
        parts = d.rsplit(" ", 1)
        if len(parts) == 2 and _f(parts[1]) is not None:
            team, val = parts[0], _f(parts[1])
            if team == home_abbr:
                return val
            if team == away_abbr:
                return -val
    if home_fav is True and away_fav is not True:
        return -mag
    if away_fav is True and home_fav is not True:
        return mag
    return None


def fetch(gid):
    for attempt in range(3):
        try:
            r = sess.get(URL.format(gid), timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return parse(gid, r.json())
        except Exception:
            time.sleep(1 + attempt)
    return "ERR"


def run(season, workers=24):
    sched = pd.read_parquet(DATA / f"sched_{season}.parquet")
    sched = sched[sched.status_type_completed == True]
    gids = sched.game_id.astype(int).tolist()
    out, errs, t0 = [], 0, time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, g): g for g in gids}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r == "ERR":
                errs += 1
            elif r:
                out.append(r)
            if i % 1000 == 0:
                print(f"  {season}: {i}/{len(gids)} games, {len(out)} with lines, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    df.to_parquet(DATA / f"lines_{season}.parquet")
    print(f"{season}: {len(gids)} games, {len(df)} with lines, {errs} errors, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(int(s))
