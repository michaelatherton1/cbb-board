"""Match venue team text ('Montana St.', 'UNC Asheville', 'Duke Blue Devils') to ESPN team ids."""
import re
from difflib import SequenceMatcher
import pandas as pd

# venue spellings that string similarity can't bridge (normalized form -> normalized ESPN form)
ALIASES = {
    "connecticut": "uconn", "ole miss": "mississippi", "umass": "massachusetts",
    "pitt": "pittsburgh", "uva": "virginia", "miami fl": "miami", "miami florida": "miami",
    "miami oh": "miami ohio", "nc state": "north carolina state", "unc": "north carolina",
    "ucf": "central florida", "usc": "southern california", "lsu": "louisiana state",
    "byu": "brigham young", "smu": "southern methodist", "tcu": "texas christian",
    "unlv": "nevada las vegas", "utep": "texas el paso", "utsa": "texas san antonio",
    "uab": "alabama birmingham", "vcu": "virginia commonwealth", "fiu": "florida international",
    "fau": "florida atlantic", "etsu": "east tennessee state", "siue": "siu edwardsville",
    "ul monroe": "louisiana monroe", "ulm": "louisiana monroe", "louisiana lafayette": "louisiana",
    "app state": "appalachian state", "umkc": "kansas city", "iupui": "iu indianapolis",
    "texas a and m commerce": "east texas a and m", "tamu commerce": "east texas a and m",
    "texas a and m corpus christi": "texas a and m corpus christi", "a and m corpus christi": "texas a and m corpus christi",
    "uconn huskies": "uconn", "st marys": "saint marys", "st josephs": "saint josephs",
    "st peters": "saint peters", "st bonaventure": "saint bonaventure", "st louis": "saint louis",
    "st thomas": "st thomas minnesota", "loyola il": "loyola chicago", "loyola md": "loyola maryland",
    "cal state fullerton": "cal state fullerton", "csun": "cal state northridge",
    "long beach state": "long beach state", "uncw": "unc wilmington", "uncg": "unc greensboro",
    "sfa": "stephen f austin", "ut martin": "ut martin", "tennessee martin": "ut martin",
    "southern miss": "southern miss", "southern mississippi": "southern miss",
    "detroit": "detroit mercy", "purdue fw": "purdue fort wayne", "pfw": "purdue fort wayne",
    "omaha": "omaha", "nebraska omaha": "omaha", "little rock": "little rock", "ualr": "little rock",
    "mcneese state": "mcneese", "nicholls state": "nicholls", "southeastern louisiana": "se louisiana",
    "hawaii": "hawaii", "queens nc": "queens", "san jose state": "san jose state",
    "mount st marys": "mount saint marys", "md eastern shore": "maryland eastern shore",
    "umes": "maryland eastern shore", "grambling state": "grambling", "prairie view": "prairie view a and m",
    "ark pine bluff": "arkansas pine bluff", "miss valley state": "mississippi valley state",
    "bethune cookman": "bethune cookman", "fdu": "fairleigh dickinson", "liu": "long island university",
    "long island": "long island university", "njit": "njit", "uic": "illinois chicago",
    "csu bakersfield": "cal state bakersfield", "cal baptist": "california baptist",
    "uc santa barbara": "uc santa barbara", "ucsb": "uc santa barbara", "ucsd": "uc san diego",
    "sam houston state": "sam houston", "gw": "george washington", "gwu": "george washington",
    # short forms seen on Polymarket US / Kalshi
    "uca": "central arkansas", "nm state": "new mexico state", "eiu": "eastern illinois",
    "ualbany": "albany", "at albany": "albany", "albany ny": "albany", "unf": "north florida",
    "ndsu": "north dakota state", "uni": "northern iowa", "unco": "northern colorado",
    "uiw": "incarnate word", "ewu": "eastern washington", "fgcu": "florida gulf coast",
    "uncw": "unc wilmington", "lmu": "loyola marymount", "csub": "cal state bakersfield",
}


def norm(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.replace("&", " and ").replace("’", "'").replace("é", "e")
    s = re.sub(r"\(([^)]*)\)", r" \1 ", s)                 # 'miami (fl)' -> 'miami fl'
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"[-/]", " ", s)
    s = re.sub(r"^st\s+", "saint ", s)                     # leading St. = Saint
    s = re.sub(r"\s+st$", " state", s)                     # trailing St. = State
    s = re.sub(r"\buniv(ersity)?\b|\bthe\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("saint ", "st ") if s.startswith("saint ") else s
    return ALIASES.get(s, s)


def _key(s):
    """Second normal form: treat 'st'/'saint' alike and drop spaces."""
    s = norm(s)
    s = re.sub(r"^st ", "saint ", s)
    return ALIASES.get(s, s).replace(" ", "")


def similarity(a, b):
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if na == nb or _key(a) == _key(b):
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    jac = len(ta & tb) / len(ta | tb)
    seq = SequenceMatcher(None, na, nb).ratio()
    # one name fully inside the other ('montana state' vs 'montana state bobcats')
    contain = 0.92 if (f" {na} " in f" {nb} " or f" {nb} " in f" {na} ") else 0.0
    return max(jac, seq, contain)


def team_aliases(sched):
    """ESPN team_id -> list of names (location, display, short display, nickname, abbreviation)."""
    rows = []
    for side in ("home", "away"):
        cols = [f"{side}_id", f"{side}_location", f"{side}_display_name", f"{side}_short_display_name",
                f"{side}_name", f"{side}_abbreviation"]
        cols = [c for c in cols if c in sched]
        rows.append(sched[cols].set_axis(["team_id"] + cols[1:], axis=1)
                    .melt(id_vars="team_id", value_name="alias")[["team_id", "alias"]])
    a = pd.concat(rows).dropna()
    a["team_id"] = a.team_id.astype(int)
    a = a[a.alias.astype(str).str.len() > 1].drop_duplicates()
    return a.groupby("team_id").alias.apply(list).to_dict()


def best_score(text, aliases):
    return max((similarity(text, x) for x in aliases), default=0.0)


def match_games(events, games, aliases, min_each=0.72, min_pair=0.82, day_slack=1):
    """events: DataFrame with event_key, date, team1, team2. games: home_id, away_id, date, game_id.

    Returns dict event_key -> (game_id, flipped) where flipped means team1 is the HOME team.
    """
    out = {}
    games = games.copy()
    games["d"] = pd.to_datetime(games.date).dt.date
    for ev in events.itertuples():
        if ev.date is None or pd.isna(ev.date):
            cand = games
        else:
            dd = pd.Timestamp(ev.date).date()
            cand = games[games.d.apply(lambda x: abs((x - dd).days) <= day_slack)]
        scored = []
        for g in cand.itertuples():
            ah, aa = aliases.get(g.home_id, []), aliases.get(g.away_id, [])
            s1h, s2a = best_score(ev.team1, ah), best_score(ev.team2, aa)
            s1a, s2h = best_score(ev.team1, aa), best_score(ev.team2, ah)
            if min(s1a, s2h) >= min_each:      # team1 away, team2 home (usual 'A vs/at B')
                scored.append(((s1a + s2h) / 2, g.game_id, False))
            if min(s1h, s2a) >= min_each:
                scored.append(((s1h + s2a) / 2, g.game_id, True))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < min_pair:
            continue
        if len(scored) > 1 and scored[1][1] != scored[0][1] and scored[0][0] - scored[1][0] < 0.04:
            continue                          # ambiguous: two different games score alike
        out[ev.event_key] = (scored[0][1], scored[0][2])
    return out
