"""Coordinate-descent tuning of rating hyperparameters on TRAIN seasons only (2012-2017)."""
import sys, json, time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from model import Params, walk_forward

DATA = Path(__file__).resolve().parent.parent / "data"
TRAIN = list(range(2012, 2018))


def score(pred):
    # calibrated (in-sample linear) errors so params aren't judged on scale bias
    bm = np.polyfit(pred.raw_margin, pred.margin, 1)
    bt = np.polyfit(pred.raw_total, pred.total, 1)
    em = pred.margin - np.polyval(bm, pred.raw_margin)
    et = pred.total - np.polyval(bt, pred.raw_total)
    return dict(m_rmse=float(np.sqrt((em ** 2).mean())), m_mae=float(em.abs().mean()),
                t_rmse=float(np.sqrt((et ** 2).mean())), t_mae=float(et.abs().mean()))


def evaluate(games, p):
    pred, _ = walk_forward(games, TRAIN, p)
    return score(pred)


GRID = {
    "half_life": [20, 30, 45, 60, 90, 150, 1e9],
    "lam_eff": [2, 4, 6, 9, 13, 20],
    "r_eff": [0.4, 0.5, 0.6, 0.7, 0.8],
    "new_team_eff": [-16, -12, -8, -4, 0],
    "blowout_cap": [25, 35, 50, 999],
    "lam_tempo": [2, 4, 6, 9, 13, 20],
    "r_tempo": [0.5, 0.65, 0.8, 0.9],
}
OBJ = {k: "m_rmse" for k in GRID}
OBJ.update(lam_tempo="t_rmse", r_tempo="t_rmse")

if __name__ == "__main__":
    games = pd.read_parquet(DATA / "games.parquet")
    p = Params()
    cache = {}

    def ev(p):
        key = tuple(sorted(asdict(p).items()))
        if key not in cache:
            cache[key] = evaluate(games, p)
        return cache[key]

    for rnd in range(2):
        for k, vals in GRID.items():
            t0 = time.time()
            res = {v: ev(replace(p, **{k: v})) for v in vals}
            best = min(res, key=lambda v: res[v][OBJ[k]])
            p = replace(p, **{k: best})
            print(f"round {rnd} {k:13s} -> {best:<8} " +
                  " ".join(f"{v}:{res[v][OBJ[k]]:.3f}" for v in vals) + f"  ({time.time()-t0:.0f}s)", flush=True)
    final = ev(p)
    print("BEST", asdict(p), final)
    (DATA / "best_params.json").write_text(json.dumps(asdict(p), indent=1))
