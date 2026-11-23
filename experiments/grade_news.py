"""Grade the game-day news notes (EXP-003) once games are final.

    python experiments/grade_news.py            # uses data/out/results_games.csv (refreshed by the morning run)

For every logged game it asks two questions, both measured from the book line when the game was flagged:
  1. Did the closing line move toward the model's number?   (the cleanest test of "the model knew something")
  2. Did the model's side cover the flagged line?            (noisier: one game is mostly luck)
and splits the answers by the news agent's lean (supports_model / supports_market / no_relevant_news).
A useful news overlay should show "supports_market" notes on games where the close did NOT move toward the model.
Needs dozens of games before any of this means anything.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "experiments" / "news_log.csv"
RES = ROOT / "data" / "out" / "results_games.csv"


def grade(log=LOG, res=RES):
    n = pd.read_csv(log)
    r = pd.read_csv(res, usecols=["game_id", "margin", "spread_close"])
    g = n.merge(r, on="game_id", how="left")
    done = g[g.margin.notna() & g.spread_close.notna()].copy()
    flag = done.books_home_spread_flagged
    model = done.model_home_spread
    # line movement toward the model: the close moved from the flagged number in the model's direction
    done["close_toward_model"] = np.sign(done.spread_close - flag) == np.sign(model - flag)
    done["close_moved"] = done.spread_close != flag
    # model side vs the flagged line: model home spread below the book's -> model likes home more
    home_side = model < flag
    home_cover = done.margin + flag
    done["model_side_covered"] = np.where(home_cover == 0, np.nan, np.where(home_side, home_cover > 0, home_cover < 0))
    return g, done


def main():
    g, done = grade()
    print(f"logged games: {len(g)}  graded (final, with a closing line): {len(done)}")
    if done.empty:
        print("nothing to grade yet - run after the next morning's dashboard update")
        return
    pd.set_option("display.width", 200)
    cols = ["date", "away", "home", "lean", "books_home_spread_flagged", "model_home_spread", "spread_close",
            "close_toward_model", "margin", "model_side_covered"]
    print(done[cols].to_string(index=False))
    moved = done[done.close_moved]
    summary = moved.groupby("lean").agg(games=("game_id", "size"), close_toward_model=("close_toward_model", "mean"))
    cover = done.groupby("lean").agg(model_side_cover=("model_side_covered", "mean"))
    print("\nby lean (close-toward-model share uses only games whose line moved):")
    print(summary.join(cover, how="outer").round(2).to_string())


if __name__ == "__main__":
    main()
