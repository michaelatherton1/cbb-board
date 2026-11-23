# College basketball model + price dashboard

Every morning it re-downloads men's D-I teams, box scores and schedule, re-rates every team, and projects
today's and tomorrow's games (margin, total, win probability). Through the day it pulls prediction-market
prices, finds the best line for each outcome, flags arbitrage, and rebuilds a phone-friendly dashboard.

## The dashboard (`docs/cbb/index.html`)

| Tab | What it shows |
|---|---|
| Games | Each game: model line, best price per outcome across venues (ranked by value vs the model), all prices, and an **Add a price** button for apps without a feed |
| Arbitrage | Pairs of outcomes that can't both lose, bought for under $1 combined after estimated fees, including cross-line spreads/totals with middles |
| Edges | Every price at least N¢ below the model's probability |
| Record | Season and daily results of the model's pre-game lines vs DraftKings closing lines, running profit at your stake |
| Ratings | Every D-I team's adjusted offense, defense, net and tempo |

## Venues

| Venue | How prices get in |
|---|---|
| Kalshi | automatic (public API): moneyline, spread ladder, total ladder |
| Polymarket US | automatic (public API): moneyline |
| Novig | automatic once you add an API key (below): moneyline, spread, total |
| ProphetX | automatic once you add API keys (below): moneyline, spread, total incl. alternate lines |
| Robinhood | automatic, derived: Robinhood has no API but routes college basketball to Kalshi's order book, so its price is Kalshi's plus Robinhood's fees (≤1¢ commission + 1¢ exchange fee per contract). If Robinhood moves a game to its own exchange (Rothera), the in-app price can differ: use *Add a price*. Turn off with `ROBINHOOD_FROM_KALSHI=0`. |
| DraftKings Predictions, FanDuel Predicts, Fanatics Markets, Underdog, others | **by hand**: no public price feed exists. Tap *Add a price* on a game and type what the app shows (¢, $ or American odds). Saved on that device; the page recalculates best lines, edges and arbitrage immediately. Or list prices in `data/manual_quotes.csv` (columns `venue,date,away,home,market,side,line,ask,fee_coef`) and the next run includes them for everyone. |

## Set up on GitHub (one time)

1. Create a repo (or use your NBA board repo) and upload everything in this folder, including `.github/`.
2. Settings → Pages → Source: *Deploy from a branch* → branch `main`, folder `/docs`. The dashboard is at
   `https://<you>.github.io/<repo>/cbb/`.
3. Settings → Actions → General → Workflow permissions: *Read and write*.
4. Optional, for Novig: Novig → Profile → Settings → Novig API → Create Key. In the repo: Settings → Secrets
   and variables → Actions → add `NOVIG_KEY_ID` and `NOVIG_PRIVATE_KEY` (the full `.pem` text).
5. Optional, for ProphetX: prophetx.co → Settings → Token → create an access key + secret key, then add repo
   secrets `PROPHETX_ACCESS_KEY` and `PROPHETX_SECRET_KEY`. (Or, if ProphetX issues you a Market Data key,
   `PROPHETX_AFFILIATE_KEY`.) The job only reads markets; it never places orders.
6. Actions → *CBB dashboard* → *Run workflow* once to build it now. After that it runs hourly, 9am–11pm ET,
   November through April.

## Files

| file | what it does |
|---|---|
| `cbb/run.py` | the scheduled job: data + model once a day, prices + dashboard every run |
| `cbb/prep.py` | download + clean ESPN box scores and schedules (sportsdataverse mirror on GitHub) |
| `cbb/model.py` | opponent-adjusted efficiency + tempo ratings, refit daily with earlier games only |
| `cbb/daily.py` | rate every team as of a date and project upcoming games |
| `cbb/results.py` | model record vs DraftKings closing lines |
| `cbb/venues.py` | Kalshi, Polymarket US, Novig (signed requests) and ProphetX clients; Robinhood derived from Kalshi |
| `cbb/names.py` | team-name matching between venues and ESPN |
| `cbb/board.py` | match prices to games, score vs the model, best lines, arbitrage |
| `cbb/webpage.py`, `cbb/dashboard.html` | build the dashboard page |
| `cbb/backtest.py`, `cbb/tune.py`, `cbb/fetch_lines.py` | backtest and tuning (2021–2026 holdout) |
| `tests/test_board.py` | offline tests: parsers, matching, signing, arbitrage math |

Run locally: `pip install -r requirements.txt` then `python cbb/run.py`.

## How to read it

- **Price** = cost of a contract that pays $1. **Edge** = model probability − price − estimated fee.
- **Fees assumed**: Kalshi `0.07·P·(1−P)`; Polymarket US = each market's fee coefficient × `P·(1−P)`
  (public sources disagree on the US fee: check yours); Novig 0 before tip-off; ProphetX ~0.4% of stake
  (from the example in its fee docs; ProphetX sets each account's rate); Robinhood 2¢ per contract. Hand-entered prices use
  the fee you type.
- **Arbitrage** uses the best price on each side and assumes you can fill both legs at those prices.
  Books can be thin: the page shows available size where the venue publishes it.
- **The model has not beaten closing spreads** in a 2021–2026 out-of-sample test (about 0.2 pts per game
  worse than closing lines; worse early in the season). Treat edges as disagreements, not proven value.
  Arbitrage doesn't depend on the model.
- No injury or lineup data. Non-D-I opponents are skipped.
