# Experiment log

## EXP-001 — Early-season priors → production/backtest mismatch fix (2026-11-23) — KEEP (pending owner merge)

**Target.** Baseline holdout (2021-2026, 32,367 games) margin MAE 9.046. Versus the closing line (2021-23 + 2026,
19,880 games): model 9.007 vs market 8.819, gap +0.19. The gap is largest early in the season:

| Days into season | Model | Closing line | Gap |
|---|---|---|---|
| 0-14 | 9.859 | 9.365 | +0.494 |
| 15-30 | 9.330 | 9.077 | +0.254 |
| 31-60 | 9.161 | 8.929 | +0.232 |
| 61+ | 8.748 | 8.640 | +0.108 |

**Research (2012-2020 only).**
- The carry-over prior (r_eff=1.0, w_conf=0.3) is already near-optimal as a prior. Every prior tweak tried
  (lam_eff, w_conf, blowout cap, roster prior, roster+conference) moved 2018-20 days 0-30 by ≤0.04.
- The roster-continuity prior as built is slightly worse (no conference term); with a conference feature, it is
  −0.020 ± 0.017 on 2018-20, which is noise. Left off.
- About 0.3 pts of the early-season gap can't be removed by any prior: even each team's own final ratings
  (a hindsight floor) miss more in days 0-14.
- **Main finding: `cbb/daily.py` called `make_prior(last, teams, p)` without `conf=`.** So the live projections
  ignored the conference shrinkage (w_conf=0.3) that the backtest and the Record tab (`results.py` uses
  `walk_forward`) both apply. The dashboard was grading one model and serving another.

**Change (engineer).** `prep.conf_map(y)` (factored out of `load_season`, behaviour unchanged). `daily.season_model(...)`
builds the prior with this season's conference map from the schedule and rates every scheduled D-I team.
New `tests/test_prior_parity.py`: production path equals `walk_forward` to ~2e-13 (2020, days 0/3/7/14/30). It
skips without local data. With the old code, the test fails by 1.9-6.5 pts.

**Scorecard.** Holdout unchanged at 9.046, as expected: the backtest already applied the conference term.

**Live season 2027 (never used for tuning), old vs new production path replayed per date, 944 games with lines:**
- Margin MAE 10.467 → 10.348 (market 9.567).
- Paired change −0.119; 95% interval resampled by date [−0.231, −0.007].
- Improves in every week so far (−0.108, −0.112, −0.121, −0.236) and on 16 of 22 dates.
- RMSE 13.11 → 12.92.
- ATS at 3+ pts vs close: 42.7% → 44.0%. Still losing.

**Skeptic: PASS WITH NOTES.**
- No leakage. Conference membership is stable within each season and known by Dec 1 in 14 of 16 seasons.
- Teams with 0 games add no rows to the fit.
- Totals also shift (mean 0.53 pts). 337 of 365 teams change rank (max 39 places).
- Early-season ratings CSV now lists all scheduled D-I teams (games=0 for some), which shifts meta `league_eff`
  by 0.2-0.3 in the first days. Projections are unaffected.
- The parity test gives no CI protection (needs local parquet).

**Effect when merged:** today's live spreads move by a mean of 0.89 pts (max 2.9) toward the backtested model.

**Still open:** the model remains ~0.8 pts worse than the closing line this season. This fix is about 15% of that
gap. Next candidates:
1. Roster prior + conference feature, re-tested for the transfer-portal era (2021+ can't be used to choose; would
   need a pre-registered single holdout run).
2. Market-informed early season (blend with opening lines).
3. A forward-tracked injury/news overlay.

## EXP-005 — Kalshi spread/total ladders priced from the sportsbooks (2026-11-23) — NO EDGE FOUND (research tool; not proposed for main)

**Question.** Kalshi lists a ladder of spread and total rungs per game ("wins by over X.5", "over X.5 points").
Priced from the DraftKings/FanDuel/BetMGM main lines with a fitted distribution, are any rungs cheap after Kalshi's fee?

**Distribution (Part A).** Fitted on closing lines, 2018-2020.
- Spreads: split normal from the favorite's side, sigma depending on total, early season, neutral site and |spread| (capped at 30),
  plus weights on whole-number margins (no ties; 1-point margins x0.86 vs a Normal).
- Totals: split normal (x0.934 below the line, x1.019 above).
- Season-out log loss improves over a plain Normal (spreads near zero margin 0.52366 -> 0.52242; totals 15+ from the line 0.21815 -> 0.21668).
- Weakness: rungs 20+ points past the line are ~0.5pp too thin.
- **Disclosure:** the agent first scored candidate shapes on 2021-2023 (per its original brief) before the "2012-2020 only" rule reached it.
  It then re-selected on 2018-2020 alone (each season held out in turn) and the same shapes won. All shipped parameters are fitted on
  2018-2020 only, but treat 2021-2023 as seen for this experiment.

**Tool (Part B).** `cbb/ladder.py` (prices every rung from the median de-vigged book center; edge after fee; Kalshi's own implied
center); `tests/test_ladder.py` (52 pass); research scripts in `experiments/exp005/`. No existing files changed.

**Today's board (Part C, run 18:12Z, books pulled 17:30Z).**
- 1,769 rungs in 30 games: edge after fee median -2.6c, 95th percentile +0.2c.
- 8 rungs at >= 2c; 5 of those vanish against Kalshi's own center (the book pull was 47 min stale).
- One rung at >= 3c with $10+ available: Illinois -30.5 vs Winthrop (ask 81c, fair 85.1%, +3.05c).
  Not trusted: a 43-point favorite, 2 books a point apart, past the data range; only +1.9c against Kalshi's center.
- Positive edges do not cluster in the tails.
- Kalshi's mids are 0.3-0.6c cheaper on long-shot rungs: within model error and below trading cost (~1c half-spread + ~1.5c fee).

**Verdict.** No evidence that Kalshi's ladders are mispriced versus the books after fees. One day is a snapshot, not a test.
No skeptic review, because nothing is proposed for the dashboard or main.

**To make it a real test (not started):**
- Log every rung each run, with the book pull time, and only count rungs whose book pull is within ~15 min.
- Grade against fair recomputed from the last pull before tip (closing value), not wins.
- Compare our fair vs Kalshi's mid on results by distance from center.
- `venues.py` doesn't store the NO-side size (= YES bid size); a one-line fix.

