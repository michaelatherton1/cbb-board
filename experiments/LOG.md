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

## EXP-002 — Blend the model with the opening line (2026-11-23) — REJECTED at research stage (holdout not spent)

**Idea.** Early in the season the market knows things the model doesn't. Predict
`w * model + (1 - w) * (-opening spread)` (and the same for totals), with w chosen before any holdout run.

**Scorecard change (made before any EXP-002 result existed).** `experiments/scoreboard.py --post module:function`
applies a post-backtest transform before scoring.

**Data problem found by the researcher.**
- The ESPN/hoopR mirror has no opening lines for 2013-2023. Coverage: 0% in every season.
- Only 2026 (DraftKings, 99%) and the live 2027 season (944/944) have opens.
- Totals for 2013-2017 are junk (mean |total - close| 43-81 pts).
- 2-4% of older neutral-site spreads are sign-flipped.
- The researcher's w=0.5 came from simulated opening lines, not real ones.
- Leak found: `backtest.attach_lines` sets the neutral-site *opening* spread's sign from the *closing* moneyline.
  Anything blending with the open must read raw `spread_open` from `lines_<season>.parquet`.

**Lead measurement on real opens (2027, never part of the holdout; raw opens; pre-game model lines from the Record tab).**

| w (model weight) | 0 (open only) | 0.1 | 0.25 | 0.5 | 1 (model only) |
|---|---|---|---|---|---|
| Margin MAE vs result, all 944 | 9.753 | 9.766 | 9.811 | 9.935 | 10.348 |
| Non-neutral only (828) | 9.742 | 9.765 | 9.825 | 9.979 | 10.447 |
| Days 0-7 (329) | 9.229 | 9.275 | 9.358 | 9.567 | 10.149 |
| Days 8-22 (615) | 10.033 | 10.029 | 10.053 | 10.131 | 10.455 |

- Best w: 0.00 vs results and 0.00 vs the close.
- Date-bootstrap: median 0.00, 80% range 0.00-0.05.
- Totals: w=0.25 gives 13.247 vs 13.251 for the open alone (noise).
- The closing line is better still: 9.570.

**Decision.** This season the model adds nothing to DraftKings' opening line, so a blend would just be the line. Not
shipped. The 2026 holdout was not run, so it stays unspent for a stronger candidate.

**Implications.**
- The dashboard's model-vs-price edges are mostly model error this early in the season. A no-vig sportsbook
  consensus is the better fair price for judging Kalshi/Polymarket/Novig prices (the local scanner already
  does this).
- Revisit after ~day 45 of 2027. The researcher's 2018-20 close-blend diagnostic and the days 8-22 hint
  (w=0.1 ties) suggest the model's weight may rise as the season goes on.
