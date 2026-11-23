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

## EXP-006 — Totals: separate prior strength for team style/pace (2026-11-23) — KEEP (pending owner merge)

**Question.** Totals use the same ridge penalty as margins (lam_eff on O and D). Is a team's scoring
style (O+D)/2 more stable than its quality (O-D)/2, so that it should be pulled harder toward the prior?

**Research (2012-2020 only).** Walk-forward grid over lam_style (4, 8, 12, 16, 20, 30) and a style carry-over.
lam_style = 12 was best: total MAE 13.544 -> 13.508 (2012-17, calibration fitted on 2012-17) and
13.526 -> 13.495 (2018-20). Margins are mathematically unchanged. The style carry-over added nothing and was dropped.

**Change.** `cbb/model.py`: 2x2 penalty block per team (lam_eff on (O-D)/2, lam_style on (O+D)/2);
`Params.lam_style` (None = old model, bit-identical). `data/best_params.json`: lam_style = 12.
Tests: `tests/test_style_penalty.py`; `tests/test_prior_parity.py` now also checks totals. 25 tests pass.

### Pre-registration (committed before the holdout run)
Candidate: `lam_style = 12` (separate ridge penalty on the style/pace direction s=(O+D)/2; quality direction unchanged), chosen on 2012-2020 only.
Run once: `python experiments/scoreboard.py exp006_lam_style12` vs `baseline.json` (2021-2026 holdout).
KEEP only if all three hold:
1. Holdout total MAE improves by >= 0.020 (baseline 13.278 -> <= 13.258).
2. Total MAE improves in >= 4 of the 6 holdout seasons.
3. Margin MAE is no worse by more than 0.005 (baseline 9.046 -> <= 9.051).
Reported but not part of the decision: total MAE vs the closing total where a usable close exists (2021-23, 2026).
Otherwise REJECT. No re-tuning after the holdout is seen.

### Result (holdout run once, 2021-2026, n = 32,367)
| | Baseline | lam_style = 12 |
|---|---|---|
| Total MAE | 13.278 | **13.226** (-0.0515; 95% CI by date -0.066 to -0.037) |
| Seasons improved | | **6 / 6** (-0.034 to -0.080) |
| Margin MAE | 9.0457 | 9.0457 (identical) |
| Vs closing total (2021-23 + 2026, n = 19,904; market 13.064) | 13.316 | 13.261 |

All three pre-registered criteria are met. Production `data/calibration.json` refit on all 2012-2026 seasons:
only the totals terms change (bt [0.826, 24.26] -> [0.884, 16.00]; total_sd 17.04 -> 16.99).

### Skeptic (independent): PASS WITH NOTES
- No leakage: research used 2012-2020 only; calibration in the backtest is fit on TRAIN only; the optimum is flat for
  lam_style 8-16; the holdout gain is larger than the research gain.
- **Process gap (disclosed):** the engineer's reproduction script called `backtest.run`, which computes 2021-2026
  predictions too (`data/backtest_exp006_check.parquet`, 13:07), before the pre-registration was committed (17:05). The
  lead's written criteria (13:06) predate it, the script deleted the frame and filtered to <= 2020, and no holdout
  metrics were printed; no evidence anyone looked. Rule from now on: reproduction scripts run on a games slice ending 2020.
- Margins are invariant on the production path, but NOT under a RosterPrior override (2015-16 test: mean |diff| 0.097).
  Docstring updated. `scoreboard --roster` baselines shift after merge.
- Live 2027 check (not holdout, n = 945, Nov 1-22): total MAE 13.552 -> 13.516 (-0.036; 95% CI -0.151 to +0.094). Same sign,
  too noisy to confirm. Both are worse than the opening total (13.251) and the close (13.041).
- **The model adds no information beyond the closing total** (regression coefficient 0.01). Holdout O/U win rate is below
  the 52.4% breakeven at every edge threshold. EXP-006 is a modelling improvement, not a betting edge: totals edges on
  the dashboard should be labelled not usable, hidden, or given a much higher threshold.

### If merged, expect
- Ratings tab: adj_em and ranks unchanged; adj_o / adj_d move ~1 pt/100 on average (up to 4.1).
- Record tab: the 2027 O/U record is recomputed with the new model (all games 52.3% -> 51.8%; |edge| >= 3: 53.7% -> 55.3%).
- Today's projections stay on the old model unless `data/out/model_<date>.json` is removed and the workflow re-run.
- `experiments/results/baseline.json` must be re-run for totals. `cbb/tune.py` writes `asdict(Params())` and would reset
  lam_style to null; re-add it after any re-tune.

