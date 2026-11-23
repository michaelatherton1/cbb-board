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

## EXP-008 — Roster-continuity prior with a conference term, transfer-portal era (2026-11-23)

**Pre-registration (committed before the holdout run).**
- **Candidate:** `RosterPrior(conf_feature=True)` (env `ROSTER_CONF_FEATURE=1`) with production params. It is the
  existing roster prior plus the current-season conference average of last season's ratings.
- **2012-2020 evidence:** days 0-30 margin MAE 9.358 vs 9.376 (−0.018 ± 0.016). Not significant; the hypothesis is
  that roster turnover matters more since the portal (2021+).
- **Single holdout run:** `ROSTER_CONF_FEATURE=1 python experiments/scoreboard.py exp008_roster_conf --roster`.
- **KEEP only if all hold:**
  1. Holdout days 0-30 margin MAE (2021-2026) improves vs baseline by >= 0.03.
  2. Days 0-30 improves in >= 4 of 6 holdout seasons.
  3. Overall holdout margin MAE is not worse than baseline (9.046).
- Otherwise REJECT. No re-tuning after the run.

**Holdout result (single run, `exp008_roster_conf`).**
- Overall margin MAE: 9.0419 vs 9.0457 (−0.004).
- Days 0-30 (8,040 games): −0.0139 ± 0.0235 (95%, game-level).
- By season (days 0-30): 2021 +0.021, 2022 −0.032, 2023 −0.015, 2024 −0.022, 2025 −0.029, 2026 +0.005. That's 4 of 6
  improved.
- Total MAE: 13.284 vs 13.278.

**Verdict: REJECT.** Criterion 1 fails (−0.014, short of the pre-registered −0.03). Criteria 2 and 3 pass. The
portal-era seasons do lean the right way (2022-25 all −0.015 to −0.032), but the effect is small and not
significant. No re-tuning. The code stays on this branch, off by default, in case a richer roster model (player
value rather than minutes share) is tried later. See EXP-004.
