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

## EXP-004 — What lineup/injury information is worth (2026-11-23) — RESEARCH ONLY: small lever, not built

**Method (2012-2017 fit, 2018-2020 evaluation, holdout untouched).**
- Each player gets a leak-free Game Score-per-minute value (prior games plus half of last season, shrunk toward
  0.8x league).
- A team's lineup delta = minutes-weighted value of tonight's available players minus its season-to-date average.
- Margin adjustment = β·(Δ_home − Δ_away).
- Oracle = who actually played (hindsight ceiling). Proxy = a player who appeared this season but missed the team's
  previous game is "out".

**2018-2020 margin MAE vs result (16,468 games).**

| | 0-14 | 15-30 | 31-60 | 61+ | all |
|---|---|---|---|---|---|
| baseline | 9.582 | 9.161 | 9.029 | 8.667 | 8.915 |
| oracle (who played) | 9.570 | 9.163 | 9.015 | 8.636 | 8.894 |
| proxy (missed last game) | 9.578 | 9.165 | 9.029 | 8.651 | 8.907 |

- Gains: oracle 0.021 (SE 0.005), proxy 0.0085 (SE 0.0037).
- vs the closing line: baseline 1.677, proxy 1.664.
- The proxy explains 1.7% of the variance in (market − model).
- Even perfect lineup knowledge is far under the 0.1-pt bar. Game noise (SD ~11) swamps lineup effects in MAE.

**The part that matters for betting (event study).**
- When a team's top player (~32 mpg) is newly out, the team underperforms the model by 2.4-2.5 pts
  (n=372 in 2018-20, 611 in 2012-17).
- The closing line moved only 0.4-0.6 pts in those games.
- About half of key-player absences are new (he played the previous game), so the proxy can't see them.
- "Missed last game → misses this game" hits 62.5%, rising to 86% after 5+ straight absences.

**Decision.** Don't build the lineup adjustment into the model: it's tiny for accuracy, and production would need
daily player box downloads. Keep the value where it is concentrated: the daily news check (EXP-003) should look
hardest for NEWLY absent top players. Those are the games where the market underreacts.
Scratch code: /home/claude/scratch/exp004/.
