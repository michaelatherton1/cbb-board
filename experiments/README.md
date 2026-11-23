# Model experiments (multi-agent)

One idea at a time, judged by code, not by the agent that proposed it.

| Role | Who | Instructions |
|---|---|---|
| Lead | the coordinating Claude session | picks the idea, runs the scorecard, keeps or reverts, writes `LOG.md` |
| Research | `.claude/agents/model-researcher.md` | diagnoses misses on 2012-2020 only, proposes ONE change |
| Engineer | `.claude/agents/model-engineer.md` | implements it leak-free on a branch, adds tests |
| Scorecard | `experiments/scoreboard.py` (locked) | walk-forward backtest, 2021-2026 holdout, by season / days into season / vs closing line |
| Skeptic | `.claude/agents/model-skeptic.md` | independent audit: leakage, overfitting, effect size, production parity |

Rules
- Ideas are chosen on 2012-2020. The 2021-2026 holdout is run once per finished candidate, by the lead.
- Nobody edits `experiments/scoreboard.py`, `cbb/backtest.py`, or the TRAIN/TEST seasons inside an experiment.
- Every experiment, kept or not, gets an entry in `LOG.md`.
- Work happens on `agents/exp-NNN-*` branches; merging to `main` is the owner's call.

Running the scorecard
```
pip install -r requirements.txt
python cbb/prep.py                       # after downloading seasons (see cbb/run.py / prep.download)
python cbb/fetch_lines.py 2021 2022 2023 2024 2025 2026 2027   # closing lines (none exist for 2024-25 in the source)
python experiments/scoreboard.py baseline
```

Note: running `tests/test_board.py` rewrites tracked files in `data/out/` (pre-existing behaviour). Restore them
with `git checkout -- data/out` before committing.
