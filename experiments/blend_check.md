# EXP-007: monthly opening-line re-check

See experiments/blend_check.py for the method and the pre-registered rule.
## 2026-11-23 12:51 CT: season 2027, through day 21

Margin MAE vs result by days into season (w = weight on the model; 0 = opening line only):

| days   |   games |   model |   open |   close |   best_w |   blend | w_80pct   | signal   |
|:-------|--------:|--------:|-------:|--------:|---------:|--------:|:----------|:---------|
| 0-14   |     637 |  10.08  |  9.352 |   9.133 |      0   |   9.352 | 0.00-0.00 | False    |
| 15-30  |     307 |  10.906 | 10.585 |  10.477 |      0.1 |  10.573 | 0.05-0.20 | False    |
| all    |     944 |  10.348 |  9.753 |   9.57  |      0   |   9.753 | 0.00-0.05 | False    |

Totals: 944 games, model 13.550 | open 13.251 | best w 0.15 -> 13.240

**Re-open EXP-002? no** (rule: a bucket with >= 300 games, best w >= 0.15, 80% range excluding 0)
