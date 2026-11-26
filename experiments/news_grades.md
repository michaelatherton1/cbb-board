Graded 2026-11-26 07:51 CT

```
logged games: 27  graded (final, with a closing line): 16
      date                     away             home             lean  books_home_spread_flagged  model_home_spread  spread_close  close_toward_model  margin  model_side_covered
2026-11-23                    Akron     Wright State   supports_model                       -1.5               -7.0          -1.5               False     2.0                 1.0
2026-11-23                   Temple        UT Martin   supports_model                        2.5               -1.7           2.5               False     9.0                 1.0
2026-11-23 Texas A&M-Corpus Christi    Wichita State no_relevant_news                      -19.5              -16.3         -18.5                True    18.0                 1.0
2026-11-23                   Denver           Kansas  supports_market                      -32.5              -29.1         -31.5                True    17.0                 1.0
2026-11-23           UNC Greensboro       Pepperdine no_relevant_news                      -10.5               -6.7         -10.5               False    27.0                 0.0
2026-11-23           San José State     UC San Diego no_relevant_news                       -1.5                1.6          -2.5               False    -9.0                 1.0
2026-11-24             Wright State  St. Bonaventure  supports_market                       -0.5                6.2           1.5                True     2.0                 0.0
2026-11-24             Oregon State    San Francisco no_relevant_news                        9.5                1.9          -7.5                True    14.0                 1.0
2026-11-24                  Wyoming       Kent State no_relevant_news                        8.5               15.7          -9.5               False    -3.0                 0.0
2026-11-24                Creighton         Michigan  supports_market                       -6.5              -13.1          -6.5               False   -26.0                 0.0
2026-11-24             West Georgia   Tennessee Tech no_relevant_news                      -15.5               -9.8         -13.5                True    39.0                 0.0
2026-11-24             Oral Roberts     UT Arlington no_relevant_news                       -4.5              -10.4          -5.5                True    -4.0                 0.0
2026-11-24            UNC Asheville    Florida State no_relevant_news                      -28.5              -21.3         -30.5               False    36.0                 0.0
2026-11-24               Evansville         Missouri no_relevant_news                      -32.5              -26.8         -31.5                True    11.0                 1.0
2026-11-24              Sam Houston Washington State no_relevant_news                       -4.5                0.6          -6.5               False     7.0                 0.0
2026-11-24             Grand Canyon      Saint Louis no_relevant_news                        1.5               -6.7          -1.5                True     3.0                 1.0

by lean (close-toward-model share uses only games whose line moved):
                  games  close_toward_model  model_side_cover
lean                                                         
no_relevant_news   10.0                 0.6              0.45
supports_market     2.0                 1.0              0.33
supports_model      NaN                 NaN              1.00
```

Note: main's results_games.csv (dashboard build 2026-11-26T03:12Z, still the latest on main at grading time) has games
through 2026-11-24 plus 2 from 2026-11-25, so the six Nov 25 notes and today's five Nov 26 notes are not graded yet.

## Orientation check (home/away swapped in results for 3 neutral-site games)

grade_news.py joins on game_id only. For 3 graded neutral-site games, results_games.csv lists home/away the opposite
way from the news log, so spread_close and margin are compared from the other team's side:

| date | log: away @ home | results: home | as graded above | after flipping close and margin to the log's side |
|---|---|---|---|---|
| 2026-11-24 | Oregon State @ San Francisco | Oregon State | close -7.5, toward model True, model side covered 1 | close +7.5, toward model True, covered 0 |
| 2026-11-24 | Wyoming @ Kent State | Wyoming | close -9.5, toward model False, covered 0 | close +9.5, toward model True, covered 0 |
| 2026-11-24 | Grand Canyon @ Saint Louis | Grand Canyon | close -1.5, toward model True, covered 1 | close +1.5 (no move), covered 0 |

Re-computed with those 3 flipped (scratch calculation; grade_news.py itself was not changed):

| lean | graded | line moved | close toward model | model side covered |
|---|---|---|---|---|
| no_relevant_news | 11 | 9 | 6 of 9 (0.67) | 3 of 11 (0.27) |
| supports_market | 3 | 2 | 2 of 2 (1.00) | 1 of 3 (0.33) |
| supports_model | 2 | 0 | - | 2 of 2 (1.00) |
| all | 16 | 11 | 8 of 11 (0.73) | 6 of 16 (0.38) |

16 graded games is noise; nothing here should be read as a result until ~40-60 games.

## NEW ABSENCE notes

None graded yet. No logged note so far (Nov 23-26, 27 notes) starts with "NEW ABSENCE": no top-2-3-minutes player was
found newly out in any flagged game. Closest earlier case: Nov 25 Northeastern @ Wisconsin - rotation guard Jack Janicki
(~17.8 mpg, not top-3 minutes) newly out for the season; logged as supports_model, ungraded (Nov 25 results not on main yet).
