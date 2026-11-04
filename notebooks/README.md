# Notebooks: reading guide

Read them in order. Each one opens with a short table (question, data, answer, next) and ends
with **Takeaways**, so you can get the gist from the first and last cells and dip into the
middle for the method. Every table and figure has a short "how to read it" note under it.

| # | Notebook | The question in plain words | Main figures |
|---|---|---|---|
| **Part 1 · One e-mail test, from plan to rollout** | | | |
| 01 | [Design and trust](01_email_test_checks.ipynb) | Was the test big enough, and did the random split work? | Spend distribution and balance; raw averages by group |
| 02 | [Readout, decision, rollout](02_email_test_readout.ipynb) | What did each e-mail do, and which should we send? | Effect of each e-mail with intervals; value per 1,000 e-mailed; holdback trade-off; bootstrap and Bayesian cross-checks |
| 03 | [Targeting](03_email_test_targeting.ipynb) | Should different customers get different e-mails? | What targeting seemed to gain vs what held up on new customers |
| **Part 2 · A whole testing programme (Upworthy)** | | | |
| 04 | [Sample-ratio check](04_programme_sample_ratio.ipynb) | Would an automatic check have caught a known randomization bug? | Flag rate by month; p-values inside vs outside the problem window; test power and false discoveries |
| 05 | [Winners](05_programme_winners.ipynb) | How much of a winning headline's lift is real? | The winner's curse; shrinkage vs measured lift |

The [StreamFlix appendix](streamflix/README.md) runs the same workflow on simulated data where
the true answer is known.

**Running them.** Each `.ipynb` is generated from the `.py` file beside it (the `.py` is the
source). `python scripts/build_notebooks.py --execute` rebuilds them all with outputs; it needs
the Hillstrom file in `data/raw/` (see [`data/README.md`](../data/README.md)).
