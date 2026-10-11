# Robustness check: a 400-stock universe (pre-registered)

Committed **before** any 400-stock result existed. The 150-stock results, their protocol and their
verdicts remain the primary, pre-registered findings; this run asks only whether the conclusions hold on
a wider universe.

## What changes

Only the universe size: the 400 most liquid eligible stocks each day instead of 150, chosen point-in-time
exactly as before (index member on that date, enough history, price above $5, ranked by trailing dollar
volume). Roughly 300 stocks are eligible in 2010–2012, so the early years hold fewer than 400; no
survivorship filter is applied.

## What stays frozen

- Every hyperparameter from the 150-stock Fold 1 tuning (`runs/best_*.json`), with no retuning.
- Walk-forward folds, embargoes, seeds 0–2 for the networks, costs, portfolio layer and backtester.
- The trial counts used by the DSR (15 for the original models, 22 for the baselines), since no new
  configurations are tried.

The networks are retrained on the 400-stock data. Training now scores large blocks in checkpointed chunks,
which gives identical gradients and only saves GPU memory.

## Strategies and outputs

OU, MLP, temporal CNN, Transformer, ridge and gradient boosting, on 2020–2023 and then once on the
2024–2025 holdout. The same three rules and amendment rule 4 are reported, at 5 bps as before and also at
10 bps, because stocks ranked 150–400 by liquidity cost more to trade than a flat 5 bps.

## How it is read

The conclusions "hold" if the verdict wording on 2020–2023 at 5 bps is the same as for 150 stocks. Any
difference is reported as it comes out, next to the 150-stock results, and does not replace them.

## Presentation (decided before any 400-stock result)

Whatever the 400-stock numbers turn out to be, the dashboard will show the 400-stock run only, since it is
the wider and more realistic universe. The README and the research report keep the 150-stock results as
the pre-registered primary findings and show both universes side by side. This is only a choice of what
the dashboard displays; it does not change which result counts as primary.
