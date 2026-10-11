# Phase 3 amendment: non-neural baselines (pre-registered)

This amendment was committed **after** the Phase 3 results were published but **before** any of the
models below were run on the 2020–2023 test years or the 2024–2025 holdout. It does not change the
original protocol ([`phase3_protocol.md`](phase3_protocol.md)) or its verdict: the Transformer's three
rules, including its Deflated Sharpe Ratio with N = 15, are reported exactly as before.

## Why

The original comparison shows that the neural models roughly match the OU rule after costs. It cannot
say whether **the neural part matters**, or whether any learned model on the same inputs would do the
same. Three baselines answer that.

## The baselines

All three use the same inputs (each stock's 30-day cumulative residual path, scaled by its own
volatility), the same portfolio layer (scores centered across stocks, each residual bet capped at 2%
before hedging, `x = Φᵀw`), the same 5 bps costs, the same walk-forward folds and 30-day embargoes, and
are tuned on Fold 1 validation (2018–2019) only, with every configuration logged in the trial log.

| Baseline | Model | Trained on | Tuning grid |
|---|---|---|---|
| **Ridge** | Ridge regression | Next-day residual payoff of each stock's hedged bet, z-scored across stocks each day (clipped at ±5) | alpha ∈ {1, 100, 10,000} |
| **Gradient boosting (GBM)** | scikit-learn `HistGradientBoostingRegressor`, 200 iterations, min 200 samples per leaf, L2 1.0 | Same target as ridge | max leaf nodes ∈ {15, 63} × learning rate ∈ {0.03, 0.1} |
| **Linear, Sharpe-trained** | One linear layer (30 → 1), no hidden layers | The same Sharpe-after-costs loss and training settings as the networks | learning rate ∈ {0.001, 0.0003} |

Ridge and GBM are what standard machine learning does: forecast, then trade the forecasts. They are
deterministic, so each is run once per fold. The linear model is trained exactly like the networks, with
seeds 0, 1 and 2 combined, so that comparing it with the Transformer isolates depth and nonlinearity
while holding the objective fixed.

## Questions and rules

The Transformer remains the primary candidate. Using the same paired stationary bootstrap as rule 3
(mean block 10 days, 2,000 resamples, seed 0), on 2020–2023 net of 5 bps:

| # | Question | "Yes" if |
|---|---|---|
| 4 | Does the Transformer beat gradient boosting? | The 95% interval of Sharpe(Transformer) − Sharpe(GBM) is above 0 |
| 5 | Does depth and nonlinearity beat a linear rule trained the same way? | The 95% interval of Sharpe(Transformer) − Sharpe(Linear, Sharpe-trained) is above 0 |

Ridge is reported alongside, with no rule attached. Each baseline also gets the same PSR, DSR and
bootstrap interval as the other strategies; **its DSR counts every configuration in the trial log
(15 original + 9 new = 24)**, while the original models keep N = 15 as pre-registered.

## Holdout

Each baseline is run once on 2024–2025 with its frozen Fold 1 settings, and reported as it comes out.

## Deviation (recorded after the results)

The linear control and its rule 5 were **removed at the author's request after the amendment results
had been seen**. Its results (a net Sharpe of +0.00 on 2020–2023 and −1.04 on 2024–2025, with the
Transformer's advantage over it not distinguishable from zero) remain in the git history and its runs
are archived, not deleted. Its 2 configurations were taken out of the trial log, so the remaining
baselines' DSR counts 22 configurations. Without this control, the amendment can no longer separate how
much of the Transformer's advantage comes from its objective and how much from its architecture.
