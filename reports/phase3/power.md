# Power analysis (150 stocks, net of 5 bps)

Minimum detectable effect (MDE) at 80% power, from stationary-bootstrap standard errors. Sharpe vs 0 is one-sided at 5%; differences are two-sided at 5%.

## 2020-2023 (4.0 years)

| Test | Strategy | Observed | SE | MDE | Years needed | Daily corr. |
|---|---|---|---|---|---|---|
| Sharpe vs 0 | OU | +0.15 | 0.57 | 1.41 | 369 |  |
| Sharpe vs 0 | Ridge | -0.32 | 0.63 | 1.56 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Gradient boosting | -1.77 | 0.84 | 2.10 | never (observed ≤ 0) |  |
| Sharpe vs 0 | MLP | +0.16 | 0.46 | 1.15 | 217 |  |
| Sharpe vs 0 | Temporal CNN | -0.12 | 0.46 | 1.15 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Transformer | +0.17 | 0.46 | 1.16 | 180 |  |
| Transformer minus | OU | +0.03 | 0.47 | 1.31 | 10,690 | 0.60 |
| Transformer minus | Ridge | +0.50 | 0.48 | 1.34 | 29 | 0.77 |
| Transformer minus | Gradient boosting | +1.94 | 0.78 | 2.19 | 5 | 0.67 |
| Transformer minus | MLP | +0.02 | 0.18 | 0.49 | 3,595 | 0.95 |
| Transformer minus | Temporal CNN | +0.29 | 0.20 | 0.55 | 14 | 0.95 |

## 2024-2025 (2.0 years)

| Test | Strategy | Observed | SE | MDE | Years needed | Daily corr. |
|---|---|---|---|---|---|---|
| Sharpe vs 0 | OU | -1.65 | 0.67 | 1.67 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Ridge | -0.81 | 0.60 | 1.49 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Gradient boosting | -2.17 | 0.55 | 1.36 | never (observed ≤ 0) |  |
| Sharpe vs 0 | MLP | -0.48 | 0.62 | 1.55 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Temporal CNN | -0.78 | 0.64 | 1.59 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Transformer | -0.55 | 0.61 | 1.51 | never (observed ≤ 0) |  |
| Transformer minus | OU | +1.10 | 0.68 | 1.90 | 6 | 0.56 |
| Transformer minus | Ridge | +0.26 | 0.59 | 1.67 | 80 | 0.64 |
| Transformer minus | Gradient boosting | +1.62 | 0.67 | 1.87 | 3 | 0.56 |
| Transformer minus | MLP | -0.06 | 0.13 | 0.37 | 71 | 0.98 |
| Transformer minus | Temporal CNN | +0.23 | 0.17 | 0.49 | 9 | 0.96 |
