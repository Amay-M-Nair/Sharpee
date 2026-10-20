# Power analysis (400 stocks, net of 5 bps)

Minimum detectable effect (MDE) at 80% power, from stationary-bootstrap standard errors. Sharpe vs 0 is one-sided at 5%; differences are two-sided at 5%.

## 2020-2023 (4.0 years)

| Test | Strategy | Observed | SE | MDE | Years needed | Daily corr. |
|---|---|---|---|---|---|---|
| Sharpe vs 0 | OU | -0.04 | 0.58 | 1.44 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Ridge | -0.64 | 0.62 | 1.55 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Gradient boosting | -1.14 | 0.77 | 1.93 | never (observed ≤ 0) |  |
| Sharpe vs 0 | MLP | +0.40 | 0.49 | 1.22 | 37 |  |
| Sharpe vs 0 | Temporal CNN | +0.20 | 0.49 | 1.22 | 150 |  |
| Sharpe vs 0 | Transformer | +0.39 | 0.50 | 1.25 | 40 |  |
| Transformer minus | OU | +0.43 | 0.41 | 1.16 | 29 | 0.73 |
| Transformer minus | Ridge | +1.03 | 0.43 | 1.19 | 5 | 0.78 |
| Transformer minus | Gradient boosting | +1.53 | 0.62 | 1.74 | 5 | 0.81 |
| Transformer minus | MLP | -0.01 | 0.10 | 0.28 | 4,664 | 0.98 |
| Transformer minus | Temporal CNN | +0.20 | 0.17 | 0.49 | 25 | 0.95 |

## 2024-2025 (2.0 years)

| Test | Strategy | Observed | SE | MDE | Years needed | Daily corr. |
|---|---|---|---|---|---|---|
| Sharpe vs 0 | OU | -0.79 | 0.64 | 1.58 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Ridge | -1.81 | 0.70 | 1.73 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Gradient boosting | -2.78 | 0.76 | 1.90 | never (observed ≤ 0) |  |
| Sharpe vs 0 | MLP | -0.95 | 0.67 | 1.66 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Temporal CNN | -1.44 | 0.68 | 1.68 | never (observed ≤ 0) |  |
| Sharpe vs 0 | Transformer | -0.83 | 0.66 | 1.64 | never (observed ≤ 0) |  |
| Transformer minus | OU | -0.04 | 0.62 | 1.74 | 3,404 | 0.68 |
| Transformer minus | Ridge | +0.98 | 0.63 | 1.78 | 7 | 0.65 |
| Transformer minus | Gradient boosting | +1.95 | 0.63 | 1.76 | 2 | 0.70 |
| Transformer minus | MLP | +0.12 | 0.06 | 0.17 | 4 | 1.00 |
| Transformer minus | Temporal CNN | +0.61 | 0.15 | 0.42 | 1 | 0.97 |
