# Phase 3 protocol (pre-registered)

This file was committed **before any neural-model result on the 2020–2023 test years existed**; the
commit timestamp on GitHub is the proof. It fixes what will be measured and what counts as a win, so the
rules cannot be adjusted after seeing the results. Any deviation will be listed in the results with its
reason.

## 1. What is frozen

| Item | Frozen as |
|---|---|
| Data | Raw snapshot `20261009_130137`; universe, eligibility and residual construction as in Phase 1 |
| Portfolio layer | Cap each residual bet at 2% before hedging (`x = Φᵀw`); model scores centered, OU positions not; gross ≤ 1 |
| Costs | 5 bps per unit of traded notional; weights set at close *t* earn *t → t+1* |
| OU baseline | Avellaneda–Lee rules, window 60, entry 1.25, exits +0.75 / −0.50, κ > 252/30 (Fold 1 trial `12c3561748`) |
| MLP | hidden 32, dropout 0.1, lr 0.001 (trial `d7721835c8`) |
| Temporal CNN | channels 16, dropout 0.1, lr 0.0003 (trial `691daabcfe`) |
| Transformer | d_model 32, 4 heads, 2 layers, d_ff 128, dropout 0.1, lr 0.001 (trial `a7b7faa58f`) |
| Training | lookback 30, blocks of 64 days, max 40 epochs, patience 6, AdamW (weight decay 1e-5), grad clip 1.0, 5 bps costs inside the loss |
| Seeds | 0, 1, 2 for every neural model |

No hyperparameter is re-tuned in Phase 3.

## 2. Walk-forward

| Fold | Train | Validate (early stopping only) | Test |
|---|---|---|---|
| 1 | 2010–2017 | 2018–2019 | 2020 |
| 2 | 2010–2018 | 2019–2020 | 2021 |
| 3 | 2010–2019 | 2020–2021 | 2022 |
| 4 | 2010–2020 | 2021–2022 | 2023 |

The last 30 trading days of each training and validation period are dropped (embargo). Each fold trains
every model from scratch. The four test years are joined into one 2020–2023 daily return series per
strategy.

## 3. Primary hypothesis

- **Primary candidate: the Transformer.** It was chosen in Phase 2 on validation data only. The MLP and
  temporal CNN are reported as secondary results, and no claim is made about them.
- **Primary series:** the seed-combined Transformer portfolio, with one third of capital in each seed's
  book, so its daily net return is the average of the three seeds' net returns at 5 bps, 2020–2023.
- **Primary metric:** the annualized net Sharpe ratio of that series.
- **Comparison:** the OU baseline on the same days, same costs and same backtester.

## 4. Decision rules

All three use per-period (daily) Sharpe ratios internally and a stationary bootstrap with a mean block
length of 10 days, 2,000 resamples and random seed 0.

| # | Question | Passes if |
|---|---|---|
| 1 | Is there an edge after costs? | Probabilistic Sharpe Ratio, P(true Sharpe > 0) ≥ 0.95 |
| 2 | Does it survive the selection of 15 configurations? | Deflated Sharpe Ratio ≥ 0.95, with N = 15 (the Fold 1 trial log) and V[SR] = 3.46e-4, the variance of those trials' daily validation Sharpes |
| 3 | Does it beat OU? | The paired bootstrap 95% interval of (Sharpe of Transformer − Sharpe of OU), resampling the same days for both, excludes 0 |

Verdict wording, fixed in advance:

- Rules 1, 2 and 3 pass: "beats OU after costs and survives multiple testing".
- Rule 1 passes but not 3: "positive after costs, but not distinguishable from OU".
- Rule 3 passes but not 1: "better than OU, but no reliable positive edge".
- Neither 1 nor 3: "no reliable edge after costs".

Rule 2 is reported alongside whichever applies. The verdict is reported whatever it is.

## 5. Secondary results (reported, no decision attached)

- Cost sweep at 0, 5, 10 and 20 bps, and a one-day execution delay, for every strategy.
- Per-year Sharpe, turnover, beta, max drawdown, and the spread across seeds.
- **Ablation, costs in the loss:** the Transformer retrained with `train_cost_bps = 0` (same everything
  else) and evaluated at 5 bps. Expected if the Phase 2 explanation is right: higher turnover and lower
  net Sharpe than the cost-trained version.
- Interpretation on the test years: network scores against the OU s-score, and Transformer attention maps.

## 6. Holdout

2024–2025 (train 2010–2021, validate 2022–2023, test 2024–2025) is run **once**, after every result
above has been committed, with the same frozen settings, and reported as it comes out.

## 7. Disclosures

- OU has no trained parameters, so its 2020–2023 results were computed in Phase 1 and recomputed after
  the portfolio-layer fixes. No neural-model setting was chosen using 2020–2023 data.
- Two portfolio-layer bugs (a cap applied after hedging, and uncentered model scores) were found and fixed
  in Phase 2 using validation data only. The runs affected are archived and excluded from the trial log.
- The DSR benchmark comes from validation-period Sharpes and is applied to test-period returns, which is
  the conservative choice.
