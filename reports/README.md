# Phase 1 findings: the evidence behind Sharpee's design

Each figure below comes from the notebooks ([01 data exploration](../notebooks/01_data_exploration.ipynb), [02 residual analysis](../notebooks/02_residual_analysis.ipynb)) or from `scripts/run_baseline.py`, and rerunning them regenerates it. Every section covers what the data shows and which design decision it justifies.

---

## 1. The data is incomplete in the past, so the universe must be point-in-time

![Membership coverage](figures/01_membership_coverage.png)

**What it shows.** The S&P 500 always has about 500 members. Yahoo has prices for only 67% of 2008's members, rising to 98% by 2025. The missing companies are the ones that later left the index: delisted, acquired or merged.

**Design decision.** Membership comes from a historical list, and eligibility is decided day by day. A full-period history filter would quietly remove every company that later disappeared. The remaining gap can't be fixed with free data, so it's disclosed as survivorship bias that likely flatters results.

---

## 2. A fixed-size, liquid universe after a warm-up period

![Eligible universe](figures/01_eligible_universe.png)

**What it shows.** Once the 312-day history requirement is met (March 2009), 300–500 stocks pass the membership, history and price rules each day. The 150 most liquid of them form the universe.

**Design decision.** 2008–2009 is used only as warm-up for the PCA and OU windows, so no trading or training happens before 2010. The universe has a constant size, which keeps every model's input the same shape. Ranking by liquidity keeps strategies in stocks they could actually trade, and it also screens out reused tickers whose Yahoo prices belong to unrelated securities (checked in notebook 01).

---

## 3. Returns are heavily fat-tailed

![Return tails](figures/01_return_tails.png)

**What it shows.** Excess kurtosis is about 20. Days beyond 4 standard deviations are about 100 times more common than a normal distribution predicts.

**Design decision.** Sharpe ratios computed on fat-tailed returns are less reliable than they look. That's why Phase 3 uses the **Probabilistic and Deflated Sharpe Ratios**, which adjust for skewness and kurtosis. It's also why a 2% single-name cap stops any one stock's jump from dominating the portfolio.

---

## 4. Markets change regime

![Market regimes](figures/01_market_regimes.png)

**What it shows.** Volatility spikes in 2008, 2011, 2020, 2022 and 2025, with long calm stretches in between.

**Design decision.** One backtest number can hide regime dependence, so the models are evaluated with **walk-forward retraining, one test year at a time**, and results are reported per year. 2024–2025 stays an untouched holdout.

---

## 5. Common factors explain about half of all movement

![Factor variance](figures/02_factor_variance.png)

**What it shows.** The first PCA factor (roughly "the market") explains about 37% of the variance on average, and the top 5 factors about 52%. Both rise in crises, when stocks move together, and fall in calm markets.

**Design decision.** Strategies trade **residuals**: what is left after removing these factors. Because the factor structure drifts, PCA is **refit every month** using only past data. Each position is held as stock weights `x = Φᵀw`, meaning long the stock and short its factor hedge, so the portfolio stays close to factor-neutral (beta 0.04 in the baseline).

---

## 6. The core evidence: residuals mean-revert, weakly but consistently

![Residual autocorrelation](figures/02_residual_autocorrelation.png)

**What it shows.** Raw-return autocorrelations swing between positive and negative from lag to lag. That pattern is market-wide noise, and it disappears once the factors are removed. Residual autocorrelation is small (about −0.01) but **negative at every lag from 1 to 10 days**.

**Why this is the basis of the strategy.** A stock that drifted away from its factor-implied path tends to drift back. The effect on any single day is tiny, but it accumulates over one to two weeks. That's the signal both the OU baseline and the neural models trade. Removing factors also cut the average correlation with the market from about 0.6 to 0.02–0.06.

---

## 7. How fast reversion happens sets the windows

![OU half-life](figures/02_ou_half_life.png)

**What it shows.** OU fits on 60-day cumulative residuals revert in most cases, with a median half-life of 8.6 trading days. 83% of reverting fits are faster than 21 days.

**Design decision.** A 60-day window is long enough to estimate a reversion that takes 1–2 weeks. The baseline only trades half-lives under about 21 days (κ > 252/30). The neural models read the last 30 days, a few half-lives of history.

---

## 8. What the OU rules actually do

![s-score example](figures/02_sscore_example.png)

**What it shows.** Starbucks' s-score in 2021 and the baseline's positions. It goes long the residual when the s-score falls below −1.25 and exits above −0.5. It goes short above +1.25 and exits below +0.75. That's 24 position changes in one year for one stock.

**Design decision.** This is a transparent, rule-based benchmark that the neural models must beat. It also shows how often the strategy trades, which leads to the next point.

---

## 9. Trading costs come from two sources

![Universe churn](figures/02_universe_churn.png)

**What it shows.** About 9% of the universe is replaced at each monthly refit, and positions in names that leave are closed. On top of that, the signal itself needs frequent rebalancing.

**Design decision.** Turnover is computed in ticker space, so a stock that changes universe slot isn't counted as a trade. Costs are charged on every unit traded, and the neural models are trained on returns **after** costs.

---

## 10. The result that defines Phase 2

![OU baseline equity](figures/ou_equity_test.png)

| OU baseline, 2020–2023 | Sharpe | Annual return | Turnover per day |
|---|---|---|---|
| Before costs | 0.70 | 3.5% | 34% |
| After 5 bps | −0.17 | −0.8% | 34% |
| Before costs, 1-day delay | 0.23 | 1.1% | 34% |

**What it shows.** The mean-reversion signal makes money before costs. At 5 bps per unit traded, trading 34% of the book every day costs more than the strategy earns. The edge also fades quickly: most of it is gone if trades are delayed by one day.

**What comes next.** Phase 2 trains MLP and Transformer models whose loss is the portfolio's **Sharpe ratio after costs**, using the same portfolio construction and backtester. The question is whether a model optimized this way can keep the signal while trading less.
