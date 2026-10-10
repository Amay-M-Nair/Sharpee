# Sharpee: cost-aware deep learning for statistical arbitrage

A research framework that compares a classical mean-reversion strategy with neural trading models on US equities. The neural models are trained to maximize the **portfolio's Sharpe ratio after transaction costs**, not to forecast prices. Everything is evaluated walk-forward, with no look-ahead.

> **Status: Phases 1 and 2 of 4 complete.** Data pipeline, factor residuals, OU baseline, backtester, and three neural models trained on Sharpe after costs and compared on validation data. Phase 3 (the sealed 2020–2023 test years and significance tests) is next (see [Roadmap](#roadmap)).

**Built with:** Python, PyTorch, NumPy, pandas, SciPy, pytest, Jupyter.

## At a glance

| | Result so far |
|---|---|
| **Universe** | The 150 most liquid point-in-time S&P 500 stocks each day, 2010–2025 |
| **Classic baseline (OU)**, test years 2020–2023 | Sharpe 0.94 before costs, 0.15 after 5 bps costs |
| **Best neural model (Transformer)**, validation 2018–2019 | Net Sharpe +0.21 vs. OU's −0.35, with a third less trading |
| **Market exposure** | Beta ≈ 0 for every strategy |
| **Correctness** | 53 tests, a look-ahead test and a synthetic sanity check; 3 bugs caught and fixed |

## How it works

```mermaid
flowchart LR
    A["S&P 500 membership + Yahoo prices"] --> B["Daily eligibility: 150 most liquid"]
    B --> C["Rolling PCA: residual returns"]
    C --> D1["OU s-score rules"]
    C --> D2["MLP / temporal CNN / Transformer"]
    D1 --> E["Portfolio layer: cap each bet, then hedge (x = Φᵀw)"]
    D2 --> E
    E --> F["Backtest: next-day returns minus costs"]
    F -. trains on Sharpe after costs .-> D2
```

1. **Remove what the market and sectors did.** Each stock's daily return is split into a part explained by 5 PCA factors and a stock-specific **residual**.
2. **Bet that residuals snap back.** The OU baseline uses fixed rules. The neural networks learn their own rule from each stock's last 30 days of residual moves.
3. **Hold hedged positions.** Every bet is long the stock and short its factor exposure, so the portfolio stays market-neutral.
4. **Charge realistic costs.** Weights set at the close earn the next day's return minus 5 bps per unit traded. The networks are trained on exactly this after-cost result.

## Phase 1: data and baseline

| Component | What it does |
|---|---|
| **Point-in-time universe** | Uses historical S&P 500 membership (884 tickers, 2008–2025), not today's constituents. Eligibility is decided day by day: index member, enough history, price > $5, top 150 by dollar volume. |
| **Rolling PCA residuals** | Monthly refits remove 5 market/sector factors. The residual is `ε = Φ R` with `Φ = I − B Qᵀ`, estimated only from data before each decision date. |
| **OU baseline** | Avellaneda–Lee (2010): fits an AR(1) to 60-day cumulative residuals, trades on the s-score, and only trades stocks with a short mean-reversion half-life. |
| **Factor-hedged portfolio** | One shared layer maps residual positions to stock weights `x = Φᵀw` (each position is long the stock, short its factor hedge). Each residual bet is capped at 2% *before* hedging, so the cap can't break the hedge, and gross exposure is at most 1. Every strategy uses it. |
| **Cost-aware backtester** | Weights set at close *t* earn the *t → t+1* return. Costs are charged per unit of traded notional (0–20 bps), and a one-day execution-delay check is built in. |
| **Walk-forward folds** | Expanding training window, 2-year validation, yearly test, 30-day embargoes. 2024–25 is a holdout that gets run once, at the end. |

### Phase 1 results: OU baseline on the test years 2020–2023

OU has no trained parameters (it uses the published Avellaneda–Lee thresholds), so it can be reported on the test years already. The neural models are only evaluated there in Phase 3.

Before trading, the residuals pass these checks:

| Check | Raw returns | Residuals |
|---|---|---|
| Mean \|correlation\| with the market | 0.61 | **0.06** |
| Share of raw return variance remaining | 100% | 43% |
| Median OU half-life | — | 8.5 trading days |

Strategy performance:

| Cost per unit traded | Ann. return | Ann. vol | Sharpe | Max drawdown |
|---|---|---|---|---|
| 0 bps | 3.6% | 3.8% | **0.94** | −4.9% |
| 5 bps | 0.6% | 3.8% | **0.15** | −8.0% |
| 10 bps | −2.4% | 3.8% | −0.65 | −14.4% |
| 0 bps, 1-day delay | 1.3% | 3.7% | 0.36 | −6.2% |

![OU baseline equity](reports/figures/ou_equity_test.png)

- **The mean-reversion signal is real but small.** The strategy trades about 24% of the book per day, so 5 bps costs take most of its gross return and 10 bps take all of it. This matches the documented decline of classical stat arb.
- **The edge decays within a day.** Most of it disappears with a one-day execution delay.
- **Market exposure is low:** beta is 0.05.
- **This set the question for Phase 2:** can a model keep the edge while trading less?

## Phase 2: neural models

| Component | What it does |
|---|---|
| **Three scoring models** | An MLP, a temporal CNN (dilated 1-D convolutions) and a compact Transformer whose encoder blocks are reused from my transformer-from-scratch project. Each reads a stock's last 30 days of cumulative residual and outputs one score. |
| **Sharpe-after-costs training** | A training example is a block of 64 consecutive days × all 150 stocks. Scores go through the same portfolio layer and cost model as OU, and the loss is the negative Sharpe ratio of the portfolio's daily returns after 5 bps costs. Gradients flow through `Φᵀw`, the cap and the costs, so trading too much is penalized directly. |
| **Fold 1 tuning only** | A small grid per model (15 configurations in total, including OU's threshold), trained on 2010–2017 and validated on 2018–2019 with early stopping. Every configuration tried is logged; the Deflated Sharpe Ratio uses that count in Phase 3. The 2020–2023 test years stay sealed. |
| **Seed check** | Each model's best configuration is retrained with 3 random seeds, so a lucky initialization can't pass for a better model. |

### Phase 2 results: validation 2018–2019, 3 seeds per model

| Strategy | Gross Sharpe | Net Sharpe, 5 bps | Turnover per day | Beta |
|---|---|---|---|---|
| OU baseline | 0.81 | −0.35 | 23.7% | 0.00 |
| MLP | 0.76 | +0.05 ± 0.05 | 18.1% | 0.00 |
| Temporal CNN | 0.62 | −0.06 ± 0.15 | 17.4% | −0.01 |
| **Transformer** | **0.82** | **+0.21 ± 0.02** | **15.5%** | 0.00 |

![Validation comparison](reports/figures/03_validation_comparison.png)

- **Same signal, less trading.** The Transformer matches OU's gross Sharpe with about a third less turnover. That turns OU's loss after costs into a profit, consistently across seeds, and breaks even at about 7 bps against OU's 3.5.
- **The networks learned a rule OU can't express.** All three independently learned to bet on reversion for moderate residual moves and on *continuation* for extreme ones, which are often news-driven ([details](reports/README.md#14-what-the-networks-learned-revert-small-moves-follow-big-ones)).
- **Not yet evidence.** These are validation numbers that were also used to choose configurations, and one Sharpe over 473 days has a standard error of about 0.7. Phase 3's sealed test years and the Deflated Sharpe Ratio decide.

**Why everything is built this way:** [reports/README.md](reports/README.md) walks through 14 figures from both phases (data coverage, fat tails, factor structure, residual autocorrelation, half-lives, costs, training stability, turnover, what the networks learned) and the design decision each one supports.

## How correctness is verified

- **Look-ahead test:** scramble every price after a cutoff date, rebuild the whole pipeline, and assert that every residual, signal, weight and model score on or before the cutoff is unchanged.
- **Synthetic sanity check:** with planted mean reversion, OU and all three networks must win (net Sharpe +19 to +22). With pure random-walk residuals, none may earn anything (net Sharpe at or below zero). Networks are scored on a held-out segment, so a lucky validation period can't pass.
- **53 unit tests:** hand-worked P&L, cost and timing examples; the residual identity `wᵀ(ΦR) = (Φᵀw)ᵀR`; eligibility rules; portfolio constraints, including that the capped book stays factor-neutral; the loss and its gradients; metrics.

### What the checks caught

1. **NaN training blocks.** Random cut points occasionally produced a 1-day training block, whose Sharpe is undefined, and the NaN silently poisoned the model weights. The synthetic check flagged it, because every network scored exactly 0. Blocks are now at least 32 days, and a broken return series reports NaN instead of 0.
2. **A cap that broke the hedge.** The 2% cap was first applied to stock weights *after* hedging, which cut a stock's leg but kept its hedge legs. The MLP learned to exploit this: its portfolio became 80–90% correlated with the S&P 500, and its apparent edge was really riding the 2010–2019 bull market. The cap now applies to residual bets before hedging. A regression test fails under the old order and passes under the new one.
3. **A collapsed model.** Removing score centering let a constant output act as an equal-weight bet on every residual at once, and the networks collapsed onto it (every Transformer configuration scored the same Sharpe). Model scores are centered again, so a constant output now holds no position.

Each bad run is archived, not deleted, and excluded from the trial log.

## Run it

Python 3.10+, in a virtual environment:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py      # membership + Yahoo prices -> data/ (about 10 min)
python scripts/run_baseline.py       # OU baseline -> reports/
python scripts/sanity_synthetic.py   # pipeline check on synthetic markets
python scripts/run_baseline.py --tune                     # OU threshold on Fold 1 validation
python scripts/train_model.py --model transformer --tune  # same for mlp and temporal_cnn; --resume skips logged configs
pytest
```

The notebooks run top to bottom with the project's environment, e.g. `python -m nbconvert --to notebook --execute --inplace notebooks/03_train_models.ipynb`.

## Repository layout

```
configs/          data, experiment, OU baseline and model settings (YAML)
docs/             full project plan (v2)
src/sharpee/
  data/           universe, download, cleaning + eligibility, synthetic markets
  features/       returns, rolling PCA residuals, diagnostics
  strategies/     OU / s-score baseline
  portfolio/      shared weight construction and exposure diagnostics
  backtest/       accounting and costs, backtest engine
  models/         MLP, temporal CNN, Transformer (+ vendored encoder blocks)
  training/       day-block dataset, Sharpe-after-costs loss, training loop, trial log, tuning
  evaluation/     metrics, walk-forward folds
notebooks/        01 data exploration, 02 residual analysis, 03 model training (outputs saved, readable on GitHub)
scripts/          download_data, run_baseline, train_model, sanity_synthetic
tests/            53 tests, including the look-ahead test
reports/          results tables, figures and the findings write-up (Phases 1–2)
```

## Roadmap

1. ~~**Data and baseline:** universe, residuals, OU, backtester, tests~~ ✅
2. ~~**Neural models:** MLP, temporal CNN and a compact Transformer, trained end-to-end on Sharpe after costs and tuned on Fold 1 only, with every configuration logged~~ ✅
3. **Evaluation:** a pre-registered protocol committed before any test results; yearly walk-forward retraining on 2020–2023; significance tests (Probabilistic and Deflated Sharpe Ratio, block-bootstrap confidence intervals); cost and delay sensitivity; an ablation that trains without costs in the loss; then the one-shot 2024–25 holdout.
4. **Delivery:** Streamlit dashboard and research report.

## Limitations

- **Survivorship bias remains.** Yahoo has no prices for 250 of the 884 historical members, mostly delisted or acquired companies. Coverage rises from 67% of index members in 2008 to 98% in 2025, so results are likely biased upward.
- **Some tickers are reused.** A few delisted symbols (e.g. `CPWR`) now belong to unrelated securities on Yahoo. The price and liquidity filters keep all of them out of the tradable universe; [notebook 01](notebooks/01_data_exploration.ipynb) shows the check.
- **Simplified execution.** Trades happen at the closing price, costs are a flat proportional rate, and weights don't drift between daily rebalances.
- **Simplified relative to Avellaneda–Lee.** PCA factors are refit monthly rather than daily, and the s-score has no drift adjustment.
- **Small validation sample.** Phase 2 compares strategies on two years of validation data, so differences between them are not yet statistically meaningful.

## References

- Avellaneda & Lee (2010), *Statistical Arbitrage in the U.S. Equities Market*
- Guijarro-Ordonez, Pelger & Zanotti (2021), *Deep Learning Statistical Arbitrage* ([arXiv:2106.04028](https://arxiv.org/abs/2106.04028))
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*

*Research and educational project; not investment advice.*
