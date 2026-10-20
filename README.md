# Sharpee: cost-aware deep learning for statistical arbitrage

A research framework that compares a classical mean-reversion strategy with neural trading models and non-neural machine-learning baselines on US equities. The neural models are trained to maximize the **portfolio's Sharpe ratio after transaction costs**, not to forecast prices; the ridge and gradient-boosting baselines forecast, as standard machine learning would. Everything is evaluated walk-forward, with no look-ahead.

> **Status: research complete (Phases 1-3), dashboard and report built (Phase 4a).** Data pipeline, factor residuals, OU baseline, backtester, three neural models trained on Sharpe after costs, a pre-registered test on the sealed 2020–2023 years, a one-shot 2024–25 holdout, ridge and gradient-boosting baselines, a 400-stock robustness check, an interactive dashboard and a [research report](reports/research_report.md). Live paper trading with GitHub Actions and MLflow tracking come next (Phase 4b).

**Built with:** Python, PyTorch, scikit-learn, NumPy, pandas, SciPy, Streamlit, Plotly, pytest, Jupyter.

**Read more:** [research report](reports/research_report.md) · [findings, figure by figure](reports/README.md) · [interactive dashboard](#interactive-dashboard)

## At a glance

| | Result |
|---|---|
| **Universe** | The 150 most liquid point-in-time S&P 500 stocks each day, 2010–2025 (primary); 400 as a robustness check |
| **Classic baseline (OU)**, test years 2020–2023 | Sharpe 0.94 before costs, 0.15 after 5 bps costs |
| **Best neural model (Transformer)**, validation 2018–2019 | Net Sharpe +0.21 vs. OU's −0.35, with a third less trading |
| **Pre-registered test, 2020–2023** | No reliable edge after costs: Transformer +0.17 vs. OU +0.15 net Sharpe, not statistically distinguishable |
| **One-shot holdout, 2024–2025** | Every strategy lost as residual reversion broke down; at 150 stocks the learned models lost far less (Transformer −0.55 vs. OU −1.65), at 400 they did not |
| **What held up** | Training on costs beat the same model without them in every seed; the learned models degrade about half as fast as OU with higher costs or a one-day delay |
| **Versus non-neural ML** | Gradient boosting and ridge, which forecast then trade, find as much raw signal but lose it to turnover; the Transformer beats gradient boosting on both periods |
| **Robustness: 400 stocks** | Same verdict, no reliable edge after costs. The networks did better (Transformer +0.39, MLP +0.40 vs OU −0.04 net on 2020–2023), but not significantly, and lost about as much as OU in 2024–2025 |
| **Market exposure** | Beta ≈ 0 for every strategy |
| **Correctness** | 71 tests, a look-ahead test and a synthetic sanity check; 3 bugs caught and fixed |

## How it works

```mermaid
flowchart TD
    A["S&P 500 membership + Yahoo prices"] --> B["Daily eligibility: 150 most liquid"]
    B --> C["Rolling PCA: residual returns"]
    C --> D1["OU: fixed s-score rules"]
    C --> D2["MLP / temporal CNN / Transformer: trained on Sharpe after costs"]
    C --> D3["Ridge / gradient boosting: forecast next-day payoff"]
    D1 --> E["Portfolio layer: cap each bet, then hedge (x = Φᵀw)"]
    D2 --> E
    D3 --> E
    E --> F["Backtest: next-day returns minus costs"]
```

1. **Remove what the market and sectors did.** Each stock's daily return is split into a part explained by 5 PCA factors and a stock-specific **residual**.
2. **Bet that residuals snap back.** Three kinds of strategy read each stock's last 30 days of residual moves:
   - **OU** (classic baseline): fixed Avellaneda–Lee rules.
   - **Neural networks** (MLP, temporal CNN, Transformer): learn their own rule, trained directly on the portfolio's Sharpe ratio after costs.
   - **Non-neural ML** (ridge, gradient boosting): standard forecast-then-trade, predicting each stock's next-day residual payoff and trading the forecasts.
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

## Phase 3: the sealed test years, 2020–2023

The rules were committed in [`docs/phase3_protocol.md`](docs/phase3_protocol.md) **before** any test-year model result existed. Each neural model was retrained every year (walk-forward) with frozen settings and 3 seeds, and the Transformer was named the primary candidate in advance.

| Pre-registered rule for the Transformer | Result | Needed | Outcome |
|---|---|---|---|
| 1. Edge after costs (Probabilistic Sharpe Ratio) | 0.64 | ≥ 0.95 | fail |
| 2. Survives 15 configurations tried (Deflated Sharpe Ratio) | 0.24 | ≥ 0.95 | fail |
| 3. Beats OU (paired bootstrap Sharpe difference) | +0.03, 95% CI −0.84 to +0.98 | interval excludes 0 | fail |

**Verdict: no reliable edge after costs.**

| Net Sharpe, 2020–2023 | 0 bps | 5 bps | 10 bps | 20 bps | 5 bps, one day late | Turnover per day |
|---|---|---|---|---|---|---|
| OU | **0.94** | 0.15 | −0.65 | −2.24 | −0.44 | 23.8% |
| MLP | 0.55 | 0.16 | −0.23 | −1.01 | −0.02 | 19.0% |
| Temporal CNN | 0.34 | −0.12 | −0.58 | −1.50 | −0.25 | 20.8% |
| **Transformer** | 0.51 | **0.17** | **−0.16** | **−0.83** | **−0.03** | **15.6%** |

![Cost and delay](reports/figures/04_cost_and_delay.png)

- **No winner at 5 bps.** The Transformer earns +0.17 against OU's +0.15, and with only four years of data the intervals are about 1.8 Sharpe points wide.
- **The learned models are far more robust.** OU has the strongest raw signal but needs cheap, immediate execution. The Transformer trades a third less, so it degrades about half as fast as costs rise and barely notices a one-day delay.
- **Costs in the loss work.** Retraining the same Transformer without costs in its loss made it worse in every seed (combined +0.17 vs −0.08) and raised turnover from 15.6% to 23.7%.
- **The learned rule holds out of sample:** revert moderate residual moves, follow extreme ones.
- **One seed broke in one year.** Seed 0's 2023 model never trained past its starting point and lost heavily; the protocol keeps it in.

### The one-shot holdout, 2024–2025

Run once, after all of the above was committed, with the same frozen settings:

| Net Sharpe, 5 bps | OU | MLP | Temporal CNN | Transformer |
|---|---|---|---|---|
| 2024–2025 | −1.65 | −0.48 | −0.78 | **−0.55** |

Every original strategy lost, even before costs: residual mean reversion broke down in 2024, when the first factor's share of variance hit multi-year lows and stocks kept trending on their own stories. The learned models lost far less than OU, helped by trading less and by following extreme moves instead of fading them, though this advantage did not replicate on 400 stocks (see the robustness check). The verdict is again **no reliable edge after costs**.

### Amendment: non-neural baselines

Matching OU doesn't show how the networks compare with standard machine learning on the same inputs. Two baselines were added afterwards, with their own rules committed in [`docs/phase3_amendment_baselines.md`](docs/phase3_amendment_baselines.md) before they saw the test years. They use the same inputs, portfolio layer, costs and folds:

| Net of 5 bps | Training | Gross Sharpe 2020–23 | Net 2020–23 | Net 2024–25 | Turnover per day |
|---|---|---|---|---|---|
| Ridge | Forecast next-day payoff, then trade | 0.70 | −0.32 | −0.81 | 52.6% |
| Gradient boosting | Forecast next-day payoff, then trade | 0.84 | −1.77 | −2.17 | 100.5% |
| **Transformer** | Sharpe after costs | 0.51 | **+0.17** | **−0.55** | **15.6%** |

- **Beats gradient boosting (rule 4): yes**, +1.94 on 2020–23 (95% CI +0.82 to +3.78) and +1.62 on the holdout. But gradient boosting finds *more* raw signal; it loses by rebuilding the whole book every day. This is a win for cost-aware training over forecast-then-trade.
- **Versus ridge:** +0.50 (95% CI −0.32 to +1.56), not distinguishable.
- **Conclusion: training on after-cost Sharpe beats forecast-then-trade.** Even ridge and gradient boosting learned the same "revert small, follow big" rule, so the rule is in the data. (A linear control was later removed from the amendment; see its deviation note.)

Full details: [findings write-up](reports/README.md#21-amendment-non-neural-baselines-the-objective-beats-forecast-then-trade), [Phase 3 sections](reports/README.md#phase-3-the-sealed-test-years-2020-2023) and [notebook 04](notebooks/04_results.ipynb).

### Robustness check: a 400-stock universe

Do the conclusions survive a wider, less liquid universe? The check was pre-registered in [`docs/robustness_universe_400.md`](docs/robustness_universe_400.md) before any 400-stock result existed. It uses the 400 most liquid stocks each day, chosen point-in-time exactly as before, with delisted names kept. Every hyperparameter stays frozen from the 150-stock tuning, and the networks are retrained with the same folds and seeds. The 150-stock results above remain the primary findings.

| Net Sharpe | 150 stocks, 5 bps | **400 stocks, 5 bps** | 150 stocks, 10 bps | **400 stocks, 10 bps** | 400 stocks, 2024–25, 5 bps |
|---|---|---|---|---|---|
| OU | +0.15 | **−0.04** | −0.65 | **−0.90** | −0.79 |
| Ridge | −0.32 | **−0.64** | −1.34 | **−2.25** | −1.81 |
| Gradient boosting | −1.77 | **−1.14** | −4.37 | **−3.33** | −2.78 |
| MLP | +0.16 | **+0.40** | −0.23 | **+0.07** | −0.95 |
| Temporal CNN | −0.12 | **+0.20** | −0.58 | **−0.17** | −1.44 |
| **Transformer** | +0.17 | **+0.39** | −0.16 | **+0.09** | −0.83 |

2020–2023 unless marked. Neural models are 3 seeds combined.

- **The verdict holds: no reliable edge after costs.** The Transformer's PSR is 0.79, its DSR is 0.40, and its edge over OU is +0.43 with a 95% interval of −0.31 to +1.31. All three rules fail again, as pre-registered.
- **The networks gained from the wider universe and the others did not.** The extra names raise gross Sharpe for every learned model (Transformer 0.51 → 0.70, gradient boosting 0.84 → 1.05), while OU's falls (0.94 → 0.82). The networks trade only 17–20% of the book per day and keep most of the gain. OU (31%), ridge (86%) and gradient boosting (107%) trade more, and costs take more than the extra signal adds.
- **Beats gradient boosting (rule 4): yes again.** The margin is +1.53 on 2020–2023 (95% CI +0.77 to +3.08) and +1.95 on the holdout. Against ridge, the Transformer now also wins on 2020–2023: +1.03 (95% CI +0.37 to +2.05).
- **The holdout is weaker for the networks.** In 2024–25 they lost about as much as OU (Transformer −0.83 vs OU −0.79), unlike the clear advantage at 150 stocks. OU and the forecasting baselines still had a positive gross signal there, while the networks did not.
- **Costs are optimistic.** A flat 5 bps understates trading costs for stocks ranked 150–400 by liquidity, so the 10 bps column is the more realistic one. At 10 bps, only the MLP and Transformer stay above zero, and only just.

The interactive dashboard shows this 400-stock run. Results are in [`reports/n400/phase3/`](reports/n400/phase3/).

## Interactive dashboard

A dark-themed Streamlit app with a top navigation bar lets anyone explore the results without running the research. It shows the **400-stock run**; the primary 150-stock results are the tables above. A control strip under the bar switches between the 2020–2023 test years and the 2024–2025 holdout and sets the trading cost for every page:

- **Overview:** the pre-registered verdict, headline numbers and confidence intervals.
- **Strategies:** equity, drawdown, Sharpe vs cost and by year, and a one-day execution-delay toggle. The **cost slider** recomputes every number exactly (costs are linear in traded notional).
- **Risk:** rolling beta, net exposure and turnover, the ongoing check that every book stays market-neutral.
- **Signal:** pick any stock to see its s-score with OU's trades and each network's scores, plus the learned rule and Transformer attention.
- **Integrity:** the protocol, every configuration tried, and the bugs the checks caught.

It reads only small result files in `reports/` (no models or price data), so it loads instantly:

```bash
pip install -r app/requirements.txt
streamlit run app/dashboard.py
```

To host it for free, deploy the repository on [Streamlit Community Cloud](https://share.streamlit.io) with `app/dashboard.py` as the entry point; it installs the light `app/requirements.txt`.

## How correctness is verified

- **Look-ahead test:** scramble every price after a cutoff date, rebuild the whole pipeline, and assert that every residual, signal, weight and model score on or before the cutoff is unchanged.
- **Synthetic sanity check:** with planted mean reversion, OU and all three networks must win (net Sharpe +19 to +22). With pure random-walk residuals, none may earn anything (net Sharpe at or below zero). Networks are scored on a held-out segment, so a lucky validation period can't pass.
- **71 unit tests:** hand-worked P&L, cost and timing examples; the residual identity `wᵀ(ΦR) = (Φᵀw)ᵀR`; eligibility rules; portfolio constraints, including that the capped book stays factor-neutral; the loss and its gradients; metrics; PSR, DSR and bootstrap intervals against hand-computed values; and the dashboard's numbers against the official Phase 3 results.

### What the checks caught

1. **NaN training blocks.** Random cut points occasionally produced a 1-day training block, whose Sharpe is undefined, and the NaN silently poisoned the model weights. The synthetic check flagged it, because every network scored exactly 0. Blocks are now at least 32 days, and a broken return series reports NaN instead of 0.
2. **A cap that broke the hedge.** The 2% cap was first applied to stock weights *after* hedging, which cut a stock's leg but kept its hedge legs. The MLP learned to exploit this: its portfolio became 80–90% correlated with the S&P 500, and its apparent edge was really riding the 2010–2019 bull market. The cap now applies to residual bets before hedging. A regression test fails under the old order and passes under the new one.
3. **A collapsed model.** Removing score centering let a constant output act as an equal-weight bet on every residual at once, and the networks collapsed onto it (every Transformer configuration scored the same Sharpe). Model scores are centered again, so a constant output now holds no position.

Runs affected by these bugs were excluded from the trial log, and each fix has a regression test.

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
python scripts/walk_forward.py --model transformer        # yearly retraining on 2020-2023 (also mlp, temporal_cnn; --no-cost for the ablation)
python scripts/train_tabular.py --model gbm --tune --walk-forward   # ridge and gradient-boosting baselines
python scripts/run_backtest.py                           # test-year results, significance tests and the verdict
python scripts/export_dashboard.py                       # small result files for the dashboard
streamlit run app/dashboard.py
pytest
```

For the 400-stock robustness run, set `SHARPEE_VARIANT=n400` before `walk_forward.py`, `train_tabular.py --walk-forward`, `run_backtest.py` and `export_dashboard.py`. They then read and write `runs/n400/` and `reports/n400/`, using the frozen 150-stock settings copied into `runs/n400/`.

The notebooks run top to bottom with the project's environment, e.g. `python -m nbconvert --to notebook --execute --inplace notebooks/04_results.ipynb`.

## Repository layout

```
app/              Streamlit dashboard and its data layer
configs/          data, experiment, OU baseline and model settings (YAML); variants/ for robustness runs
docs/             full project plan (v2), the pre-registered Phase 3 protocol, its baseline amendment and the 400-stock robustness check
src/sharpee/
  data/           universe, download, cleaning + eligibility, synthetic markets
  features/       returns, rolling PCA residuals, diagnostics
  strategies/     OU / s-score baseline
  portfolio/      shared weight construction and exposure diagnostics
  backtest/       accounting and costs, backtest engine
  models/         MLP, temporal CNN, Transformer (+ vendored encoder blocks)
  training/       day-block dataset, Sharpe-after-costs loss, training loop, trial log, tuning, ridge and gradient boosting
  evaluation/     metrics, walk-forward folds, significance tests (PSR, DSR, bootstrap), strategy books
notebooks/        01 data exploration, 02 residual analysis, 03 model training, 04 results (outputs saved, readable on GitHub)
scripts/          download_data, run_baseline, train_model, train_tabular, walk_forward, run_backtest, export_dashboard, sanity_synthetic
tests/            71 tests, including the look-ahead test
reports/          research report, findings write-up, figures and Phase 3 results; n400/ holds the 400-stock results and the dashboard data
```

## Roadmap

1. ~~**Data and baseline:** universe, residuals, OU, backtester, tests~~ ✅
2. ~~**Neural models:** MLP, temporal CNN and a compact Transformer, trained end-to-end on Sharpe after costs and tuned on Fold 1 only, with every configuration logged~~ ✅
3. ~~**Evaluation:** a pre-registered protocol, yearly walk-forward retraining on 2020–2023, significance tests, cost and delay sensitivity, the costs-in-loss ablation, and the one-shot 2024–25 holdout~~ ✅ Plus a pre-registered amendment adding ridge and gradient-boosting baselines ✅
4. **Delivery:**
   - ~~**4a:** interactive Streamlit dashboard and research report~~ ✅
   - **4b:** live paper-trading tracker (a daily GitHub Actions job runs the frozen models after the close and records an out-of-sample track record) and MLflow experiment tracking.

## Future work

**More realistic costs.** This comes first, because a wider universe adds less liquid names.

- **Per-stock spreads:** free data has no bid-ask quotes, so estimate each stock's spread from daily highs and lows (Corwin–Schultz) or from liquidity tiers by dollar volume, using trailing data only.
- **Market impact:** add a square-root term, cost ≈ k · volatility · √(trade size / average daily volume), for a few assumed book sizes such as $10M and $100M. For liquid names the spread should dominate, which is itself worth showing.
- **Borrow fees:** the hedge legs are real shorts, so charge a flat annual rate on short notional, plus a harsher stress case.
- **Execution timing:** trade at the next day's open instead of the close, using OHLC data.
- **Training vs evaluation:** train on the per-stock spread cost, which is still linear and differentiable. Apply impact and borrow fees at evaluation only, at first.
- **Dashboard:** the exact cost slider relies on costs being linear, so it would cover the spread component only.

**More statistical power.**

- **Power analysis:** state the smallest Sharpe difference this design can detect. The current difference intervals are about 1.8 Sharpe points wide; shrinking them to about ±0.3 would take roughly ten times more data, so extra years alone cannot fix it.
- **Breadth:** more independent bets raise the signal-to-noise ratio. The 400-stock run is a first step, but it needs the cost work above to be trusted.
- **More seeds:** go from 3 to 10–20 seeds per model and report the spread. This reduces training noise, though not the noise from a short test period.
- **More history:** extend the sample backward where membership and price data allow, and add 2026 as it completes.

## Limitations

- **Survivorship bias remains.** Yahoo has no prices for 250 of the 884 historical members, mostly delisted or acquired companies. Coverage rises from 67% of index members in 2008 to 98% in 2025, so results are likely biased upward.
- **Some tickers are reused.** A few delisted symbols (e.g. `CPWR`) now belong to unrelated securities on Yahoo. The price and liquidity filters keep all of them out of the tradable universe; [notebook 01](notebooks/01_data_exploration.ipynb) shows the check.
- **Simplified execution.** Trades happen at the closing price, costs are a flat proportional rate with no market impact, short positions pay no borrow fee, and weights don't drift between daily rebalances. A flat 5 bps is especially optimistic for the less liquid names in the 400-stock run.
- **Simplified relative to Avellaneda–Lee.** PCA factors are refit monthly rather than daily, and the s-score has no drift adjustment.
- **Short samples.** Phase 2 compares strategies on two validation years and Phase 3 on four test years. At this length, Sharpe ratios within about one point of each other can't be told apart statistically.
- **Few seeds.** Each network is trained with 3 seeds, so one failed seed in one year moves the combined result noticeably.

## References

- Avellaneda & Lee (2010), *Statistical Arbitrage in the U.S. Equities Market*
- Guijarro-Ordonez, Pelger & Zanotti (2021), *Deep Learning Statistical Arbitrage* ([arXiv:2106.04028](https://arxiv.org/abs/2106.04028))
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*

*Research and educational project; not investment advice.*
