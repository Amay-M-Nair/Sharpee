# Sharpee: Does Deep Learning Beat a Classic Statistical-Arbitrage Rule After Costs?

*Research report. Code, data pipeline, notebooks and every result file: [github.com/Amay-M-Nair/Sharpee](https://github.com/Amay-M-Nair/Sharpee). Research and educational project; not investment advice.*

## Abstract

Statistical arbitrage bets that the stock-specific part of a price move, what remains after removing
market and sector moves, tends to reverse. This project compares the classic Avellaneda–Lee
Ornstein–Uhlenbeck (OU) rule with three neural networks (an MLP, a temporal CNN and a Transformer)
trained end-to-end to maximize the **portfolio's Sharpe ratio after transaction costs**, on point-in-time
S&P 500 data from 2010 to 2025. All strategies share one factor-hedged portfolio layer and one cost-aware
backtester. Models were tuned on 2018–2019 only; the rules for judging them were committed publicly before
the 2020–2023 test years were opened.

Under those rules, **no strategy shows a reliable edge after costs**: the Transformer (the pre-registered
candidate) earns a net Sharpe of +0.17 against OU's +0.15, with a 95% interval of −0.72 to +1.08, and its
Deflated Sharpe Ratio is 0.24. Three findings do hold up. Training with costs inside the loss beat an
otherwise identical model trained without them in every seed. The learned strategies trade a third less
than OU and degrade about half as fast as costs rise or execution is delayed. And when residual reversion
broke down in the 2024–2025 holdout, every strategy lost money, but the learned models lost about a third
as much as OU.

## 1. Question

Classic statistical arbitrage still has a measurable signal, but it trades a lot, and trading costs have
eroded its profits since the early 2000s. A model trained to maximize *profit after costs*, rather than to
forecast prices, could in principle keep the signal while trading less. The question is whether that
works, and whether any edge is distinguishable from luck.

## 2. Data

- **Universe.** Historical S&P 500 membership (the public `fja05680/sp500` dataset): 884 tickers were
  members at some point in 2008–2025. Daily adjusted prices and volume come from Yahoo Finance, which
  supplies 634 of them; the missing 250 are mostly delisted or acquired companies. Coverage rises from 67%
  of index members in 2008 to 98% in 2025, so survivorship bias remains and likely flatters results.
- **Eligibility, decided daily** from data up to that day: index member, at least 312 days of history,
  price above $5, and among the 150 most liquid by 20-day median dollar volume.
- **Sample.** 2008–2009 is warm-up only. Development uses 2010–2019, the test years are 2020–2023, and
  2024–2025 is a holdout run once at the end.
- **Data quality.** Some delisted symbols now belong to unrelated securities on Yahoo (for example `CPWR`,
  with +2,100% daily moves). The price and liquidity filters keep all of them out of the universe; the
  largest moves inside the universe are real events such as AIG in 2009 and the March 2020 oil crash.

## 3. Method

**Residuals.** On the first trading day of each month, PCA on the previous 252 days of standardized
returns gives 5 eigenportfolios `Q`. Each stock is regressed on their returns to get loadings `B`, and the
residual is `ε = Φ R` with `Φ = I − B Qᵀ`. Each row of `Φ` is a tradable portfolio: long one stock, short
its factor exposure. On every decision date, the trailing residual history uses the `Φ` in force on that
date, so nothing after it is used. Out of sample, the average correlation of residuals with the market is
0.06, against 0.61 for raw returns; residuals keep 43% of the variance.

**OU baseline.** An AR(1) fit to each stock's 60-day cumulative residual gives a reversion speed κ,
mean μ and equilibrium deviation σ_eq, and the s-score `(x − μ)/σ_eq`. Positions follow Avellaneda–Lee:
open long below −1.25 or short above +1.25, close a long above −0.50 and a short below +0.75, trading only
half-lives under about 21 days. The median half-life is 8.6 days.

**Neural models.** Each reads a stock's last 30 days of cumulative residual, scaled by its own volatility,
and outputs one score: an MLP, a temporal CNN with dilated convolutions, and a Transformer encoder whose
blocks come from the author's transformer-from-scratch project.

**Portfolio layer, shared by every strategy.** Model scores are demeaned across stocks (OU's ±1
positions are used as they are) and normalized to residual weights `w`. Each residual bet is capped at 2%
**before** hedging, then mapped to stock weights `x = Φᵀw`, so the book stays factor-hedged.

**Costs and timing.** Weights set at the close of day *t* earn the return from *t* to *t+1*. Every unit
of traded notional costs 5 bps, charged to the position it creates.

**Training objective.** One example is a block of 64 consecutive days × all 150 stocks. Scores pass
through the portfolio layer and cost model, and the loss is the negative Sharpe ratio of the portfolio's
daily returns after costs. Gradients flow through the hedge, the cap and the costs, so excessive trading
is penalized directly.

**Evaluation design.**
- *Tuning (Fold 1 only):* train 2010–2017, validate 2018–2019, 30-day embargoes. Fifteen configurations
  in total (3 OU thresholds, 4 per network), every one logged.
- *Pre-registration:* `docs/phase3_protocol.md` fixed the primary candidate (Transformer), primary metric
  (net Sharpe at 5 bps, 3 seeds combined with equal capital), and three pass/fail rules before any
  test-year result existed.
- *Walk-forward test:* each network is retrained from scratch for each test year with frozen settings.
- *Significance:* the Probabilistic Sharpe Ratio (PSR), the Deflated Sharpe Ratio (DSR, which raises the
  bar for the 15 configurations tried) and stationary-bootstrap intervals, including a paired interval for
  the Sharpe difference against OU.

## 4. Development: what the checks caught

The pipeline is guarded by a look-ahead test (scramble every price after a cutoff date and assert that
nothing decided earlier changes), 64 unit tests, and a synthetic sanity check (with planted mean
reversion every strategy must win; with pure noise none may). They caught three silent bugs:

1. **NaN training blocks.** Random cut points occasionally produced a 1-day block whose Sharpe is
   undefined, silently corrupting the weights.
2. **A cap that broke the hedge.** Capping stock weights after hedging cut stock legs but kept their hedge
   legs. The MLP learned to exploit this: its book became 80–90% correlated with the S&P 500 and its
   apparent edge was really the 2010s bull market.
3. **Models collapsing to one bet.** Without centering, a constant output was an equal-weight bet on every
   residual, and the networks collapsed onto it.

Each was fixed with a regression test; the affected runs are archived and excluded from the trial log.

## 5. Validation results (2018–2019)

| Strategy | Gross Sharpe | Net Sharpe, 5 bps (3 seeds) | Turnover per day |
|---|---|---|---|
| OU | 0.81 | −0.35 | 23.7% |
| MLP | 0.76 | +0.05 ± 0.05 | 18.1% |
| Temporal CNN | 0.62 | −0.06 ± 0.15 | 17.4% |
| Transformer | 0.82 | +0.21 ± 0.02 | 15.5% |

The Transformer matched OU's gross Sharpe while trading about a third less, which made it the
pre-registered candidate. These numbers were also used for selection, so they were treated as
optimistic.

## 6. Test results (2020–2023)

| Pre-registered rule (Transformer) | Result | Needed | Outcome |
|---|---|---|---|
| 1. Edge after costs, PSR | 0.64 | ≥ 0.95 | fail |
| 2. Survives 15 configurations, DSR | 0.24 | ≥ 0.95 | fail |
| 3. Beats OU, paired Sharpe difference | +0.03 (−0.84 to +0.98) | excludes 0 | fail |

**Verdict: no reliable edge after costs.** With 15 configurations of this spread, the best one would reach
an annualized Sharpe of about 0.52 by luck alone, which is the bar the DSR applies.

| Net Sharpe | 0 bps | 5 bps | 10 bps | 20 bps | 5 bps, one day late | Turnover |
|---|---|---|---|---|---|---|
| OU | **0.94** | 0.15 | −0.65 | −2.24 | −0.44 | 23.8% |
| MLP | 0.55 | 0.16 | −0.23 | −1.01 | −0.02 | 19.0% |
| Temporal CNN | 0.34 | −0.12 | −0.58 | −1.50 | −0.25 | 20.8% |
| Transformer | 0.51 | **0.17** | **−0.16** | **−0.83** | **−0.03** | **15.6%** |

**Robustness.** OU has the strongest raw signal but depends on cheap, immediate execution: each extra
5 bps costs it about 0.8 Sharpe points, and a one-day delay turns it negative. The Transformer gives up
part of the gross signal for a third less trading, so it degrades about half as fast and barely notices
the delay.

**Ablation: costs in the loss.** The same Transformer retrained with zero cost inside its loss, evaluated
at 5 bps:

| Transformer | Seed 0 | Seed 1 | Seed 2 | Combined | Turnover |
|---|---|---|---|---|---|
| Costs in loss | −0.16 | +0.34 | +0.31 | +0.17 | 15.6% |
| No costs in loss | −0.23 | −0.02 | +0.02 | −0.08 | 23.7% |

The cost-trained model was better in every seed and traded less in every seed, and its gross Sharpe was
higher too (0.51 vs 0.42). This is the clearest result of the project, although not a formal significance
test.

**By year.** OU's best year was 2020 (1.23) and its worst 2022 (−1.26); the Transformer's best was 2021
(0.96) and its worst 2023 (−1.52). The 2023 loss comes from one model: seed 0's 2023 network never
improved on its initial state during training (best validation Sharpe −1.43 at epoch 0) and lost heavily
(−3.44), while the other two seeds earned +0.33 and +0.37. The protocol keeps every seed. A rule such as
"don't trade a model whose validation Sharpe is negative" is a sensible safeguard for future work, but
adding it after seeing the test years would be fitting to them.

**Model complexity.** As an exploratory observation, the MLP and Transformer hold nearly the same book:
their daily returns are 0.95 correlated in 2020–2023 and 0.98 in 2024–2025, and their Sharpe difference is
indistinguishable from zero. With a 30-number input from one series, a small MLP captures what the
Transformer does; the Transformer is more stable across seeds and trades less.

## 7. Holdout (2024–2025)

Run once, after all of the above was committed, with the same frozen settings.

| 2024–2025, 5 bps | Gross Sharpe | Net Sharpe (95% CI) | Turnover |
|---|---|---|---|
| OU | −0.77 | −1.65 (−3.01 to −0.33) | 22.8% |
| MLP | −0.03 | −0.48 (−1.74 to +0.72) | 16.3% |
| Temporal CNN | −0.32 | −0.78 (−2.06 to +0.47) | 17.3% |
| Transformer | −0.16 | −0.55 (−1.79 to +0.59) | 13.1% |

Every strategy lost money, even before costs: residual reversion broke down, with 2024 the worst year for
all of them. This is consistent with the factor structure, whose first factor's share of variance fell to
multi-year lows in 2024, as stocks moved on their own stories and kept trending. The learned models lost
far less than OU (paired difference for the Transformer +1.10, 95% CI −0.16 to +2.46), and in 2025 the MLP
and Transformer were slightly positive again while OU was not. The verdict is again "no reliable edge
after costs".

## 8. What the networks learned

Bucketing stock-days by OU s-score shows that all three networks independently learned the same
nonlinear rule: **revert moderate stretches** (|s| below about 2), like OU, but **follow extreme ones**,
most strongly after large up-moves. Large residual moves are often news-driven, and news-driven moves tend
to keep drifting; OU fades them by construction. The pattern holds on the test years for the MLP and
Transformer. The Transformer's attention concentrates on the last three days of its window, with a second
peak about 12 days back, roughly one reversion half-life; attention shows where the model looks, not why.

## 9. Limitations

- Survivorship bias from free data; results are likely flattered, especially early in the sample.
- Simplified execution: closing prices, flat proportional costs, no weight drift between rebalances.
- PCA factors refit monthly rather than daily; no drift adjustment to the s-score.
- Short evaluation samples: four test years and two holdout years cannot separate Sharpe ratios within
  about one point of each other.
- Three seeds per model; one seed's failure in one fold moves the combined result noticeably.

## 10. Conclusion and future work

On this data, learning the trading rule end-to-end does not produce a reliable edge over the textbook rule
after costs, and in 2024–2025 the signal both rely on stopped working. What the learned models do provide
is a cheaper, more robust way to trade the same signal, and that advantage traces directly to putting
costs inside the training objective.

Natural next steps: survivorship-free data (CRSP), next-day-open execution and per-stock cost models;
richer inputs (volume, volatility, earnings dates, news) and attention across stocks rather than across
days; a safeguard against deploying models whose validation Sharpe is negative; a learning-curve study of
how much data the models need; and a live paper-trading record, which is the only test that cannot be
fitted to.

## Reproduce

```bash
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py
python scripts/run_baseline.py --tune
python scripts/train_model.py --model transformer --tune      # also mlp, temporal_cnn
python scripts/walk_forward.py --model transformer            # also mlp, temporal_cnn; --no-cost for the ablation
python scripts/run_backtest.py                                # test years and verdict; --holdout runs 2024-2025
python scripts/export_dashboard.py && streamlit run app/dashboard.py
pytest
```

## References

- M. Avellaneda and J.-H. Lee (2010). Statistical arbitrage in the U.S. equities market. *Quantitative Finance*.
- J. Guijarro-Ordonez, M. Pelger and G. Zanotti (2021). Deep learning statistical arbitrage. arXiv:2106.04028.
- D. Bailey and M. López de Prado (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*.
- D. Politis and J. Romano (1994). The stationary bootstrap. *Journal of the American Statistical Association*.
- E. Gatev, W. Goetzmann and K. G. Rouwenhorst (2006). Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies*.
