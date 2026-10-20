# Realistic costs: per-stock spread tiers and borrow fees (pre-registered)

Committed **before** any result under these costs existed. The flat 5 bps results, their protocol and
their verdicts remain the primary findings; this check asks how they change under more realistic trading
costs. It is an evaluation-only change: no model is retrained or retuned, and every strategy is rescored
from the scores already saved for the test years and the holdout.

## Spread tiers

Each day, every S&P 500 member is ranked by its trailing 20-day median dollar volume. This is the same
point-in-time quantity the universe filter uses, computed from data up to and including that day. The
cost per unit of traded notional depends on the rank:

| Liquidity rank among index members | Cost per unit traded |
|---|---|
| 1–100 | 3 bps |
| 101–200 | 5 bps |
| 201–300 | 7 bps |
| 301 and below, or no dollar volume | 10 bps |

A trade's cost is `Σ cᵢ |Δxᵢ|`, so costs stay linear in traded notional. The ranking is the same for both
universes, so a stock costs the same in the 150-stock and the 400-stock run. The tiers are assumptions,
roughly in line with published effective half-spreads plus fees for large and mid-cap US stocks, not
estimates from quote data. That is why a stress case is also reported.

## Borrow fees

Every short position pays a borrow fee on its notional, charged daily at `rate / 252 × Σ max(−xᵢ, 0)`. This
covers both the stocks shorted as bets and the short legs of the factor hedges, netted per stock as the
book actually holds them. The base rate is 0.5% a year, typical for easy-to-borrow S&P 500 names; no
stock is treated as hard to borrow.

## Scenarios

| Scenario | Spread | Borrow |
|---|---|---|
| Flat (primary, unchanged) | 5 bps on every stock | none |
| **Realistic** | tiers above | 0.5% a year |
| Stress | tiers × 2 | 2% a year |

## What is reported

For OU, ridge, gradient boosting, MLP, temporal CNN and Transformer, on 2020–2023 and 2024–2025, for both
the 150-stock and the 400-stock universe:

- net Sharpe and how much each cost component takes, with and without the one-day delay;
- the same pre-registered rules (PSR, DSR with the original trial counts, the paired comparison with OU,
  and amendment rule 4 against gradient boosting), with the realistic scenario as the headline and the
  stress scenario alongside.

Any change in a verdict is reported next to the flat-cost result and does not replace it.
