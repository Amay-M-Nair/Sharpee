"""Sharpee dashboard: explore the research results.

    streamlit run app/dashboard.py

Reads only the small files in reports/ (see scripts/export_dashboard.py); no
model runs here.
"""

from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import dash_data as dd

REPO = "https://github.com/Amay-M-Nair/Sharpee"
PAGES = ["Overview", "Strategy explorer", "Risk and exposure", "Inside the signal", "Research integrity"]
EULER_GAMMA = 0.5772156649015329

st.set_page_config(page_title="Sharpee", page_icon=":chart_with_upwards_trend:", layout="wide")


@st.cache_data
def returns_for(period):
    return dd.load_returns(period)


@st.cache_data
def signals_for(period):
    return dd.load_signals(period)


@st.cache_data
def significance_for(period):
    return dd.load_significance(period)


@st.cache_data
def trials():
    return dd.load_trials()


def style(df: pd.DataFrame):
    fmt = {"net Sharpe": "{:+.2f}", "gross Sharpe": "{:+.2f}", "ann. return": "{:+.1%}", "ann. vol": "{:.1%}",
           "max drawdown": "{:.1%}", "turnover / day": "{:.1%}", "beta": "{:+.2f}"}
    return df.style.format({k: v for k, v in fmt.items() if k in df.columns})


def lines(series: dict, title: str, yaxis: str = "", hline=None, percent=False, height=360):
    fig = go.Figure()
    for key, s in series.items():
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=dd.NAMES[key], mode="lines",
            line=dict(color=dd.COLORS[key], width=2.5 if key in ("ou", "transformer") else 1.5,
                      dash="dash" if key == "transformer_nocost" else None)))
    if hline is not None:
        fig.add_hline(y=hline, line_width=1, line_color="gray")
    fig.update_layout(title=title, yaxis_title=yaxis, height=height, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.12))
    if percent:
        fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("Sharpee")
    st.caption("Cost-aware deep learning for statistical arbitrage")
    page = st.radio("Page", PAGES, label_visibility="collapsed")
    st.divider()
    period = st.radio("Period", list(dd.PERIODS), format_func=dd.PERIODS.get)
    cost = st.slider("Cost per unit traded (bps)", 0, 30, 5)
    st.caption("Costs are linear in traded notional, so every number is recomputed exactly: "
               "net = gross − cost × turnover.")
    st.divider()
    st.markdown(f"[Code, notebooks and write-up on GitHub]({REPO})")
    st.caption("Research and educational project; not investment advice.")

returns = returns_for(period)
available = dd.strategies_in(returns)
main = [s for s in dd.MAIN if s in available]


# ---------------------------------------------------------------- pages
def overview():
    st.header("Can deep learning beat a classic stat-arb rule after costs?")
    st.markdown(
        "Sharpee removes market and sector moves from S&P 500 stocks with rolling PCA, then bets that what is "
        "left (the residual) snaps back. A classic **Ornstein–Uhlenbeck (OU)** rule is compared with an MLP, a "
        "temporal CNN and a Transformer trained to maximize the **portfolio's Sharpe ratio after trading "
        "costs**. Every portfolio is factor-hedged, and the rules for judging the result were committed "
        "before the test years were opened.")
    sig = significance_for(period)
    st.warning(f"**Pre-registered verdict, {dd.PERIODS[period].lower()}: {dd.verdict_line(period)}.**")

    table = dd.summary(returns, main, cost)
    c = st.columns(4)
    c[0].metric(f"Transformer net Sharpe, {cost} bps", f"{table.loc['Transformer', 'net Sharpe']:+.2f}")
    c[1].metric(f"OU net Sharpe, {cost} bps", f"{table.loc['OU', 'net Sharpe']:+.2f}")
    c[2].metric("PSR (needs 0.95)", f"{sig.loc['transformer', 'psr']:.2f}",
                help="Probability that the Transformer's true Sharpe is above zero, at 5 bps, allowing for "
                     "fat tails and sample length.")
    c[3].metric("DSR (needs 0.95)", f"{sig.loc['transformer', 'dsr']:.2f}",
                help="The same probability with the bar raised for the 15 configurations tried during tuning.")

    table.insert(1, "95% CI at 5 bps",
                 [f"{sig.loc[s, 'ci_low']:+.2f} to {sig.loc[s, 'ci_high']:+.2f}" for s in main])
    st.dataframe(style(table))
    st.caption(f"{dd.PERIODS[period]}, net of {cost} bps, neural models' 3 seeds combined. "
               "Confidence intervals are stationary-bootstrap intervals at 5 bps.")

    st.subheader("What held up")
    st.markdown(
        "- **Costs inside the training loss work.** The same Transformer trained without costs did worse in all "
        "3 seeds on 2020–2023 (+0.17 vs −0.08 net) and traded about 50% more.\n"
        "- **The learned models are more robust.** OU has the strongest raw signal but needs cheap, immediate "
        "execution; move the cost slider, or turn on the one-day delay in the strategy explorer.\n"
        "- **When the signal broke down in 2024–2025**, every strategy lost money, but the learned models lost "
        "about a third as much as OU (switch the period in the sidebar).\n"
        "- **The networks learned a rule OU can't express:** revert moderate residual moves, follow extreme "
        "ones (see *Inside the signal*).")


def explorer():
    picks = st.multiselect("Strategies", available, default=main, format_func=dd.NAMES.get)
    if not picks:
        st.info("Pick at least one strategy.")
        return
    delay = 1 if st.toggle("Trade one day late (execution-delay check)") else 0
    nets = {s: dd.net_returns(returns, s, cost, delay) for s in picks}
    when = " one day late" if delay else ""
    lines({s: dd.equity(r) for s, r in nets.items()}, f"Growth of 1, net of {cost} bps{when}", hline=1)
    lines({s: dd.drawdown(r) for s, r in nets.items()}, "Drawdown", percent=True, height=280)

    left, right = st.columns(2)
    with left:
        curve = dd.sharpe_vs_cost(returns, picks, range(0, 31), delay)
        fig = go.Figure()
        for s in picks:
            fig.add_trace(go.Scatter(x=curve.index, y=curve[dd.NAMES[s]], name=dd.NAMES[s], mode="lines",
                                     line=dict(color=dd.COLORS[s])))
        fig.add_vline(x=cost, line_dash="dot", line_color="gray")
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.update_layout(title="Net Sharpe vs cost", xaxis_title="bps per unit traded", height=340,
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig)
    with right:
        years = dd.sharpe_by_year(returns, picks, cost)
        fig = go.Figure([go.Bar(x=[str(y) for y in years.index], y=years[dd.NAMES[s]], name=dd.NAMES[s],
                                marker_color=dd.COLORS[s]) for s in picks])
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.update_layout(title=f"Net Sharpe by year, {cost} bps", barmode="group", height=340,
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig)
    st.dataframe(style(dd.summary(returns, picks, cost, delay)))


def risk():
    st.markdown("Every book is meant to be close to market-neutral: each bet is long a stock and short its "
                "factor exposure. These charts are the ongoing check.")
    market = returns["market.return"]
    lines({s: dd.rolling_beta(dd.net_returns(returns, s, cost), market) for s in main},
          "Rolling 63-day beta to SPY", "beta", hline=0)
    lines({s: returns[f"{s}.net_exposure"] for s in main}, "Net dollar exposure (share of the book)",
          hline=0, height=300)
    lines({s: returns[f"{s}.turnover_d0"].rolling(21).mean() for s in main},
          "Turnover, 21-day average (share of the book traded per day)", percent=True, height=300)
    st.info("Why this matters: during development, a position cap applied after hedging let a model quietly "
            "go long the market, 80–90% correlated with the S&P 500. The cap now acts before hedging, and a "
            "regression test checks that the capped book stays factor-neutral.")


def signal():
    sigs = signals_for(period)
    counts = sigs["ticker"].value_counts()
    tickers = sorted(counts[counts >= 60].index)
    ticker = st.selectbox("Stock", tickers, index=tickers.index("SBUX") if "SBUX" in tickers else 0)
    days = pd.Index(sorted(sigs["date"].unique()))
    d = sigs[sigs["ticker"] == ticker].set_index("date").reindex(days)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06,
                        subplot_titles=("OU s-score: how stretched the residual is",
                                        "Positions: OU (+1 long / −1 short) and model scores"))
    fig.add_trace(go.Scatter(x=d.index, y=d["s_score"], name="s-score", line=dict(color="#444441", width=1.2)),
                  row=1, col=1)
    for level, dash in [(1.25, "dash"), (-1.25, "dash"), (0.75, "dot"), (-0.5, "dot")]:
        fig.add_hline(y=level, line_dash=dash, line_color="gray", line_width=1, row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ou"], name="OU position", line=dict(color=dd.COLORS["ou"],
                                                                                 shape="hv", width=2)),
                  row=2, col=1)
    for s in ["mlp", "temporal_cnn", "transformer"]:
        if s in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[s], name=f"{dd.NAMES[s]} score",
                                     line=dict(color=dd.COLORS[s], width=1)), row=2, col=1)
    fig.update_yaxes(range=[-4, 4], row=1, col=1)  # a few fits give |s| near 10; zoom out by dragging
    fig.update_layout(height=560, hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig)
    st.caption("Dashed lines: OU enters a long below −1.25 or a short above +1.25. Dotted lines: it exits a "
               "long above −0.5 and a short below +0.75. Model scores are standardized across stocks each "
               "day and averaged over the 3 seeds; above 0 means the model leans long. Gaps are days the "
               "stock was outside the 150-stock universe.")

    left, right = st.columns(2)
    with left:
        st.image(str(dd.FIGURES / "04_score_vs_sscore_test.png"),
                 caption="Learned rule on the test years: revert moderate stretches, follow extreme ones.")
    with right:
        st.image(str(dd.FIGURES / "04_attention.png"),
                 caption="Transformer attention: mostly the last few days, plus a peak about 12 days back.")


def integrity():
    st.subheader("Rules fixed before the test")
    st.markdown(f"The primary candidate, metric and pass/fail rules were written in "
                f"[`docs/phase3_protocol.md`]({REPO}/blob/main/docs/phase3_protocol.md) and pushed to GitHub "
                "before any test-year model result existed, so they couldn't be adjusted after seeing results.")

    st.subheader("Every configuration tried")
    t = trials().copy()
    t["strategy"] = t["model"].map(dd.NAMES)
    fig = go.Figure(go.Bar(x=t["val_sharpe"], y=t["strategy"] + ": " + t["config"], orientation="h",
                           marker_color=t["model"].map(dd.COLORS)))
    fig.update_layout(title="Validation net Sharpe (2018–2019) of all 15 configurations", height=440,
                      margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig)
    daily = t["val_sharpe"] / np.sqrt(252)
    n, z = len(t), NormalDist()
    luck = daily.std() * ((1 - EULER_GAMMA) * z.inv_cdf(1 - 1 / n)
                          + EULER_GAMMA * z.inv_cdf(1 - 1 / (n * np.e))) * np.sqrt(252)
    st.markdown(f"With {n} configurations this spread, the best one would reach an annualized Sharpe of about "
                f"**{luck:.2f} by luck alone**. The Deflated Sharpe Ratio requires a result to clear that bar, "
                "which is why it's stricter than the plain probability of a positive Sharpe.")

    st.subheader("Ablation: costs in the training loss")
    pair = [s for s in ["transformer", "transformer_nocost"] if s in available]
    if len(pair) == 2:
        st.dataframe(style(dd.summary(returns, pair, cost)))
        st.caption("Same architecture, settings, seeds and folds; the only change is whether the 5 bps cost is "
                   "inside the training loss. On 2020–2023 the cost-trained version was better in every seed.")
    else:
        st.caption("The ablation was run on the test years only; switch the period to 2020–2023.")

    st.subheader("What the checks caught")
    with st.expander("NaN training blocks"):
        st.markdown("Random cut points occasionally produced a 1-day training block, whose Sharpe is undefined, "
                    "and the NaN silently poisoned the weights. The synthetic sanity check flagged it because "
                    "every network scored exactly 0.")
    with st.expander("A cap that broke the hedge"):
        st.markdown("The 2% cap was first applied to stock weights after hedging. It cut a stock's leg but kept "
                    "its hedge legs, and a model learned to use this to go long the market. The cap now acts on "
                    "each bet before hedging, with a regression test.")
    with st.expander("Models collapsing to one bet"):
        st.markdown("Without centering, a constant model output was an equal-weight bet on every residual, and "
                    "the networks collapsed onto it. Model scores are centered, so a constant output holds "
                    "nothing.")
    st.markdown("Every bad run is archived and excluded from the trial log. The repository's automated tests "
                "include a look-ahead test that scrambles future prices and checks nothing earlier changes.")


{"Overview": overview, "Strategy explorer": explorer, "Risk and exposure": risk,
 "Inside the signal": signal, "Research integrity": integrity}[page]()
