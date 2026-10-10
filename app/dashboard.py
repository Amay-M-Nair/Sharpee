"""Sharpee dashboard: explore the research results.

    streamlit run app/dashboard.py

Reads only the small files in reports/ (see scripts/export_dashboard.py); no
model runs here. Pages sit in a top navigation bar; the period and cost
controls under it apply to every page.
"""

from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import dash_data as dd

REPO = "https://github.com/Amay-M-Nair/Sharpee"
EULER_GAMMA = 0.5772156649015329

st.set_page_config(page_title="Sharpee", page_icon=":material/monitoring:", layout="wide")


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
def rule_for(period):
    return dd.learned_rule(dd.load_signals(period))


def style(df: pd.DataFrame):
    fmt = {"net Sharpe": "{:+.2f}", "gross Sharpe": "{:+.2f}", "ann. return": "{:+.1%}", "ann. vol": "{:.1%}",
           "max drawdown": "{:.1%}", "turnover / day": "{:.1%}", "beta": "{:+.2f}"}
    return df.style.format({k: v for k, v in fmt.items() if k in df.columns})


def finish(fig: go.Figure, title: str, height: int = 360, yaxis: str = "", percent: bool = False,
           legend_below: bool = False):
    """Shared dark styling. The legend sits under the title, or below the chart when subplot titles need the top."""
    legend = (dict(orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0) if legend_below
              else dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_layout(title=title, yaxis_title=yaxis, height=height, hovermode="x unified",
                      margin=dict(l=10, r=10, t=60 if legend_below else 80, b=10), legend=legend,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="#21262D", zerolinecolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D", zerolinecolor="#21262D")
    if percent:
        fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig)


def lines(series: dict, title: str, yaxis: str = "", hline=None, percent=False, height=360):
    fig = go.Figure()
    for key, s in series.items():
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=dd.NAMES[key], mode="lines",
            line=dict(color=dd.COLORS[key], width=2.6 if key in ("ou", "transformer") else 1.8,
                      dash="dash" if key == "transformer_nocost" else None)))
    if hline is not None:
        fig.add_hline(y=hline, line_width=1, line_color=dd.REFERENCE)
    finish(fig, title, height, yaxis, percent)


# ---------------------------------------------------------------- pages
def overview():
    st.markdown("### Can deep learning beat a classic stat-arb rule after costs?")
    st.markdown(
        "Sharpee removes market and sector moves from S&P 500 stocks with rolling PCA, then bets that what is "
        "left (the residual) snaps back. A classic **Ornstein–Uhlenbeck (OU)** rule is compared with an MLP, a "
        "temporal CNN and a Transformer trained to maximize the **portfolio's Sharpe ratio after trading "
        "costs**. Every portfolio is factor-hedged, and the rules for judging the result were committed "
        "before the test years were opened.")
    sig = significance_for(period)
    st.warning(f"**Pre-registered verdict, {dd.PERIODS[period].lower()}: {dd.verdict_line(period)}.**",
               icon=":material/gavel:")

    table = dd.summary(returns, main, cost)
    c = st.columns(4)
    c[0].metric(f"Transformer net Sharpe, {cost} bps", f"{table.loc['Transformer', 'net Sharpe']:+.2f}",
                border=True)
    c[1].metric(f"OU net Sharpe, {cost} bps", f"{table.loc['OU', 'net Sharpe']:+.2f}", border=True)
    c[2].metric("PSR (needs 0.95)", f"{sig.loc['transformer', 'psr']:.2f}", border=True,
                help="Probability that the Transformer's true Sharpe is above zero, at 5 bps, allowing for "
                     "fat tails and sample length.")
    c[3].metric("DSR (needs 0.95)", f"{sig.loc['transformer', 'dsr']:.2f}", border=True,
                help="The same probability with the bar raised for the 15 configurations tried during tuning.")

    table.insert(1, "95% CI at 5 bps",
                 [f"{sig.loc[s, 'ci_low']:+.2f} to {sig.loc[s, 'ci_high']:+.2f}" for s in main])
    st.dataframe(style(table))
    st.caption(f"{dd.PERIODS[period]}, net of {cost} bps, neural models' 3 seeds combined. "
               "Confidence intervals are stationary-bootstrap intervals at 5 bps.")

    st.markdown("#### What held up")
    st.markdown(
        "- **Costs inside the training loss work.** The same Transformer trained without costs did worse in all "
        "3 seeds on 2020–2023 (+0.17 vs −0.08 net) and traded about 50% more.\n"
        "- **The learned models are more robust.** OU has the strongest raw signal but needs cheap, immediate "
        "execution: move the cost slider, or turn on the one-day delay in *Strategies*.\n"
        "- **When the signal broke down in 2024–2025**, every strategy lost money, but the learned models lost "
        "about a third as much as OU (switch the period above).\n"
        "- **The networks learned a rule OU can't express:** revert moderate residual moves, follow extreme "
        "ones (see *Signal*).")


def explorer():
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    picks = left.multiselect("Strategies", available, default=main, format_func=dd.NAMES.get)
    delay = 1 if right.toggle("Trade one day late") else 0
    if not picks:
        st.info("Pick at least one strategy.")
        return
    nets = {s: dd.net_returns(returns, s, cost, delay) for s in picks}
    when = ", one day late" if delay else ""
    lines({s: dd.equity(r) for s, r in nets.items()}, f"Growth of 1, net of {cost} bps{when}", hline=1)
    lines({s: dd.drawdown(r) for s, r in nets.items()}, "Drawdown", percent=True, height=280)

    left, right = st.columns(2)
    with left:
        curve = dd.sharpe_vs_cost(returns, picks, range(0, 31), delay)
        fig = go.Figure([go.Scatter(x=curve.index, y=curve[dd.NAMES[s]], name=dd.NAMES[s], mode="lines",
                                    line=dict(color=dd.COLORS[s], width=2.4)) for s in picks])
        fig.add_vline(x=cost, line_dash="dot", line_color=dd.REFERENCE)
        fig.add_hline(y=0, line_width=1, line_color=dd.REFERENCE)
        fig.update_xaxes(title="bps per unit traded")
        finish(fig, "Net Sharpe vs cost", 340)
    with right:
        years = dd.sharpe_by_year(returns, picks, cost)
        fig = go.Figure([go.Bar(x=[str(y) for y in years.index], y=years[dd.NAMES[s]], name=dd.NAMES[s],
                                marker_color=dd.COLORS[s]) for s in picks])
        fig.add_hline(y=0, line_width=1, line_color=dd.REFERENCE)
        fig.update_layout(barmode="group")
        finish(fig, f"Net Sharpe by year, {cost} bps", 340)
    st.dataframe(style(dd.summary(returns, picks, cost, delay)))


def risk():
    st.markdown("Every book is meant to be close to market-neutral: each bet is long a stock and short its "
                "factor exposure. These charts are the ongoing check.")
    market = returns["market.return"]
    lines({s: dd.rolling_beta(dd.net_returns(returns, s, cost), market) for s in main},
          "Rolling 63-day beta to SPY", "beta", hline=0)
    left, right = st.columns(2)
    with left:
        lines({s: returns[f"{s}.net_exposure"] for s in main}, "Net dollar exposure (share of the book)",
              hline=0, height=320)
    with right:
        lines({s: returns[f"{s}.turnover_d0"].rolling(21).mean() for s in main},
              "Turnover, 21-day average", percent=True, height=320)
    st.info("Why this matters: during development, a position cap applied after hedging let a model quietly "
            "go long the market, 80–90% correlated with the S&P 500. The cap now acts before hedging, and a "
            "regression test checks that the capped book stays factor-neutral.", icon=":material/shield:")


def signal():
    sigs = signals_for(period)
    counts = sigs["ticker"].value_counts()
    tickers = sorted(counts[counts >= 60].index)
    ticker = st.selectbox("Stock", tickers, index=tickers.index("SBUX") if "SBUX" in tickers else 0)
    days = pd.Index(sorted(sigs["date"].unique()))
    d = sigs[sigs["ticker"] == ticker].set_index("date").reindex(days)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.07,
                        subplot_titles=("OU s-score: how stretched the residual is",
                                        "Positions: OU (+1 long / −1 short) and model scores"))
    fig.add_trace(go.Scatter(x=d.index, y=d["s_score"], name="s-score", line=dict(color="#E6EDF3", width=1.3)),
                  row=1, col=1)
    for level, dash in [(1.25, "dash"), (-1.25, "dash"), (0.75, "dot"), (-0.5, "dot")]:
        fig.add_hline(y=level, line_dash=dash, line_color=dd.REFERENCE, line_width=1, row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ou"], name="OU position",
                             line=dict(color=dd.COLORS["ou"], shape="hv", width=2.4)), row=2, col=1)
    for s in ["mlp", "temporal_cnn", "transformer"]:
        if s in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[s], name=f"{dd.NAMES[s]} score",
                                     line=dict(color=dd.COLORS[s], width=1.4)), row=2, col=1)
    fig.update_yaxes(range=[-4, 4], row=1, col=1)  # a few fits give |s| near 10; drag to zoom out
    finish(fig, f"{ticker}, {dd.PERIODS[period].lower()}", 600, legend_below=True)
    st.caption("Dashed lines: OU enters a long below −1.25 or a short above +1.25. Dotted lines: it exits a "
               "long above −0.5 and a short below +0.75. Model scores are standardized across stocks each "
               "day and averaged over the 3 seeds; above 0 means the model leans long. Gaps are days the "
               "stock was outside the 150-stock universe.")

    left, right = st.columns(2)
    with left:
        rule = rule_for(period)
        fig = go.Figure([go.Scatter(x=rule.index, y=rule[s], name=dd.NAMES[s], mode="lines+markers",
                                    line=dict(color=dd.COLORS[s], width=2.4)) for s in rule.columns])
        fig.add_hline(y=0, line_width=1, line_color=dd.REFERENCE)
        fig.update_xaxes(title="OU s-score bucket")
        fig.update_layout(hovermode="closest")
        finish(fig, "Learned rule: mean model score by stretch", 360, "standardized score")
        st.caption("Below about |s| = 2 the networks bet on reversion (positive when the residual fell). "
                   "Beyond it they follow the move, which OU never does. All stock-days, 3 seeds averaged.")
    with right:
        att = dd.load_attention()
        fig = go.Figure([go.Scatter(x=att.index, y=att[c], name=label, mode="lines+markers",
                                    line=dict(color=color, width=2.4))
                         for c, label, color in [("moderate", "moderate stretch, |s| < 2", "#58A6FF"),
                                                 ("extreme", "extreme stretch, |s| ≥ 2", "#F2CC60")]])
        fig.add_hline(y=1 / len(att), line_dash="dot", line_color=dd.REFERENCE)
        fig.update_xaxes(title="day in the window (0 = decision day)")
        fig.update_layout(hovermode="closest")
        finish(fig, "Transformer attention by day", 360, "attention received")
        st.caption("Last layer, test years, seed 0. Mostly the last few days, plus a peak about 12 days back, "
                   "roughly one reversion half-life. The dotted line is uniform attention.")


def integrity():
    st.markdown("#### Rules fixed before the test")
    st.markdown(f"The primary candidate, metric and pass/fail rules were written in "
                f"[`docs/phase3_protocol.md`]({REPO}/blob/main/docs/phase3_protocol.md) and pushed to GitHub "
                "before any test-year model result existed, so they couldn't be adjusted after seeing results.")

    st.markdown("#### Every configuration tried")
    t = dd.load_trials()
    t["strategy"] = t["model"].map(dd.NAMES)
    fig = go.Figure(go.Bar(x=t["val_sharpe"], y=t["strategy"] + ": " + t["config"], orientation="h",
                           marker_color=t["model"].map(dd.COLORS)))
    fig.add_vline(x=0, line_width=1, line_color=dd.REFERENCE)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    finish(fig, "Validation net Sharpe (2018–2019) of all 15 configurations", 460)
    daily = t["val_sharpe"] / np.sqrt(252)
    n, z = len(t), NormalDist()
    luck = daily.std() * ((1 - EULER_GAMMA) * z.inv_cdf(1 - 1 / n)
                          + EULER_GAMMA * z.inv_cdf(1 - 1 / (n * np.e))) * np.sqrt(252)
    st.markdown(f"With {n} configurations this spread, the best one would reach an annualized Sharpe of about "
                f"**{luck:.2f} by luck alone**. The Deflated Sharpe Ratio requires a result to clear that bar, "
                "which is why it's stricter than the plain probability of a positive Sharpe.")

    st.markdown("#### Ablation: costs in the training loss")
    pair = [s for s in ["transformer", "transformer_nocost"] if s in available]
    if len(pair) == 2:
        st.dataframe(style(dd.summary(returns, pair, cost)))
        st.caption("Same architecture, settings, seeds and folds; the only change is whether the 5 bps cost is "
                   "inside the training loss. On 2020–2023 the cost-trained version was better in every seed.")
    else:
        st.caption("The ablation was run on the test years only; switch the period to 2020–2023.")

    st.markdown("#### What the checks caught")
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


# ---------------------------------------------------------------- layout
nav = st.navigation([
    st.Page(overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(explorer, title="Strategies", icon=":material/show_chart:", url_path="strategies"),
    st.Page(risk, title="Risk", icon=":material/balance:", url_path="risk"),
    st.Page(signal, title="Signal", icon=":material/insights:", url_path="signal"),
    st.Page(integrity, title="Integrity", icon=":material/verified:", url_path="integrity"),
], position="top")

head, period_col, cost_col = st.columns([2.2, 1.6, 2.2], vertical_alignment="center")
head.markdown("## Sharpee")
head.caption("Cost-aware deep learning for statistical arbitrage")
period = period_col.segmented_control("Period", list(dd.PERIODS), default="test", key="period",
                                      format_func=lambda p: "2020–2023 test" if p == "test" else "2024–2025 holdout")
period = period or "test"
cost = cost_col.slider("Cost per unit traded (bps)", 0, 30, 5, key="cost",
                       help="Costs are linear in traded notional, so every number is recomputed exactly: "
                            "net = gross − cost × turnover.")
st.divider()

returns = returns_for(period)
available = dd.strategies_in(returns)
main = [s for s in dd.MAIN if s in available]

nav.run()

st.divider()
st.caption(f"[Code, notebooks, research report and write-up on GitHub]({REPO}) · "
           "Research and educational project; not investment advice.")
