import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from data.data import get_data
from fitting.fitting import fit_gbm, fit_ou
from simulation.simulate import simulate_gbm, simulate_ou, compute_cone

st.set_page_config(page_title="Stochastic Process Visualizer", layout="wide")
st.title("Stochastic Process Visualizer")

with st.sidebar:
    ticker = st.text_input("Ticker", value="SPY")
    period = st.selectbox("Historical Period", ["1y", "2y", "5y"], index=1)
    process = st.selectbox("Process", ["GBM", "Ornstein-Uhlenbeck"])
    n_paths = st.slider("Simulation Paths", 100, 2000, 500, step=100)
    n_steps = st.slider("Forecast Days", 10, 252, 60)
    run = st.button("Run")

if run:
    with st.spinner("Fetching data and fitting model..."):
        prices, returns = get_data(ticker, period)
        S0 = float(prices.iloc[-1])

        if process == "GBM":
            params = fit_gbm(returns)
            paths = simulate_gbm(params, S0, n_paths, n_steps)
            label = f"μ = {params.mu:.2%} | σ = {params.sigma:.2%}"
        else:
            params = fit_ou(prices)
            paths = simulate_ou(params, S0, n_paths, n_steps)
            label = f"θ = {params.theta:.2f} | μ = {params.mu:.2f} | σ = {params.sigma:.2f}"

        cone = compute_cone(paths)
        t_hist = list(range(len(prices)))
        t_fwd = list(range(n_steps + 1))

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Price", f"${S0:.2f}")
    col2.metric("Fitted Params", label)
    col3.metric("Paths Simulated", f"{n_paths:,}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(prices))),
        y=prices.values.flatten(),
        mode="lines",
        name="Historical",
        line=dict(color="#7b7bff", width=1.5)
    ))

    offset = len(prices) - 1
    colors = {10: "rgba(100,100,255,0.15)", 25: "rgba(100,100,255,0.25)", 50: "rgba(100,100,255,0)"}

    fig.add_trace(go.Scatter(
        x=[offset + t for t in t_fwd],
        y=cone[90],
        mode="lines", line=dict(width=0),
        showlegend=False, name="90th"
    ))
    fig.add_trace(go.Scatter(
        x=[offset + t for t in t_fwd],
        y=cone[10],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(100,100,255,0.10)",
        name="80% band"
    ))
    fig.add_trace(go.Scatter(
        x=[offset + t for t in t_fwd],
        y=cone[75],
        mode="lines", line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[offset + t for t in t_fwd],
        y=cone[25],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(100,100,255,0.18)",
        name="50% band"
    ))
    fig.add_trace(go.Scatter(
        x=[offset + t for t in t_fwd],
        y=cone[50],
        mode="lines",
        line=dict(color="#ffffff", width=2, dash="dash"),
        name="Median forecast"
    ))

    fig.add_vline(x=offset, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#050510",
        height=500,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=-0.1),
        xaxis_title="Trading Days",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Return Distribution"):
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=returns.values.flatten(),
            nbinsx=60,
            name="Log Returns",
            marker_color="#7b7bff",
            opacity=0.75
        ))
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="#050510", height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)