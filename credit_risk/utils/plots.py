"""Plotly figure builders — all return JSON-serialisable dicts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

_PALETTE = ["#3b82f6", "#22c55e", "#ef4444", "#f59e0b", "#a855f7", "#06b6d4"]
_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,23,42,0)",
    plot_bgcolor="rgba(15,23,42,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#cbd5e1"),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
)


def fig_to_json(fig: go.Figure) -> str:
    fig.update_layout(**_DARK_LAYOUT)
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def loss_distribution_fig(
    losses: np.ndarray,
    el: float,
    var_99: float,
    var_999: float,
    es_975: float,
    title: str = "Loss Distribution",
) -> str:
    losses_m = losses / 1e6
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=losses_m, nbinsx=80,
        marker_color=_PALETTE[0], opacity=0.8,
        name="Loss",
    ))
    for label, val, color in [
        ("EL", el / 1e6, "#22c55e"),
        ("VaR 99%", var_99 / 1e6, "#f59e0b"),
        ("VaR 99.9%", var_999 / 1e6, "#ef4444"),
        ("ES 97.5%", es_975 / 1e6, "#a855f7"),
    ]:
        fig.add_vline(x=val, line_width=2, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="top right")
    fig.update_layout(
        title=title,
        xaxis_title="Loss (€ millions)",
        yaxis_title="Frequency",
        showlegend=False,
        height=380,
    )
    return fig_to_json(fig)


def scenario_comparison_fig(scenarios: list[dict]) -> str:
    """Bar chart comparing EL, VaR99, VaR999 across scenarios."""
    names = [s["name"] for s in scenarios]
    metrics = ["el", "var_99", "var_999"]
    labels = ["Expected Loss", "VaR 99%", "VaR 99.9%"]
    colors = [_PALETTE[1], _PALETTE[2], _PALETTE[0]]

    fig = go.Figure()
    for metric, label, color in zip(metrics, labels, colors):
        vals = [s.get(metric, 0) / 1e6 for s in scenarios]
        fig.add_trace(go.Bar(name=label, x=names, y=vals, marker_color=color))
    fig.update_layout(
        title="Scenario Comparison — Credit Risk Metrics",
        barmode="group",
        xaxis_title="Scenario",
        yaxis_title="€ millions",
        height=400,
        legend=dict(orientation="h", y=-0.25),
    )
    return fig_to_json(fig)


def tornado_fig(base_pd: float, sensitivities: list[dict]) -> str:
    """Horizontal bar chart showing Δ PD per ±1σ shock to each factor."""
    sensitivities = sorted(sensitivities, key=lambda r: abs(r["down"]) + abs(r["up"]), reverse=True)
    factors = [r["factor"] for r in sensitivities]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=factors, x=[r["down"] * 100 for r in sensitivities],
        orientation="h", name="−1σ", marker_color=_PALETTE[0],
    ))
    fig.add_trace(go.Bar(
        y=factors, x=[r["up"] * 100 for r in sensitivities],
        orientation="h", name="+1σ", marker_color=_PALETTE[2],
    ))
    fig.update_layout(
        title=f"PD Sensitivity Tornado — base PD = {base_pd:.2%}",
        barmode="overlay",
        xaxis_title="Δ PD (pp)",
        height=380,
    )
    return fig_to_json(fig)


def macro_series_fig(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    cols = cols or list(df.columns)
    fig = go.Figure()
    for i, col in enumerate(cols):
        s = df[col].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            name=col, line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
        ))
    fig.update_layout(
        title="ECB Macroeconomic Panel",
        height=380,
        legend=dict(orientation="h", y=-0.30),
    )
    return fig_to_json(fig)


def pd_term_structure_fig(pd_paths: np.ndarray, horizon_q: int) -> str:
    qtrs = np.arange(1, horizon_q + 1)
    mean_pd = pd_paths.mean(axis=0) * 100
    p05 = np.quantile(pd_paths, 0.05, axis=0) * 100
    p95 = np.quantile(pd_paths, 0.95, axis=0) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([qtrs, qtrs[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill="toself", fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95% band",
    ))
    fig.add_trace(go.Scatter(
        x=qtrs, y=mean_pd, mode="lines+markers",
        name="Mean PD", line=dict(width=3, color=_PALETTE[0]),
    ))
    fig.update_layout(
        title="Simulated PD Term Structure",
        xaxis_title="Quarter",
        yaxis_title="PD (%)",
        height=350,
    )
    return fig_to_json(fig)


def gbm_paths_fig(paths: np.ndarray, debt: float, title: str = "Asset Value Paths") -> str:
    """Plot a sample of GBM asset value paths with the default threshold."""
    n_show = min(50, paths.shape[0])
    t = np.linspace(0, 1, paths.shape[1])
    fig = go.Figure()
    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=t, y=paths[i], mode="lines",
            line=dict(width=1, color="rgba(59,130,246,0.25)"),
            showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=t, y=paths.mean(axis=0), mode="lines",
        name="Mean path", line=dict(width=3, color=_PALETTE[0]),
    ))
    fig.add_hline(y=debt, line_dash="dash", line_color=_PALETTE[2],
                  annotation_text="Debt threshold D", annotation_font_color=_PALETTE[2])
    fig.update_layout(title=title, xaxis_title="Time (years)", yaxis_title="Asset Value", height=350)
    return fig_to_json(fig)


def correlation_heatmap_fig(corr: np.ndarray, names: list[str]) -> str:
    fig = go.Figure(data=go.Heatmap(
        z=np.round(corr, 2), x=names, y=names,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr, 2), texttemplate="%{text}",
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(title="Factor Correlation Matrix", height=380)
    return fig_to_json(fig)
