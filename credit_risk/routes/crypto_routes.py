"""Crypto Market Analytics — YTD metrics + liquidity/depth dashboard."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from flask import Blueprint, jsonify, render_template, request

crypto_bp = Blueprint("crypto", __name__, url_prefix="/crypto")

COIN_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
}

COIN_COLORS = {
    "BTC": "#f7931a",
    "ETH": "#627eea",
    "SOL": "#9945ff",
    "XRP": "#346aa9",
}

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter, sans-serif", size=11),
    margin=dict(l=8, r=8, t=8, b=8),
    hovermode="x unified",
    showlegend=False,
    xaxis=dict(gridcolor="#334155", zeroline=False, showspikes=True,
               spikecolor="#475569", spikethickness=1),
    yaxis=dict(gridcolor="#334155", zeroline=False),
)


# ── Data fetching ──────────────────────────────────────────────────────────────

def _ytd_start_ts() -> int:
    return int(datetime(datetime.now(timezone.utc).year, 1, 1, tzinfo=timezone.utc).timestamp())


def _fetch_ytd_prices(coin_id: str) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    resp = requests.get(url, params={
        "vs_currency": "usd",
        "from": _ytd_start_ts(),
        "to": int(time.time()),
    }, timeout=12)
    resp.raise_for_status()
    raw = resp.json()

    prices = pd.DataFrame(raw["prices"], columns=["ts", "price"])
    volumes = pd.DataFrame(raw["total_volumes"], columns=["ts", "volume"])
    prices["date"] = pd.to_datetime(prices["ts"], unit="ms", utc=True).dt.normalize()
    prices = prices.drop_duplicates("date").set_index("date")[["price"]]
    prices["volume"] = volumes["volume"].values[: len(prices)]
    return prices


def _fetch_market_data(coin_id: str) -> dict:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    resp = requests.get(url, params={
        "localization": "false",
        "tickers": "true",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
    }, timeout=12)
    resp.raise_for_status()
    return resp.json()


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _rolling_vol(prices: pd.Series, window: int = 30) -> pd.Series:
    """Annualized rolling volatility from log-returns."""
    return np.log(prices / prices.shift(1)).rolling(window).std() * np.sqrt(365)


def _drawdown(prices: pd.Series) -> pd.Series:
    return (prices - prices.cummax()) / prices.cummax()


def _compute_kpis(df: pd.DataFrame, market: dict) -> dict:
    p = df["price"]
    log_ret = np.log(p / p.shift(1)).dropna()
    vol = float(log_ret.rolling(30).std().iloc[-1] * np.sqrt(365) * 100)
    dd = _drawdown(p)
    max_dd = float(dd.min() * 100)
    cur_dd = float(dd.iloc[-1] * 100)
    ytd_ret = float((p.iloc[-1] / p.iloc[0] - 1) * 100)
    sharpe = float((log_ret.mean() * 365) / (log_ret.std() * np.sqrt(365))) if log_ret.std() > 0 else 0.0

    md = market.get("market_data", {})
    price_usd = md.get("current_price", {}).get("usd", 0)
    mcap = md.get("market_cap", {}).get("usd", 0)
    vol24h = md.get("total_volume", {}).get("usd", 0)
    vol_mcap = vol24h / mcap if mcap else 0

    # Best bid-ask spread from USD/USDT/USDC tickers
    tickers = market.get("tickers", [])
    usd_t = [t for t in tickers
             if t.get("target") in ("USD", "USDT", "USDC")
             and t.get("bid_ask_spread_percentage")]
    spread_bps = float(usd_t[0]["bid_ask_spread_percentage"]) * 100 if usd_t else None

    # Exchange volume share for depth chart (top 12 USD pairs)
    exch_vols = {}
    for t in tickers:
        if t.get("target") in ("USD", "USDT", "USDC"):
            name = t.get("market", {}).get("name", "Unknown")
            exch_vols[name] = exch_vols.get(name, 0) + (t.get("converted_volume", {}).get("usd") or 0)
    top_exchanges = sorted(exch_vols.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "price":        f"${price_usd:,.2f}",
        "ytd_return":   f"{ytd_ret:+.1f}%",
        "ytd_cls":      "green" if ytd_ret >= 0 else "red",
        "vol_30d":      f"{vol:.1f}%",
        "max_dd":       f"{max_dd:.1f}%",
        "cur_dd":       f"{cur_dd:.1f}%",
        "cur_dd_cls":   "red" if cur_dd < -1 else "green",
        "sharpe":       f"{sharpe:.2f}",
        "market_cap":   f"${mcap / 1e9:.1f}B",
        "vol_24h":      f"${vol24h / 1e9:.1f}B",
        "vol_mcap":     f"{vol_mcap:.4f}",
        "spread_bps":   f"{spread_bps:.1f} bps" if spread_bps is not None else "N/A",
        "top_exchanges": top_exchanges,
    }


# ── Chart builders ─────────────────────────────────────────────────────────────

def _price_fig(df: pd.DataFrame, coin: str) -> str:
    c = COIN_COLORS[coin]
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    fig = go.Figure(go.Scatter(
        x=df.index, y=df["price"],
        mode="lines",
        line=dict(color=c, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.07)",
        hovertemplate="$%{y:,.2f}<extra></extra>",
    ))
    layout = dict(**_LAYOUT_BASE)
    layout["yaxis"] = dict(gridcolor="#334155", zeroline=False, tickprefix="$")
    fig.update_layout(**layout)
    return fig.to_json()


def _vol_fig(df: pd.DataFrame) -> str:
    vol = _rolling_vol(df["price"]) * 100
    fig = go.Figure(go.Scatter(
        x=vol.index, y=vol,
        mode="lines",
        line=dict(color="#f59e0b", width=2),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.08)",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    layout = dict(**_LAYOUT_BASE)
    layout["yaxis"] = dict(gridcolor="#334155", zeroline=False, ticksuffix="%")
    fig.update_layout(**layout)
    return fig.to_json()


def _dd_fig(df: pd.DataFrame) -> str:
    dd = _drawdown(df["price"]) * 100
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd,
        mode="lines",
        line=dict(color="#ef4444", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.10)",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    layout = dict(**_LAYOUT_BASE)
    layout["yaxis"] = dict(gridcolor="#334155", zeroline=True,
                           zerolinecolor="#475569", ticksuffix="%")
    fig.update_layout(**layout)
    return fig.to_json()


def _volume_fig(df: pd.DataFrame) -> str:
    fig = go.Figure(go.Bar(
        x=df.index, y=df["volume"] / 1e9,
        marker_color="#3b82f6",
        marker_opacity=0.65,
        hovertemplate="$%{y:.2f}B<extra></extra>",
    ))
    layout = dict(**_LAYOUT_BASE)
    layout["yaxis"] = dict(gridcolor="#334155", zeroline=False, tickprefix="$", ticksuffix="B")
    layout["bargap"] = 0.15
    fig.update_layout(**layout)
    return fig.to_json()


def _exchange_depth_fig(top_exchanges: list[tuple[str, float]]) -> str:
    if not top_exchanges:
        return go.Figure().to_json()
    names = [e[0] for e in top_exchanges]
    vols = [e[1] / 1e6 for e in top_exchanges]
    total = sum(vols)
    shares = [v / total * 100 for v in vols]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names[::-1], x=vols[::-1],
        orientation="h",
        marker=dict(
            color=shares[::-1],
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#7dd3fc"]],
            showscale=False,
        ),
        text=[f"${v:.0f}M ({s:.1f}%)" for v, s in zip(vols[::-1], shares[::-1])],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
        hovertemplate="%{y}: $%{x:.0f}M<extra></extra>",
    ))
    layout = dict(**_LAYOUT_BASE)
    layout["xaxis"] = dict(gridcolor="#334155", zeroline=False, tickprefix="$", ticksuffix="M")
    layout["yaxis"] = dict(gridcolor="rgba(0,0,0,0)", zeroline=False)
    layout["margin"] = dict(l=8, r=80, t=8, b=8)
    fig.update_layout(**layout)
    return fig.to_json()


# ── Routes ─────────────────────────────────────────────────────────────────────

@crypto_bp.route("/")
def index():
    coin = request.args.get("coin", "BTC").upper()
    if coin not in COIN_IDS:
        coin = "BTC"
    return render_template("crypto.html", coin=coin, coins=list(COIN_IDS.keys()))


@crypto_bp.route("/api/data")
def api_data():
    coin = request.args.get("coin", "BTC").upper()
    if coin not in COIN_IDS:
        coin = "BTC"

    try:
        df = _fetch_ytd_prices(COIN_IDS[coin])
        market = _fetch_market_data(COIN_IDS[coin])
    except requests.HTTPError as e:
        return jsonify({"error": f"CoinGecko API error: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    kpis = _compute_kpis(df, market)

    return jsonify({
        "kpis":        kpis,
        "price_fig":   _price_fig(df, coin),
        "vol_fig":     _vol_fig(df),
        "dd_fig":      _dd_fig(df),
        "volume_fig":  _volume_fig(df),
        "depth_fig":   _exchange_depth_fig(kpis.pop("top_exchanges")),
    })
