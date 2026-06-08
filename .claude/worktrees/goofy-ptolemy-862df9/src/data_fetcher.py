"""ECB Data Portal fetcher with synthetic fallback.

Pulls macro and banking series from the ECB SDMX REST API and aligns them on a
quarterly grid. If the API is unreachable, generates a realistic synthetic
panel covering 2008-Q1 to today so the rest of the pipeline still runs.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import requests

ECB_BASE = "https://data-api.ecb.europa.eu/service/data"

SERIES = {
    "gdp":          ("MNA",  "Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N"),
    "hicp":         ("ICP",  "M.U2.N.000000.4.ANR"),
    "unemployment": ("LFSI", "M.I9.S.UNEHRT.TOTAL0.15_74.T"),
    "policy_rate":  ("FM",   "M.U2.EUR.4F.KR.DFR.LEV"),
    "euribor3m":    ("FM",   "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"),
    "yield_aaa10y": ("YC",   "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"),
    "yield_all10y": ("YC",   "B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y"),
    "npl_ratio":    ("SUP",  "Q.B01.W0._Z.I7005._T.SII._Z._Z._Z.PCT.C"),
}


@dataclass
class DataBundle:
    panel: pd.DataFrame      # quarterly panel of macro factors and NPL
    source: str              # "ecb" or "synthetic"
    notes: list[str]


def _fetch_one(dataset: str, key: str, timeout: int = 15) -> pd.Series | None:
    url = f"{ECB_BASE}/{dataset}/{key}?format=csvdata"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200 or not r.text.strip():
            return None
        df = pd.read_csv(io.StringIO(r.text))
        time_col = "TIME_PERIOD" if "TIME_PERIOD" in df.columns else df.columns[-2]
        val_col = "OBS_VALUE" if "OBS_VALUE" in df.columns else df.columns[-1]
        df = df[[time_col, val_col]].dropna()
        df[time_col] = pd.PeriodIndex(df[time_col].astype(str), freq=_freq(df[time_col].iloc[0])).to_timestamp(how="end")
        s = pd.Series(df[val_col].astype(float).values, index=df[time_col]).sort_index()
        return s
    except Exception:
        return None


def _freq(sample: str) -> str:
    s = str(sample)
    if "Q" in s:
        return "Q"
    if len(s) == 7 and "-" in s:
        return "M"
    if len(s) == 4:
        return "Y"
    return "D"


def _to_quarterly(s: pd.Series, how: str = "mean") -> pd.Series:
    if s is None:
        return None
    return s.resample("QE").mean() if how == "mean" else s.resample("QE").last()


def fetch_ecb(start: str = "2008-01-01") -> DataBundle:
    notes = []
    raw = {name: _fetch_one(ds, key) for name, (ds, key) in SERIES.items()}
    ok = {k: v for k, v in raw.items() if v is not None and len(v) > 8}
    if len(ok) < 5:
        notes.append("ECB API unreachable or sparse; using synthetic panel.")
        return DataBundle(panel=_synthetic_panel(start), source="synthetic", notes=notes)

    q = {name: _to_quarterly(s) for name, s in ok.items()}
    df = pd.DataFrame(q)
    df = df[df.index >= pd.Timestamp(start)]

    if "gdp" in df:
        df["gdp_growth"] = df["gdp"].pct_change(4, fill_method=None) * 100
    if "yield_all10y" in df and "yield_aaa10y" in df:
        df["spread"] = df["yield_all10y"] - df["yield_aaa10y"]

    keep = [c for c in ["gdp_growth", "unemployment", "policy_rate",
                         "euribor3m", "yield_aaa10y", "spread", "hicp",
                         "npl_ratio"] if c in df.columns]
    df = df[keep].dropna(how="all").ffill().bfill()
    if df.empty or len(df) < 12:
        notes.append("ECB returned too few rows; using synthetic panel.")
        return DataBundle(panel=_synthetic_panel(start), source="synthetic", notes=notes)
    if "npl_ratio" not in df.columns:
        df["npl_ratio"] = _synthetic_npl(df)
        notes.append("NPL series unavailable; synthesised from macro factors.")
    notes.append(f"Loaded {len(df)} quarterly observations from ECB Data Portal.")
    return DataBundle(panel=df, source="ecb", notes=notes)


def _synthetic_panel(start: str) -> pd.DataFrame:
    """Realistic synthetic euro-area macro panel calibrated by hand to look
    like 2008-2025: GFC, sovereign crisis, COVID, rate-hiking cycle."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(start=start, end=datetime.today(), freq="QE")
    n = len(idx)

    t = np.arange(n)
    gfc = np.exp(-0.5 * ((t - 4) / 2.0) ** 2)
    sov = np.exp(-0.5 * ((t - 16) / 3.0) ** 2)
    covid = np.exp(-0.5 * ((t - 49) / 1.5) ** 2)
    hike = 1 / (1 + np.exp(-(t - 58) / 2))

    gdp_growth = (1.5 - 6.0 * gfc - 1.5 * sov - 14.0 * covid + 0.8 * rng.standard_normal(n))
    unemployment = (8.5 + 3.5 * gfc + 4.0 * sov + 1.2 * covid - 1.5 * hike
                    + 0.3 * rng.standard_normal(n))
    policy_rate = np.maximum(-0.5, 4.0 - 4.5 * gfc - 4.5 * sov + 4.0 * hike
                              + 0.1 * rng.standard_normal(n))
    euribor3m = policy_rate + 0.1 + 0.1 * rng.standard_normal(n)
    yield_aaa10y = np.maximum(-0.5, 3.5 - 1.5 * gfc - 1.0 * sov + 2.5 * hike
                               + 0.3 * rng.standard_normal(n))
    spread = np.maximum(0.05, 0.4 + 1.5 * gfc + 2.5 * sov + 0.8 * covid + 0.4 * hike
                         + 0.15 * rng.standard_normal(n))
    hicp = 1.8 + 0.5 * gfc - 0.3 * sov + 0.2 * covid + 7.0 * hike * np.exp(-(t - 60) / 8) \
           + 0.4 * rng.standard_normal(n)

    df = pd.DataFrame({
        "gdp_growth": gdp_growth,
        "unemployment": unemployment,
        "policy_rate": policy_rate,
        "euribor3m": euribor3m,
        "yield_aaa10y": yield_aaa10y,
        "spread": spread,
        "hicp": hicp,
    }, index=idx)
    df["npl_ratio"] = _synthetic_npl(df)
    return df


def _synthetic_npl(df: pd.DataFrame) -> pd.Series:
    """Generate an NPL-ratio series consistent with the macro panel via a
    logit link with sensible signs, plus persistence."""
    g = df.get("gdp_growth", pd.Series(0, index=df.index)).fillna(0)
    u = df.get("unemployment", pd.Series(8, index=df.index)).fillna(8)
    s = df.get("spread", pd.Series(0.5, index=df.index)).fillna(0.5)
    logit = -3.5 - 0.08 * g + 0.12 * (u - 8) + 0.45 * s
    pd_unc = 1 / (1 + np.exp(-logit))
    npl = np.zeros(len(df))
    npl[0] = pd_unc.iloc[0] * 100
    for i in range(1, len(df)):
        npl[i] = 0.85 * npl[i - 1] + 0.15 * pd_unc.iloc[i] * 100
    return pd.Series(npl, index=df.index)
