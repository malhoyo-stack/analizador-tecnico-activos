from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _cluster(values: list[float], tolerance_pct: float = 0.015) -> list[dict]:
    if not values:
        return []
    vals = sorted(float(v) for v in values if np.isfinite(v))
    clusters: list[list[float]] = []
    for v in vals:
        if not clusters:
            clusters.append([v]); continue
        center = float(np.mean(clusters[-1]))
        if abs(v - center) / max(center, 1e-12) <= tolerance_pct:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [{"level": float(np.mean(c)), "touches": len(c), "min": min(c), "max": max(c)} for c in clusters]


def detect_support_resistance(df: pd.DataFrame, lookback: int = 260) -> dict:
    work = df.tail(min(lookback, len(df))).copy()
    if len(work) < 20:
        return {"supports": [], "resistances": []}
    distance = max(3, len(work) // 50)
    high_idx, _ = find_peaks(work["High"].to_numpy(), distance=distance, prominence=np.nanstd(work["High"]) * 0.15)
    low_idx, _ = find_peaks(-work["Low"].to_numpy(), distance=distance, prominence=np.nanstd(work["Low"]) * 0.15)
    highs = work.iloc[high_idx]["High"].tolist()
    lows = work.iloc[low_idx]["Low"].tolist()
    tol = 0.012 if len(work) >= 100 else 0.02
    resist = _cluster(highs, tol)
    supports = _cluster(lows, tol)
    last = float(work["Close"].iloc[-1])
    resist = sorted([x for x in resist if x["level"] >= last * 0.985], key=lambda x: (x["level"], -x["touches"]))
    supports = sorted([x for x in supports if x["level"] <= last * 1.015], key=lambda x: (-x["level"], -x["touches"]))
    return {"supports": supports[:4], "resistances": resist[:4]}


def nearest_levels(levels: dict):
    support = levels["supports"][0] if levels.get("supports") else None
    resistance = levels["resistances"][0] if levels.get("resistances") else None
    return support, resistance
