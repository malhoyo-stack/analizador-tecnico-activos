from __future__ import annotations
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    denom = weights.sum()
    return s.rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, weights) / denom), raw=True)


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    return rsi.where(~both_zero, 50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef = ema(close, fast)
    es = ema(close, slow)
    line = ef - es
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx_dmi(df: pd.DataFrame, n: int = 14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_sm = plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    minus_sm = minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_sm / atr.replace(0, np.nan)
    minus_di = 100 * minus_sm / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return adx, plus_di, minus_di


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA21"] = sma(out["Close"], 21)
    out["WMA30"] = wma(out["Close"], 30)
    out["WMA150"] = wma(out["Close"], 150)
    out["SMA200"] = sma(out["Close"], 200)
    out["RSI14"] = rsi_wilder(out["Close"], 14)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(out["Close"], 12, 26, 9)
    out["ADX14"], out["PLUS_DI14"], out["MINUS_DI14"] = adx_dmi(out, 14)
    out["ATR14"] = atr_wilder(out, 14)
    if "Volume" in out:
        out["VOL_SMA20"] = sma(out["Volume"], 20)
        out["REL_VOLUME20"] = out["Volume"] / out["VOL_SMA20"].replace(0, np.nan)
        price_ret = out["Close"].pct_change(10)
        vol_trend = out["Volume"].rolling(10).mean().pct_change(10)
        out["PRICE_VOL_DIVERGENCE"] = np.select(
            [(price_ret > 0.03) & (vol_trend < -0.10), (price_ret < -0.03) & (vol_trend < -0.10)],
            ["Alcista con volumen decreciente", "Caída con presión vendedora decreciente"],
            default="Sin divergencia simple relevante",
        )
    return out
