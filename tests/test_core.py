from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_market_csv
from src.indicators import sma, wma, rsi_wilder, macd, adx_dmi, add_indicators
from src.levels import detect_support_resistance
from src.analysis_engine import analyze
from src.charts import price_chart, volume_chart, rsi_chart, macd_chart, dmi_chart

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "sample_data" / "vist_us_d.csv"


def test_loader_and_sorting():
    df, report = load_market_csv(CSV)
    assert report.ok
    assert len(df) == 1746
    assert df.Date.is_monotonic_increasing
    assert set(["Date","Open","High","Low","Close","Volume"]).issubset(df.columns)


def test_sma_and_wma_exact_small_sample():
    s = pd.Series([1.,2.,3.,4.,5.])
    assert sma(s,3).iloc[-1] == 4.0
    # WMA3 = (3*1 + 4*2 + 5*3)/6 = 26/6
    assert np.isclose(wma(s,3).iloc[-1], 26/6)


def test_indicator_ranges_and_identities():
    df, _ = load_market_csv(CSV)
    x = add_indicators(df)
    assert x.SMA21.notna().sum() == len(x)-20
    assert x.WMA30.notna().sum() == len(x)-29
    assert x.WMA150.notna().sum() == len(x)-149
    assert x.SMA200.notna().sum() == len(x)-199
    assert x.RSI14.dropna().between(0,100).all()
    z = x.dropna(subset=["MACD","MACD_SIGNAL","MACD_HIST"])
    assert np.allclose(z.MACD_HIST, z.MACD-z.MACD_SIGNAL, rtol=1e-10, atol=1e-10)
    d = x.dropna(subset=["ADX14","PLUS_DI14","MINUS_DI14"])
    assert d.ADX14.between(0,100).all()
    assert d.PLUS_DI14.ge(0).all() and d.MINUS_DI14.ge(0).all()


def test_reusable_alias_loader(tmp_path):
    df, _ = load_market_csv(CSV)
    alt = df.rename(columns={"Date":"fecha","Open":"apertura","High":"máximo","Low":"mínimo","Close":"cierre","Volume":"volumen"})
    p = tmp_path / "otro_activo.csv"
    alt.to_csv(p,index=False)
    got, report = load_market_csv(p)
    assert report.ok and len(got)==len(df)


def test_analysis_and_figures_build():
    df,_=load_market_csv(CSV)
    x=add_indicators(df)
    levels=detect_support_resistance(x)
    result=analyze(x, levels, "Mediano plazo")
    assert 0 <= result.score <= 100
    figs=[price_chart(x.tail(300),["SMA21","WMA30","WMA150","SMA200"],levels), volume_chart(x.tail(300)), rsi_chart(x.tail(300)), macd_chart(x.tail(300)), dmi_chart(x.tail(300))]
    assert all(len(f.data)>0 for f in figs)
