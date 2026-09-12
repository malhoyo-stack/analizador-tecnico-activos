from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .fibonacci import FibonacciAnalysis
from .patterns import PatternAnalysis

MA_STYLE = {"SMA21": "SMA 21", "WMA30": "WMA 30", "WMA150": "WMA 150", "SMA200": "SMA 200"}


def _agregar_niveles(fig: go.Figure, levels: dict, max_zonas: int = 2, fila: int | None = None) -> None:
    kwargs = {} if fila is None else {"row": fila, "col": 1}
    for s in levels.get("supports", [])[:max_zonas]:
        fig.add_hrect(
            y0=s["min"] * 0.997,
            y1=s["max"] * 1.003,
            opacity=0.08,
            line_width=0,
            annotation_text="Soporte",
            **kwargs,
        )
    for r in levels.get("resistances", [])[:max_zonas]:
        fig.add_hrect(
            y0=r["min"] * 0.997,
            y1=r["max"] * 1.003,
            opacity=0.08,
            line_width=0,
            annotation_text="Resistencia",
            **kwargs,
        )


def _configurar_eje_temporal(fig: go.Figure) -> None:
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])


def price_chart(df: pd.DataFrame, enabled_mas: list[str], levels: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.Date, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="OHLC"))
    for c in enabled_mas:
        if c in df:
            fig.add_trace(go.Scatter(x=df.Date, y=df[c], mode="lines", name=MA_STYLE[c], line={"width": 1.6}))
    _agregar_niveles(fig, levels)
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), legend_orientation="h")
    _configurar_eje_temporal(fig)
    return fig


def recent_four_month_chart(df: pd.DataFrame, enabled_mas: list[str], levels: dict) -> go.Figure:
    """Velas de aproximadamente los últimos cuatro meses, sin recortar el cálculo de indicadores."""
    if df.empty:
        return go.Figure()
    fecha_max = pd.Timestamp(df.Date.max())
    fecha_min = fecha_max - pd.DateOffset(months=4)
    work = df[df.Date >= fecha_min].copy()
    if work.empty:
        work = df.tail(min(90, len(df))).copy()

    tiene_volumen = "Volume" in work and work["Volume"].notna().any()
    if tiene_volumen:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.76, 0.24])
        fila_precio = 1
    else:
        fig = make_subplots(rows=1, cols=1)
        fila_precio = 1

    fig.add_trace(
        go.Candlestick(x=work.Date, open=work.Open, high=work.High, low=work.Low, close=work.Close, name="OHLC"),
        row=fila_precio,
        col=1,
    )
    for c in enabled_mas:
        if c in work and work[c].notna().any():
            fig.add_trace(go.Scatter(x=work.Date, y=work[c], mode="lines", name=MA_STYLE[c], line={"width": 1.5}), row=fila_precio, col=1)
    _agregar_niveles(fig, levels, max_zonas=2, fila=fila_precio)

    if tiene_volumen:
        fig.add_trace(go.Bar(x=work.Date, y=work.Volume, name="Volumen", opacity=0.55), row=2, col=1)
        if "VOL_SMA20" in work and work.VOL_SMA20.notna().any():
            fig.add_trace(go.Scatter(x=work.Date, y=work.VOL_SMA20, name="SMA Volumen 20", mode="lines", line={"width": 1.3}), row=2, col=1)

    fig.update_layout(
        height=700 if tiene_volumen else 610,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=35, b=10),
        legend_orientation="h",
        title="Estructura reciente — últimos cuatro meses",
    )
    _configurar_eje_temporal(fig)
    return fig


def fibonacci_chart(df: pd.DataFrame, fib: FibonacciAnalysis) -> go.Figure:
    fecha_max = pd.Timestamp(df.Date.max())
    fecha_min = fecha_max - pd.DateOffset(months=7)
    work = df[df.Date >= fecha_min].copy()
    if work.empty:
        work = df.tail(min(150, len(df))).copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=work.Date, open=work.Open, high=work.High, low=work.Low, close=work.Close, name="OHLC"))
    if not fib.disponible:
        fig.add_annotation(text=fib.explicacion or "Fibonacci no disponible", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    else:
        for nombre, valor in fib.niveles.items():
            fig.add_hline(y=float(valor), line_dash="dot", opacity=0.60, annotation_text=f"{nombre} · {valor:.2f}", annotation_position="right")
        if fib.anclajes:
            fig.add_trace(
                go.Scatter(
                    x=[a.fecha for a in fib.anclajes],
                    y=[a.precio for a in fib.anclajes],
                    mode="lines+markers+text",
                    text=[chr(65 + i) for i in range(len(fib.anclajes))],
                    textposition="top center",
                    name="Anclajes Fibonacci",
                    line={"width": 2},
                    marker={"size": 9},
                )
            )
    titulo = "Fibonacci"
    if fib.disponible:
        titulo += f" — {fib.modo.capitalize()} {fib.direccion}"
    fig.update_layout(height=610, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10), legend_orientation="h", title=titulo)
    _configurar_eje_temporal(fig)
    return fig


def pattern_chart(df: pd.DataFrame, pattern: PatternAnalysis) -> go.Figure:
    fecha_max = pd.Timestamp(df.Date.max())
    fecha_min = fecha_max - pd.DateOffset(months=7)
    work = df[df.Date >= fecha_min].copy()
    if work.empty:
        work = df.tail(min(150, len(df))).copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=work.Date, open=work.Open, high=work.High, low=work.Low, close=work.Close, name="OHLC"))
    if not pattern.detectado:
        fig.add_annotation(text=pattern.descripcion, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    else:
        for linea in pattern.lineas:
            fig.add_trace(
                go.Scatter(
                    x=[linea.fecha_inicio, linea.fecha_fin],
                    y=[linea.precio_inicio, linea.precio_fin],
                    mode="lines",
                    name=linea.etiqueta,
                    line={"width": 2},
                )
            )
        if pattern.ruptura is not None:
            fig.add_hline(y=pattern.ruptura, line_dash="dash", annotation_text=f"Ruptura {pattern.ruptura:.2f}")
        if pattern.objetivo is not None:
            fig.add_hline(y=pattern.objetivo, line_dash="dot", annotation_text=f"Objetivo {pattern.objetivo:.2f}")
        if pattern.invalidacion is not None:
            fig.add_hline(y=pattern.invalidacion, line_dash="dashdot", annotation_text=f"Invalidación {pattern.invalidacion:.2f}")
    fig.update_layout(
        height=610,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend_orientation="h",
        title=pattern.nombre if pattern.detectado else "Análisis chartista",
    )
    _configurar_eje_temporal(fig)
    return fig


def volume_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "Volume" in df:
        fig.add_trace(go.Bar(x=df.Date, y=df.Volume, name="Volumen"))
        if "VOL_SMA20" in df:
            fig.add_trace(go.Scatter(x=df.Date, y=df.VOL_SMA20, name="SMA Volumen 20", mode="lines"))
    else:
        fig.add_annotation(text="Volumen no disponible", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=260, hovermode="x unified", margin=dict(l=10, r=10, t=25, b=10), legend_orientation="h")
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "RSI14" in df:
        fig.add_trace(go.Scatter(x=df.Date, y=df.RSI14, name="RSI 14", mode="lines"))
    for y in [30, 50, 70]:
        fig.add_hline(y=y, line_dash="dash", opacity=0.5)
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if all(c in df for c in ["MACD", "MACD_SIGNAL", "MACD_HIST"]):
        fig.add_trace(go.Scatter(x=df.Date, y=df.MACD, name="MACD", mode="lines"))
        fig.add_trace(go.Scatter(x=df.Date, y=df.MACD_SIGNAL, name="Señal", mode="lines"))
        fig.add_trace(go.Bar(x=df.Date, y=df.MACD_HIST, name="Histograma"))
    fig.add_hline(y=0, line_dash="dash", opacity=0.4)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=25, b=10), legend_orientation="h")
    return fig


def dmi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if all(c in df for c in ["ADX14", "PLUS_DI14", "MINUS_DI14"]):
        fig.add_trace(go.Scatter(x=df.Date, y=df.ADX14, name="ADX 14", mode="lines"))
        fig.add_trace(go.Scatter(x=df.Date, y=df.PLUS_DI14, name="+DI", mode="lines"))
        fig.add_trace(go.Scatter(x=df.Date, y=df.MINUS_DI14, name="-DI", mode="lines"))
    fig.add_hline(y=20, line_dash="dash", opacity=0.4)
    fig.add_hline(y=25, line_dash="dot", opacity=0.4)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=25, b=10), legend_orientation="h")
    return fig
