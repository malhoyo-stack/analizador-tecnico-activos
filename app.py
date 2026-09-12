from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_market_csv
from src.indicators import add_indicators
from src.levels import detect_support_resistance, nearest_levels
from src.fibonacci import analizar_fibonacci
from src.patterns import analizar_patrones
from src.analysis_engine import analyze
from src.charts import (
    price_chart,
    recent_four_month_chart,
    fibonacci_chart,
    pattern_chart,
    volume_chart,
    rsi_chart,
    macd_chart,
    dmi_chart,
)

st.set_page_config(page_title="Analizador técnico de activos", page_icon="📈", layout="wide")

st.title("📈 Analizador técnico de activos")
st.caption(
    "OHLCV diario · SMA21 / WMA30 / WMA150 / SMA200 · RSI · MACD · ADX/DMI · "
    "volumen · soportes/resistencias · Fibonacci · chartismo · estrategia operativa"
)

with st.sidebar:
    st.header("Datos y configuración")
    upload = st.file_uploader(
        "Cargar CSV",
        type=["csv"],
        help="Debe contener como mínimo fecha, Open, High, Low y Close. Volumen es recomendado.",
    )
    horizon = st.selectbox("Horizonte", ["Corto plazo", "Mediano plazo", "Largo plazo"], index=1)
    st.subheader("Medias en gráfico")
    enabled = [
        m for m in ["SMA21", "WMA30", "WMA150", "SMA200"]
        if st.checkbox(m.replace("SMA", "SMA ").replace("WMA", "WMA "), value=True, key=m)
    ]
    st.markdown("---")
    st.caption("El score es un índice heurístico de confirmación técnica, no una probabilidad de ganancia.")

if upload is None:
    sample = Path(__file__).parent / "sample_data" / "vist_us_d.csv"
    st.info("Carga un CSV. Para facilitar la primera prueba, se está mostrando el archivo VIST incluido como ejemplo.")
    source = sample
    filename = sample.name
else:
    source = upload
    filename = upload.name

try:
    df, report = load_market_csv(source)
except Exception as e:
    st.error(f"No se pudo leer el CSV: {e}")
    st.stop()

if not report.ok:
    st.error("El archivo no contiene la información mínima necesaria.")
    st.write("Faltan columnas:", ", ".join(report.missing_required))
    st.write("Columnas detectadas:", report.column_map)
    st.stop()

if report.warnings:
    with st.expander("Validación de datos", expanded=False):
        for warning in report.warnings:
            st.warning(warning)
        st.write(f"Filas de entrada: {report.rows_input} · filas utilizables: {report.rows_output}")
        st.write("Columnas interpretadas:", report.column_map)

idf = add_indicators(df)

# El selector de período se conserva para las visualizaciones históricas generales.
with st.sidebar:
    min_date, max_date = idf.Date.min().date(), idf.Date.max().date()
    selected = st.date_input(
        "Período visible",
        value=(max(min_date, (idf.Date.max() - pd.Timedelta(days=550)).date()), max_date),
        min_value=min_date,
        max_value=max_date,
    )
if isinstance(selected, (tuple, list)) and len(selected) == 2:
    view = idf[(idf.Date.dt.date >= selected[0]) & (idf.Date.dt.date <= selected[1])].copy()
else:
    view = idf.copy()

levels = detect_support_resistance(idf, lookback=260 if horizon != "Largo plazo" else 520)
fib = analizar_fibonacci(idf, levels)
pattern = analizar_patrones(idf)
result = analyze(idf, levels, horizon, fibonacci=fib, pattern=pattern)
sup, res = nearest_levels(levels)
last = idf.iloc[-1]
prev = idf.iloc[-2] if len(idf) > 1 else last
change = (float(last.Close) / float(prev.Close) - 1) * 100 if float(prev.Close) else np.nan
rsi = float(last.get("RSI14", np.nan))
adx = float(last.get("ADX14", np.nan))
pdi = float(last.get("PLUS_DI14", np.nan))
mdi = float(last.get("MINUS_DI14", np.nan))
relv = float(last.get("REL_VOLUME20", np.nan))
macdh = float(last.get("MACD_HIST", np.nan))
ticker = re.sub(r"[_-](us|ar|d|w|m).*", "", filename.rsplit(".", 1)[0], flags=re.I).upper()

# 1. Diagnóstico general
st.header("Diagnóstico general")
st.subheader(result.rating)
st.write(result.entry)
cols = st.columns(6)
cols[0].metric("Ticker", ticker)
cols[1].metric("Último precio", f"{last.Close:,.2f}", f"{change:+.2f}%")
cols[2].metric("Tendencia", result.trend)
cols[3].metric("RSI 14", f"{rsi:.1f}" if np.isfinite(rsi) else "N/D")
cols[4].metric("ADX 14", f"{adx:.1f}" if np.isfinite(adx) else "N/D")
cols[5].metric("Score técnico", f"{result.score}/100")
cols2 = st.columns(5)
cols2[0].metric("+DI / -DI", f"{pdi:.1f} / {mdi:.1f}" if np.isfinite(pdi) and np.isfinite(mdi) else "N/D")
cols2[1].metric("MACD hist.", f"{macdh:+.3f}" if np.isfinite(macdh) else "N/D")
cols2[2].metric("Vol. relativo", f"{relv:.2f}×" if np.isfinite(relv) else "N/D")
cols2[3].metric("Soporte", f"~{sup['level']:.2f}" if sup else "N/D")
cols2[4].metric("Resistencia", f"~{res['level']:.2f}" if res else "N/D")

# 2. Condiciones para entrar
st.header("Condiciones para entrada")
for item in result.checklist:
    st.markdown(f"{item.estado} **{item.condicion}** — {item.detalle}")

if result.missing_conditions and result.rating != "🟢 ENTRADA CONFIRMADA":
    st.markdown("**Qué tiene que pasar ahora:**")
    for numero, condicion in enumerate(result.missing_conditions[:7], start=1):
        st.write(f"{numero}. {condicion}")

# 3. Estrategia operativa
st.header("Estrategia operativa")
if result.strategy is None:
    st.warning("No se construye una estrategia larga: la estructura actual no ofrece un trigger y una invalidación suficientemente defendibles.")
else:
    s = result.strategy
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entrada estimada", f"{s.entrada:.2f}")
    c2.metric("Stop / invalidación", f"{s.stop:.2f}")
    c3.metric("Objetivo 1", f"{s.objetivo_1:.2f}" if s.objetivo_1 is not None else "N/D")
    c4.metric("R/R objetivo 1", f"{s.rr_1:.2f}:1" if s.rr_1 is not None else "N/D")
    st.write(f"**Tipo:** {s.tipo}")
    st.write(f"**Zona de entrada:** {s.zona_entrada[0]:.2f} – {s.zona_entrada[1]:.2f}")
    st.write(f"**Condición de activación:** {s.trigger}")
    if s.objetivo_2 is not None:
        st.write(f"**Objetivo 2:** {s.objetivo_2:.2f}" + (f" · R/R {s.rr_2:.2f}:1" if s.rr_2 is not None else ""))
    st.write(f"**Base técnica:** {s.argumento}")
    st.caption(f"Origen de objetivos: {s.fuente_objetivos}.")

# 4. Gráfico de cuatro meses
st.header("Gráfico de velas — últimos cuatro meses")
st.plotly_chart(
    recent_four_month_chart(idf, enabled, levels),
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)

# 5. Fibonacci
st.header("Fibonacci")
st.write(fib.explicacion)
if fib.disponible:
    if fib.anclajes:
        st.caption(
            "Anclajes: " + " · ".join(
                f"{chr(65 + i)} {a.tipo} {a.fecha.date()} @ {a.precio:.2f}" for i, a in enumerate(fib.anclajes)
            )
        )
    st.plotly_chart(fibonacci_chart(idf, fib), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    if fib.confluencias:
        st.markdown("**Zonas de confluencia técnica**")
        for texto in fib.confluencias:
            st.write("•", texto)
else:
    st.info(fib.explicacion)

# 6. Chartismo
st.header("Análisis chartista")
if pattern.detectado:
    estado_legible = {
        "confirmado": "ruptura bajista confirmada",
        "potencial": "potencial, sin ruptura bajista confirmada",
        "invalidado": "figura bajista invalidada",
        "no_aplica": "estructura detectada",
    }.get(result.pattern_status, result.pattern_status)
    st.write(
        f"**{pattern.nombre}** · confianza geométrica aproximada {pattern.confianza:.0%} · **{estado_legible}**"
    )
    st.write(pattern.descripcion)
    st.plotly_chart(pattern_chart(idf, pattern), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
else:
    st.info(pattern.descripcion)

# 7. Lectura técnica resumida
st.header("Lectura técnica")
for p in result.summary:
    st.write("•", p)

# 8. Indicadores secundarios y período visible configurable
with st.expander("Gráfico histórico según período visible", expanded=False):
    st.plotly_chart(price_chart(view, enabled, levels), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

if "Volume" in view:
    st.subheader("Volumen")
    st.plotly_chart(volume_chart(view), use_container_width=True, config={"displaylogo": False})
else:
    st.subheader("Volumen")
    st.info("N/D: el CSV no contiene volumen.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("RSI")
    st.plotly_chart(rsi_chart(view), use_container_width=True, config={"displaylogo": False})
with c2:
    st.subheader("MACD")
    st.plotly_chart(macd_chart(view), use_container_width=True, config={"displaylogo": False})
st.subheader("ADX / DMI")
st.plotly_chart(dmi_chart(view), use_container_width=True, config={"displaylogo": False})

# 9. Soportes/resistencias, riesgos y escenarios
st.header("Soportes y resistencias")
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown("**Soportes estimados**")
    if levels["supports"]:
        for z in levels["supports"]:
            st.write(f"• zona ~{z['level']:.2f} · {z['touches']} pivote(s)")
    else:
        st.write("No se detectaron zonas robustas con el método automático.")
with cc2:
    st.markdown("**Resistencias estimadas**")
    if levels["resistances"]:
        for z in levels["resistances"]:
            st.write(f"• zona ~{z['level']:.2f} · {z['touches']} pivote(s)")
    else:
        st.write("No se detectaron zonas robustas con el método automático.")

st.header("Riesgos")
for r in result.risks:
    st.write("•", r)

st.header("Escenarios")
for s in result.scenarios:
    st.write("•", s)

with st.expander("Cómo se construye el score técnico"):
    st.write("El score resume confirmaciones y se usa sólo como apoyo comparativo; la decisión de entrada exige además trigger de precio, invalidación y R/R.")
    st.json({k: round(v, 1) for k, v in result.components.items()})
    if horizon == "Corto plazo":
        st.write("Ponderaciones: tendencia 25%, medias 20%, momentum 20%, volumen 15%, ADX/DMI 10%, estructura 10%.")
    elif horizon == "Largo plazo":
        st.write("Ponderaciones: tendencia 30%, medias 30%, momentum 10%, volumen 8%, ADX/DMI 7%, estructura 15%.")
    else:
        st.write("Ponderaciones: tendencia 28%, medias 25%, momentum 15%, volumen 10%, ADX/DMI 10%, estructura 12%.")

st.caption(
    "Herramienta de apoyo al análisis técnico. Las señales describen configuraciones y escenarios basados en datos históricos; "
    "no constituyen garantías de rendimiento futuro ni sustituyen la gestión de riesgo."
)
