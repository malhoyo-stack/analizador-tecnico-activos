from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .fibonacci import detectar_swings, SwingPoint


@dataclass
class PatternLine:
    fecha_inicio: pd.Timestamp
    precio_inicio: float
    fecha_fin: pd.Timestamp
    precio_fin: float
    etiqueta: str


@dataclass
class PatternAnalysis:
    detectado: bool
    nombre: str = ""
    direccion: Literal["alcista", "bajista", "neutral"] = "neutral"
    confianza: float = 0.0
    descripcion: str = "No se detecta actualmente una figura chartista suficientemente robusta."
    lineas: list[PatternLine] = field(default_factory=list)
    ruptura: float | None = None
    objetivo: float | None = None
    invalidacion: float | None = None


def _ajuste_lineal(puntos: list[SwingPoint], base_indice: int) -> tuple[float, float, float] | None:
    if len(puntos) < 3:
        return None
    x = np.array([p.indice - base_indice for p in puntos], dtype=float)
    y = np.array([p.precio for p in puntos], dtype=float)
    if np.ptp(x) <= 0:
        return None
    pendiente, intercepto = np.polyfit(x, y, 1)
    pred = pendiente * x + intercepto
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return float(pendiente), float(intercepto), float(np.clip(r2, 0, 1))


def _linea_desde_ajuste(df: pd.DataFrame, ajuste: tuple[float, float, float], inicio: int, fin: int, etiqueta: str) -> PatternLine:
    pendiente, intercepto, _ = ajuste
    base = int(df.index[0])
    y0 = pendiente * (inicio - base) + intercepto
    y1 = pendiente * (fin - base) + intercepto
    f0 = pd.Timestamp(df.loc[inicio, "Date"])
    f1 = pd.Timestamp(df.loc[fin, "Date"])
    return PatternLine(f0, float(y0), f1, float(y1), etiqueta)


def _detectar_canal(df: pd.DataFrame, swings: list[SwingPoint]) -> PatternAnalysis | None:
    if len(swings) < 6:
        return None
    bajos = [s for s in swings if s.tipo == "minimo"][-4:]
    altos = [s for s in swings if s.tipo == "maximo"][-4:]
    if len(bajos) < 3 or len(altos) < 3:
        return None

    base = int(df.index[0])
    aj_bajos = _ajuste_lineal(bajos, base)
    aj_altos = _ajuste_lineal(altos, base)
    if aj_bajos is None or aj_altos is None:
        return None
    sl, _, r2l = aj_bajos
    sh, _, r2h = aj_altos
    precio = max(float(df.Close.iloc[-1]), 1e-9)
    sl_pct, sh_pct = sl / precio, sh / precio
    paralelismo = abs(sl - sh) / max(abs(sl) + abs(sh), precio * 1e-4)
    if r2l < 0.68 or r2h < 0.68 or paralelismo > 0.65:
        return None

    if sl_pct > 0.00035 and sh_pct > 0.00035:
        nombre, direccion = "Canal alcista", "alcista"
    elif sl_pct < -0.00035 and sh_pct < -0.00035:
        nombre, direccion = "Canal bajista", "bajista"
    elif abs(sl_pct) < 0.00045 and abs(sh_pct) < 0.00045:
        nombre, direccion = "Canal lateral", "neutral"
    else:
        return None

    inicio = max(min(bajos[0].indice, altos[0].indice), int(df.index[0]))
    fin = int(df.index[-1])
    linea_baja = _linea_desde_ajuste(df, aj_bajos, inicio, fin, "Límite inferior")
    linea_alta = _linea_desde_ajuste(df, aj_altos, inicio, fin, "Límite superior")
    limite_inf = linea_baja.precio_fin
    limite_sup = linea_alta.precio_fin
    anchura = max(limite_sup - limite_inf, 0.0)
    ruptura = limite_sup if direccion == "alcista" else limite_inf if direccion == "bajista" else None
    objetivo = (ruptura + anchura) if direccion == "alcista" and ruptura else (ruptura - anchura) if direccion == "bajista" and ruptura else None
    invalidacion = limite_inf if direccion == "alcista" else limite_sup if direccion == "bajista" else None
    confianza = float((r2l + r2h) / 2 * max(0.0, 1.0 - paralelismo * 0.5))
    return PatternAnalysis(
        True,
        nombre,
        direccion,  # type: ignore[arg-type]
        confianza,
        f"{nombre} detectado a partir de swings recientes con líneas aproximadamente paralelas y ajuste geométrico consistente.",
        [linea_baja, linea_alta],
        ruptura,
        objetivo,
        invalidacion,
    )


def _detectar_triangulo(df: pd.DataFrame, swings: list[SwingPoint]) -> PatternAnalysis | None:
    bajos = [s for s in swings if s.tipo == "minimo"][-4:]
    altos = [s for s in swings if s.tipo == "maximo"][-4:]
    if len(bajos) < 3 or len(altos) < 3:
        return None
    base = int(df.index[0])
    aj_bajos = _ajuste_lineal(bajos, base)
    aj_altos = _ajuste_lineal(altos, base)
    if aj_bajos is None or aj_altos is None:
        return None
    sl, _, r2l = aj_bajos
    sh, _, r2h = aj_altos
    precio = max(float(df.Close.iloc[-1]), 1e-9)
    if min(r2l, r2h) < 0.68:
        return None

    slp, shp = sl / precio, sh / precio
    if slp > 0.0003 and abs(shp) < 0.00045:
        nombre, direccion = "Triángulo ascendente", "alcista"
    elif shp < -0.0003 and abs(slp) < 0.00045:
        nombre, direccion = "Triángulo descendente", "bajista"
    elif slp > 0.00025 and shp < -0.00025:
        nombre, direccion = "Triángulo simétrico", "neutral"
    elif slp > shp > 0.0002 and (slp - shp) > 0.00015:
        nombre, direccion = "Cuña ascendente", "bajista"
    elif shp < slp < -0.0002 and (slp - shp) > 0.00015:
        nombre, direccion = "Cuña descendente", "alcista"
    else:
        return None

    inicio = max(min(bajos[0].indice, altos[0].indice), int(df.index[0]))
    fin = int(df.index[-1])
    lin_b = _linea_desde_ajuste(df, aj_bajos, inicio, fin, "Directriz inferior")
    lin_a = _linea_desde_ajuste(df, aj_altos, inicio, fin, "Directriz superior")
    altura = max(s.precio for s in altos) - min(s.precio for s in bajos)
    ruptura = lin_a.precio_fin if direccion != "bajista" else lin_b.precio_fin
    objetivo = ruptura + altura if direccion == "alcista" else ruptura - altura if direccion == "bajista" else None
    invalidacion = lin_b.precio_fin if direccion == "alcista" else lin_a.precio_fin if direccion == "bajista" else None
    return PatternAnalysis(
        True, nombre, direccion, float((r2l + r2h) / 2),
        f"{nombre} detectado por convergencia de directrices construidas con swings significativos.",
        [lin_b, lin_a], ruptura, objetivo, invalidacion,
    )


def _detectar_doble_extremo(df: pd.DataFrame, swings: list[SwingPoint]) -> PatternAnalysis | None:
    precio = max(float(df.Close.iloc[-1]), 1e-9)
    tolerancia = max(0.018, float(df.get("ATR14", pd.Series([precio * 0.01])).dropna().tail(30).median()) * 1.2 / precio)

    for tipo, nombre, direccion in [
        ("maximo", "Doble techo", "bajista"),
        ("minimo", "Doble suelo", "alcista"),
    ]:
        pts = [s for s in swings if s.tipo == tipo][-3:]
        if len(pts) < 2:
            continue
        p1, p2 = pts[-2], pts[-1]
        if p2.indice - p1.indice < 8:
            continue
        if abs(p1.precio - p2.precio) / max((p1.precio + p2.precio) / 2, 1e-9) > tolerancia:
            continue
        intermedios = [s for s in swings if p1.indice < s.indice < p2.indice and s.tipo != tipo]
        if not intermedios:
            continue
        cuello_pt = min(intermedios, key=lambda s: s.precio) if tipo == "maximo" else max(intermedios, key=lambda s: s.precio)
        altura = abs(((p1.precio + p2.precio) / 2) - cuello_pt.precio)
        if altura < precio * 0.025:
            continue
        ruptura = cuello_pt.precio
        objetivo = ruptura - altura if tipo == "maximo" else ruptura + altura
        invalidacion = max(p1.precio, p2.precio) if tipo == "maximo" else min(p1.precio, p2.precio)
        linea = PatternLine(p1.fecha, ruptura, p2.fecha, ruptura, "Línea de cuello")
        return PatternAnalysis(
            True, nombre, direccion, 0.76,
            f"{nombre} detectado: dos extremos comparables separados por una reacción intermedia suficientemente amplia.",
            [linea], ruptura, objetivo, invalidacion,
        )
    return None


def _detectar_bandera(df: pd.DataFrame) -> PatternAnalysis | None:
    """Heurística conservadora para bandera: impulso fuerte seguido por canal correctivo corto."""
    if len(df) < 45:
        return None
    work = df.tail(55).copy()
    close = work.Close.to_numpy(dtype=float)
    # Se prueban longitudes de mástil/corrección razonables, sin imponer precios absolutos.
    for corr_len in range(8, 19):
        mastil = work.iloc[: -corr_len]
        corr = work.iloc[-corr_len:]
        if len(mastil) < 20:
            continue
        inicio = float(mastil.Close.iloc[-20])
        fin = float(mastil.Close.iloc[-1])
        retorno = fin / inicio - 1
        atr = float(work.get("ATR14", pd.Series(dtype=float)).dropna().tail(30).median()) if "ATR14" in work else np.nan
        amplitud_min = max(0.07, (4 * atr / max(fin, 1e-9)) if np.isfinite(atr) else 0.07)
        if abs(retorno) < amplitud_min:
            continue
        x = np.arange(len(corr), dtype=float)
        ph, ih = np.polyfit(x, corr.High.to_numpy(dtype=float), 1)
        pl, il = np.polyfit(x, corr.Low.to_numpy(dtype=float), 1)
        precio = max(float(corr.Close.iloc[-1]), 1e-9)
        paralelas = abs(ph - pl) / max(abs(ph) + abs(pl), precio * 1e-4) < 0.7
        if not paralelas:
            continue
        if retorno > 0 and ph < 0 and pl < 0:
            nombre, direccion = "Bandera alcista", "alcista"
        elif retorno < 0 and ph > 0 and pl > 0:
            nombre, direccion = "Bandera bajista", "bajista"
        else:
            continue
        fecha0, fecha1 = pd.Timestamp(corr.Date.iloc[0]), pd.Timestamp(corr.Date.iloc[-1])
        sup0, sup1 = ih, ph * (len(corr) - 1) + ih
        inf0, inf1 = il, pl * (len(corr) - 1) + il
        ruptura = sup1 if direccion == "alcista" else inf1
        mastil_amp = abs(fin - inicio)
        objetivo = ruptura + mastil_amp if direccion == "alcista" else ruptura - mastil_amp
        invalidacion = inf1 if direccion == "alcista" else sup1
        return PatternAnalysis(
            True, nombre, direccion, 0.72,
            f"{nombre} detectada: impulso de magnitud relevante seguido por un canal correctivo corto y aproximadamente paralelo.",
            [
                PatternLine(fecha0, float(sup0), fecha1, float(sup1), "Límite superior"),
                PatternLine(fecha0, float(inf0), fecha1, float(inf1), "Límite inferior"),
            ],
            float(ruptura), float(objetivo), float(invalidacion),
        )
    return None


def _detectar_linea_tendencia(df: pd.DataFrame, swings: list[SwingPoint]) -> PatternAnalysis | None:
    """Fallback conservador: una sola directriz robusta cuando no hay figura completa."""
    base = int(df.index[0])
    precio = max(float(df.Close.iloc[-1]), 1e-9)
    candidatos: list[PatternAnalysis] = []
    for tipo, nombre, direccion, etiqueta in [
        ("minimo", "Línea de tendencia alcista", "alcista", "Directriz alcista"),
        ("maximo", "Línea de tendencia bajista", "bajista", "Directriz bajista"),
    ]:
        pts = [s for s in swings if s.tipo == tipo][-4:]
        ajuste = _ajuste_lineal(pts, base) if len(pts) >= 3 else None
        if ajuste is None:
            continue
        pendiente, _, r2 = ajuste
        pendiente_pct = pendiente / precio
        if r2 < 0.82:
            continue
        if direccion == "alcista" and pendiente_pct <= 0.0003:
            continue
        if direccion == "bajista" and pendiente_pct >= -0.0003:
            continue
        inicio, fin = pts[0].indice, int(df.index[-1])
        linea = _linea_desde_ajuste(df, ajuste, inicio, fin, etiqueta)
        candidatos.append(PatternAnalysis(
            True, nombre, direccion, float(r2 * 0.88),
            f"{nombre} detectada mediante al menos tres swings alineados con ajuste lineal consistente.",
            [linea], None, None, linea.precio_fin,
        ))
    return max(candidatos, key=lambda c: c.confianza) if candidatos else None


def analizar_patrones(df: pd.DataFrame, ventana: int = 160) -> PatternAnalysis:
    if len(df) < 35:
        return PatternAnalysis(False, descripcion="Histórico insuficiente para un análisis chartista robusto.")
    work = df.tail(min(ventana, len(df))).copy()
    swings = detectar_swings(work, ventana=len(work), factor_prominencia=0.8)

    candidatos = [
        _detectar_bandera(work),
        _detectar_doble_extremo(work, swings),
        _detectar_triangulo(work, swings),
        _detectar_canal(work, swings),
        _detectar_linea_tendencia(work, swings),
    ]
    validos = [c for c in candidatos if c is not None and c.detectado]
    if not validos:
        return PatternAnalysis(False)
    # Se muestra sólo la estructura más defendible para evitar sobreinterpretar el gráfico.
    return max(validos, key=lambda c: c.confianza)
