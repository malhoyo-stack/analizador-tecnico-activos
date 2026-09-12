from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


TipoSwing = Literal["maximo", "minimo"]
ModoFibonacci = Literal["retroceso", "extension", "no_disponible"]
Direccion = Literal["alcista", "bajista", "indefinida"]


@dataclass(frozen=True)
class SwingPoint:
    indice: int
    fecha: pd.Timestamp
    precio: float
    tipo: TipoSwing
    prominencia: float = 0.0


@dataclass
class FibonacciAnalysis:
    disponible: bool
    modo: ModoFibonacci
    direccion: Direccion
    anclajes: list[SwingPoint] = field(default_factory=list)
    niveles: dict[str, float] = field(default_factory=dict)
    explicacion: str = ""
    confluencias: list[str] = field(default_factory=list)
    calidad: float = 0.0


def _atr_referencia(df: pd.DataFrame, ventana: int = 60) -> float:
    """Devuelve una escala de volatilidad robusta para filtrar movimientos irrelevantes."""
    if "ATR14" in df and df["ATR14"].notna().any():
        atr = float(df["ATR14"].dropna().tail(ventana).median())
        if np.isfinite(atr) and atr > 0:
            return atr
    rango = (df["High"] - df["Low"]).dropna().tail(ventana)
    if len(rango):
        valor = float(rango.median())
        if np.isfinite(valor) and valor > 0:
            return valor
    return max(float(df["Close"].iloc[-1]) * 0.01, 1e-9)


def detectar_swings(
    df: pd.DataFrame,
    ventana: int = 180,
    distancia_minima: int | None = None,
    factor_prominencia: float = 1.0,
) -> list[SwingPoint]:
    """Detecta swing highs/lows significativos mediante prominencia y separación temporal.

    La prominencia se escala con ATR/rango típico, evitando fijar una distancia monetaria
    absoluta que no sea transferible entre activos.
    """
    if len(df) < 12:
        return []

    work = df.tail(min(ventana, len(df))).copy()
    if distancia_minima is None:
        distancia_minima = max(3, min(10, len(work) // 20))

    atr_ref = _atr_referencia(work)
    prominencia = max(atr_ref * factor_prominencia, float(work["Close"].median()) * 0.004)

    idx_max, prop_max = find_peaks(
        work["High"].to_numpy(dtype=float),
        distance=distancia_minima,
        prominence=prominencia,
    )
    idx_min, prop_min = find_peaks(
        -work["Low"].to_numpy(dtype=float),
        distance=distancia_minima,
        prominence=prominencia,
    )

    base = int(work.index[0])
    swings: list[SwingPoint] = []
    for pos, prom in zip(idx_max, prop_max.get("prominences", np.zeros(len(idx_max)))):
        fila = work.iloc[int(pos)]
        swings.append(SwingPoint(base + int(pos), pd.Timestamp(fila.Date), float(fila.High), "maximo", float(prom)))
    for pos, prom in zip(idx_min, prop_min.get("prominences", np.zeros(len(idx_min)))):
        fila = work.iloc[int(pos)]
        swings.append(SwingPoint(base + int(pos), pd.Timestamp(fila.Date), float(fila.Low), "minimo", float(prom)))

    swings.sort(key=lambda s: s.indice)
    if not swings:
        return []

    # Alternancia: si aparecen dos swings consecutivos del mismo tipo se conserva el más extremo.
    depurados: list[SwingPoint] = []
    for swing in swings:
        if not depurados or swing.tipo != depurados[-1].tipo:
            depurados.append(swing)
            continue
        previo = depurados[-1]
        reemplazar = (
            swing.tipo == "maximo" and swing.precio >= previo.precio
        ) or (
            swing.tipo == "minimo" and swing.precio <= previo.precio
        )
        if reemplazar:
            depurados[-1] = swing
    return depurados


def _movimiento_significativo(a: SwingPoint, b: SwingPoint, atr_ref: float) -> bool:
    amplitud = abs(b.precio - a.precio)
    base = max(abs(a.precio), 1e-9)
    return amplitud >= max(2.5 * atr_ref, 0.035 * base)


def _seleccionar_impulso(swings: list[SwingPoint], atr_ref: float) -> tuple[SwingPoint, SwingPoint] | None:
    """Selecciona el impulso relevante más reciente, priorizando amplitud y recencia."""
    candidatos: list[tuple[float, SwingPoint, SwingPoint]] = []
    total = max(len(swings), 1)
    for i in range(len(swings) - 1):
        a, b = swings[i], swings[i + 1]
        if a.tipo == b.tipo or not _movimiento_significativo(a, b, atr_ref):
            continue
        amplitud_atr = abs(b.precio - a.precio) / max(atr_ref, 1e-9)
        recencia = (i + 1) / total
        puntaje = amplitud_atr * (0.7 + 0.6 * recencia)
        candidatos.append((puntaje, a, b))
    if not candidatos:
        return None
    # Se restringe a los movimientos más recientes para evitar anclar Fibonacci en estructuras obsoletas.
    recientes = candidatos[-4:]
    return max(recientes, key=lambda t: t[0])[1:]


def _niveles_retroceso(a: SwingPoint, b: SwingPoint) -> tuple[Direccion, dict[str, float]]:
    ratios = [("23,6 %", 0.236), ("38,2 %", 0.382), ("50,0 %", 0.5), ("61,8 %", 0.618), ("78,6 %", 0.786)]
    if a.tipo == "minimo" and b.tipo == "maximo":
        amplitud = b.precio - a.precio
        return "alcista", {nombre: b.precio - r * amplitud for nombre, r in ratios}
    amplitud = a.precio - b.precio
    return "bajista", {nombre: b.precio + r * amplitud for nombre, r in ratios}


def _niveles_extension(a: SwingPoint, b: SwingPoint, c: SwingPoint) -> tuple[Direccion, dict[str, float]]:
    ratios = [("100,0 %", 1.0), ("127,2 %", 1.272), ("161,8 %", 1.618), ("261,8 %", 2.618)]
    if a.tipo == "minimo" and b.tipo == "maximo":
        amplitud = b.precio - a.precio
        return "alcista", {nombre: c.precio + r * amplitud for nombre, r in ratios}
    amplitud = a.precio - b.precio
    return "bajista", {nombre: c.precio - r * amplitud for nombre, r in ratios}


def _buscar_correccion(swings: list[SwingPoint], b: SwingPoint, direccion: Direccion) -> SwingPoint | None:
    posteriores = [s for s in swings if s.indice > b.indice]
    tipo_esperado: TipoSwing = "minimo" if direccion == "alcista" else "maximo"
    candidatos = [s for s in posteriores if s.tipo == tipo_esperado]
    return candidatos[0] if candidatos else None


def _confluencias(
    niveles: dict[str, float],
    df: pd.DataFrame,
    levels: dict | None,
    tolerancia_pct: float = 0.009,
) -> list[str]:
    if not niveles:
        return []
    ultimo = df.iloc[-1]
    referencias: list[tuple[str, float]] = []
    if levels:
        for z in levels.get("supports", []):
            referencias.append(("soporte", float(z["level"])))
        for z in levels.get("resistances", []):
            referencias.append(("resistencia", float(z["level"])))
    for media in ["SMA21", "WMA30", "WMA150", "SMA200"]:
        if media in df and pd.notna(ultimo.get(media, np.nan)):
            referencias.append((media, float(ultimo[media])))

    hallazgos: list[str] = []
    for nombre, valor in niveles.items():
        coincidencias = [etiqueta for etiqueta, ref in referencias if abs(valor - ref) / max(abs(valor), 1e-9) <= tolerancia_pct]
        if coincidencias:
            hallazgos.append(f"{nombre} en {valor:.2f}: zona de confluencia con {', '.join(coincidencias[:3])}.")
    return hallazgos


def analizar_fibonacci(df: pd.DataFrame, levels: dict | None = None, ventana: int = 180) -> FibonacciAnalysis:
    """Determina automáticamente si corresponde retroceso o extensión de Fibonacci."""
    if len(df) < 30:
        return FibonacciAnalysis(False, "no_disponible", "indefinida", explicacion="Histórico insuficiente para identificar swings fiables.")

    swings = detectar_swings(df, ventana=ventana)
    if len(swings) < 2:
        return FibonacciAnalysis(False, "no_disponible", "indefinida", explicacion="No se identificaron swings suficientemente significativos.")

    atr_ref = _atr_referencia(df.tail(min(ventana, len(df))))
    impulso = _seleccionar_impulso(swings, atr_ref)
    if impulso is None:
        return FibonacciAnalysis(False, "no_disponible", "indefinida", explicacion="No se detectó un impulso con amplitud suficiente respecto de la volatilidad reciente.")

    a, b = impulso
    direccion, retrocesos = _niveles_retroceso(a, b)
    close = float(df["Close"].iloc[-1])
    c = _buscar_correccion(swings, b, direccion)
    amplitud = abs(b.precio - a.precio)
    calidad = min(1.0, amplitud / max(5.0 * atr_ref, 1e-9))

    usar_extension = False
    if c is not None:
        correccion = abs(c.precio - b.precio) / max(amplitud, 1e-9)
        correccion_valida = 0.18 <= correccion <= 0.88
        if direccion == "alcista":
            reanudacion = close >= b.precio or float(df["High"].tail(5).max()) >= b.precio
        else:
            reanudacion = close <= b.precio or float(df["Low"].tail(5).min()) <= b.precio
        usar_extension = correccion_valida and reanudacion

    if usar_extension and c is not None:
        direccion_ext, niveles = _niveles_extension(a, b, c)
        explicacion = (
            "Se utiliza extensión de Fibonacci porque se identificó una estructura impulso → corrección → "
            "reanudación, y el precio volvió a atacar o superar el swing que cerró el impulso previo."
        )
        confluencias = _confluencias(niveles, df, levels)
        return FibonacciAnalysis(True, "extension", direccion_ext, [a, b, c], niveles, explicacion, confluencias, calidad)

    explicacion = (
        f"Se utiliza retroceso de Fibonacci porque el último impulso {direccion} relevante todavía se encuentra "
        "en fase de corrección/recuperación sin una reanudación suficientemente confirmada como para proyectar extensiones."
    )
    confluencias = _confluencias(retrocesos, df, levels)
    return FibonacciAnalysis(True, "retroceso", direccion, [a, b], retrocesos, explicacion, confluencias, calidad)
