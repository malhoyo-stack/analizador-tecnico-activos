from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .levels import nearest_levels
from .fibonacci import FibonacciAnalysis, analizar_fibonacci
from .patterns import PatternAnalysis, analizar_patrones


RR_MINIMO_ATRACTIVO = 1.8


@dataclass
class ChecklistItem:
    estado: str
    condicion: str
    detalle: str
    cumplida: bool | None


@dataclass
class StrategyPlan:
    tipo: str
    entrada: float
    zona_entrada: tuple[float, float]
    trigger: str
    stop: float
    objetivo_1: float | None
    objetivo_2: float | None
    rr_1: float | None
    rr_2: float | None
    argumento: str
    fuente_objetivos: str


@dataclass
class AnalysisResult:
    trend: str
    rating: str
    entry: str
    score: int
    summary: list[str]
    risks: list[str]
    scenarios: list[str]
    components: dict[str, float]
    checklist: list[ChecklistItem] = field(default_factory=list)
    missing_conditions: list[str] = field(default_factory=list)
    strategy: StrategyPlan | None = None
    volume_threshold: float | None = None
    fibonacci: FibonacciAnalysis | None = None
    pattern: PatternAnalysis | None = None
    pattern_status: str = "N/D"


def _slope(series: pd.Series, n: int = 10) -> float:
    s = series.dropna().tail(n)
    if len(s) < 4:
        return 0.0
    x = np.arange(len(s))
    coef = np.polyfit(x, s.to_numpy(), 1)[0]
    base = max(abs(float(s.mean())), 1e-9)
    return float(coef / base)


def _valor_seguro(fila: pd.Series, columna: str) -> float:
    valor = fila.get(columna, np.nan)
    return float(valor) if pd.notna(valor) and np.isfinite(float(valor)) else np.nan


def _umbral_volumen(df: pd.DataFrame) -> float | None:
    if "REL_VOLUME20" not in df or not df["REL_VOLUME20"].notna().any():
        return None
    serie = df["REL_VOLUME20"].dropna().tail(120)
    if len(serie) < 10:
        return 1.0
    # Umbral adaptativo: exige un volumen por encima de lo normal para el propio activo,
    # acotado para no volver imposible la confirmación en series extremadamente ruidosas.
    return float(np.clip(serie.quantile(0.60), 1.05, 1.35))


def _nivel_ruptura_futuro(df: pd.DataFrame, levels: dict, pattern: PatternAnalysis | None, atr: float) -> float:
    close = float(df.Close.iloc[-1])
    candidatos = [float(r["level"]) for r in levels.get("resistances", []) if float(r["level"]) > close * 1.001]
    if pattern and pattern.detectado:
        if pattern.direccion == "alcista" and pattern.ruptura and pattern.ruptura > close * 1.001:
            candidatos.append(float(pattern.ruptura))
        # Si existe una figura bajista todavía no confirmada, superar su nivel de
        # invalidación es una confirmación alcista útil. No se interpreta la mera
        # presencia geométrica de la figura como señal bajista confirmada.
        if pattern.direccion == "bajista" and pattern.invalidacion and pattern.invalidacion > close * 1.001:
            candidatos.append(float(pattern.invalidacion))
    if candidatos:
        base = min(candidatos)
    elif len(df) >= 21:
        base = float(df.High.iloc[-21:-1].max())
    else:
        base = float(df.High.iloc[:-1].max()) if len(df) > 1 else close
    return float(base + 0.10 * atr)


def _ruptura_confirmada(df: pd.DataFrame, atr: float, relv: float, umbral_volumen: float | None) -> tuple[bool, float | None]:
    if len(df) < 12:
        return False, None
    ventana = min(20, len(df) - 1)
    max_previo = float(df.High.iloc[-ventana - 1:-1].max())
    close = float(df.Close.iloc[-1])
    nivel = max_previo + 0.10 * atr
    volumen_ok = True if umbral_volumen is None else (np.isfinite(relv) and relv >= umbral_volumen)
    return bool(close > nivel and volumen_ok), float(nivel)


def _estado_patron_bajista(
    df: pd.DataFrame,
    pattern: PatternAnalysis | None,
    atr: float,
    relv: float,
    umbral_volumen: float | None,
) -> str:
    """Clasifica una figura bajista como potencial, confirmada o invalidada.

    Una geometría bajista por sí sola no equivale a una señal operativa bajista.
    Para tratarla como confirmada se exige ruptura por cierre del nivel técnico y
    persistencia o acompañamiento de volumen. Esto evita que un doble techo, cuña,
    canal o triángulo todavía en formación fuerce un "NO ENTRAR" prematuro.
    """
    if not pattern or not pattern.detectado or pattern.direccion != "bajista":
        return "no_aplica"

    close = float(df.Close.iloc[-1])
    prev_close = float(df.Close.iloc[-2]) if len(df) > 1 else close
    buffer = max(0.10 * atr, close * 0.001)

    if pattern.invalidacion is not None and close > float(pattern.invalidacion) + buffer:
        return "invalidado"

    if pattern.ruptura is None:
        return "potencial"

    ruptura = float(pattern.ruptura)
    ruptura_clara = close < ruptura - buffer
    if not ruptura_clara:
        return "potencial"

    persistencia = prev_close < ruptura
    if umbral_volumen is None:
        # Sin volumen se acepta una ruptura de magnitud claramente superior al ruido
        # o dos cierres consecutivos por debajo del nivel.
        confirmacion_extra = persistencia or close < ruptura - 0.25 * atr
    else:
        volumen_ok = np.isfinite(relv) and relv >= umbral_volumen
        confirmacion_extra = persistencia or volumen_ok

    return "confirmado" if confirmacion_extra else "potencial"


def _nivel_invalidez(
    df: pd.DataFrame,
    levels: dict,
    fibonacci: FibonacciAnalysis | None,
    pattern: PatternAnalysis | None,
    entrada: float,
    atr: float,
) -> float:
    candidatos: list[float] = []
    candidatos.extend(float(s["level"]) for s in levels.get("supports", []) if float(s["level"]) < entrada)
    ultimo = df.iloc[-1]
    for media in ["SMA21", "WMA30", "WMA150", "SMA200"]:
        v = _valor_seguro(ultimo, media)
        if np.isfinite(v) and v < entrada:
            candidatos.append(v)
    if fibonacci and fibonacci.disponible:
        candidatos.extend(float(v) for v in fibonacci.niveles.values() if float(v) < entrada)
    if pattern and pattern.detectado and pattern.invalidacion and pattern.invalidacion < entrada:
        candidatos.append(float(pattern.invalidacion))

    if candidatos:
        referencia = max(candidatos)
    else:
        ventana = min(20, len(df))
        referencia = float(df.Low.tail(ventana).min())
    stop = referencia - 0.25 * atr
    if stop >= entrada:
        stop = entrada - max(atr, entrada * 0.01)
    return float(stop)


def _objetivos(
    df: pd.DataFrame,
    levels: dict,
    fibonacci: FibonacciAnalysis | None,
    pattern: PatternAnalysis | None,
    entrada: float,
    stop: float,
) -> tuple[float | None, float | None, str]:
    candidatos: list[tuple[float, str]] = []
    for r in levels.get("resistances", []):
        v = float(r["level"])
        if v > entrada * 1.003:
            candidatos.append((v, "resistencia"))
    if fibonacci and fibonacci.disponible:
        for nombre, v in fibonacci.niveles.items():
            if float(v) > entrada * 1.003:
                candidatos.append((float(v), f"Fibonacci {nombre}"))
    if pattern and pattern.detectado and pattern.objetivo and pattern.objetivo > entrada * 1.003:
        candidatos.append((float(pattern.objetivo), f"objetivo de {pattern.nombre.lower()}"))

    # También se consideran máximos previos relevantes como referencias estructurales.
    if len(df) > 20:
        for ventana in [60, 120, 252]:
            if len(df) > ventana:
                maximo = float(df.High.iloc[-ventana - 1:-1].max())
                if maximo > entrada * 1.003:
                    candidatos.append((maximo, f"máximo previo {ventana} ruedas"))

    candidatos.sort(key=lambda t: t[0])
    depurados: list[tuple[float, str]] = []
    for valor, fuente in candidatos:
        if not depurados or abs(valor - depurados[-1][0]) / valor > 0.006:
            depurados.append((valor, fuente))

    if not depurados:
        # Fallback de gestión monetaria claramente identificado: no se presenta como resistencia real.
        riesgo = entrada - stop
        if riesgo <= 0:
            return None, None, "sin objetivos estructurales fiables"
        return entrada + 2.0 * riesgo, entrada + 3.0 * riesgo, "objetivos provisionales por múltiplos de riesgo (sin resistencia estructural próxima)"

    obj1, fuente1 = depurados[0]
    obj2, fuente2 = depurados[1] if len(depurados) > 1 else (None, None)
    fuente = fuente1 if fuente2 is None else f"{fuente1}; {fuente2}"
    return float(obj1), float(obj2) if obj2 is not None else None, fuente


def _rr(entrada: float, stop: float, objetivo: float | None) -> float | None:
    if objetivo is None:
        return None
    riesgo = entrada - stop
    beneficio = objetivo - entrada
    if riesgo <= 0 or beneficio <= 0:
        return None
    return float(beneficio / riesgo)


def _construir_estrategia(
    df: pd.DataFrame,
    levels: dict,
    fibonacci: FibonacciAnalysis | None,
    pattern: PatternAnalysis | None,
    trend: str,
    score: int,
    atr: float,
    relv: float,
    umbral_volumen: float | None,
) -> tuple[StrategyPlan | None, bool, float]:
    close = float(df.Close.iloc[-1])
    ruptura_ok, nivel_ruptura_previo = _ruptura_confirmada(df, atr, relv, umbral_volumen)
    trigger_futuro = _nivel_ruptura_futuro(df, levels, pattern, atr)

    soporte, _ = nearest_levels(levels)
    ultimo = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else ultimo
    vela_alcista = float(ultimo.Close) > float(ultimo.Open) and float(ultimo.Close) > float(prev.Close)
    rebote_ok = False
    if soporte:
        nivel_soporte = float(soporte["level"])
        rebote_ok = (
            float(ultimo.Low) <= nivel_soporte + 0.40 * atr
            and close >= nivel_soporte + 0.55 * atr
            and vela_alcista
            and trend == "Alcista"
        )

    if ruptura_ok and nivel_ruptura_previo is not None:
        tipo = "Entrada por ruptura confirmada"
        entrada = close
        trigger = f"Ruptura ya confirmada mediante cierre > {nivel_ruptura_previo:.2f}"
        confirmada = True
    elif rebote_ok and soporte:
        tipo = "Entrada por rebote en soporte"
        entrada = close
        trigger = f"Rebote confirmado sobre la zona de soporte {float(soporte['level']):.2f}"
        confirmada = True
    elif trend in {"Alcista", "Transición / indefinida"} and score >= 52:
        tipo = "Entrada por ruptura condicionada"
        entrada = trigger_futuro
        trigger = f"Activar sólo con cierre diario > {trigger_futuro:.2f}"
        if umbral_volumen is not None:
            trigger += f" y volumen relativo ≥ {umbral_volumen:.2f}×"
        confirmada = False
    else:
        return None, False, trigger_futuro

    stop = _nivel_invalidez(df, levels, fibonacci, pattern, entrada, atr)
    obj1, obj2, fuente = _objetivos(df, levels, fibonacci, pattern, entrada, stop)
    rr1, rr2 = _rr(entrada, stop, obj1), _rr(entrada, stop, obj2)
    ancho = max(0.15 * atr, entrada * 0.0015)
    argumento_partes = ["estructura de precio", "tendencia", "momentum"]
    if fibonacci and fibonacci.disponible and fibonacci.confluencias:
        argumento_partes.append("confluencia Fibonacci")
    if pattern and pattern.detectado:
        argumento_partes.append(pattern.nombre.lower())
    argumento = "Convergencia entre " + ", ".join(argumento_partes) + "."
    return StrategyPlan(
        tipo=tipo,
        entrada=float(entrada),
        zona_entrada=(float(entrada - ancho), float(entrada + ancho)),
        trigger=trigger,
        stop=float(stop),
        objetivo_1=obj1,
        objetivo_2=obj2,
        rr_1=rr1,
        rr_2=rr2,
        argumento=argumento,
        fuente_objetivos=fuente,
    ), confirmada, trigger_futuro




def _integrar_confluencia_chartista(fibonacci: FibonacciAnalysis | None, pattern: PatternAnalysis | None) -> None:
    """Añade confluencias entre Fibonacci y niveles geométricos del patrón detectado."""
    if not fibonacci or not fibonacci.disponible or not pattern or not pattern.detectado:
        return
    referencias: list[tuple[str, float]] = []
    if pattern.ruptura is not None:
        referencias.append((f"ruptura de {pattern.nombre.lower()}", float(pattern.ruptura)))
    if pattern.invalidacion is not None:
        referencias.append((f"invalidación de {pattern.nombre.lower()}", float(pattern.invalidacion)))
    for linea in pattern.lineas:
        referencias.append((linea.etiqueta.lower(), float(linea.precio_fin)))
    for nombre, nivel in fibonacci.niveles.items():
        for etiqueta, ref in referencias:
            if abs(float(nivel) - ref) / max(abs(float(nivel)), 1e-9) <= 0.009:
                texto = f"{nombre} en {float(nivel):.2f}: zona de confluencia con {etiqueta}."
                if texto not in fibonacci.confluencias:
                    fibonacci.confluencias.append(texto)

def analyze(
    df: pd.DataFrame,
    levels: dict,
    horizon: str = "Mediano plazo",
    fibonacci: FibonacciAnalysis | None = None,
    pattern: PatternAnalysis | None = None,
) -> AnalysisResult:
    x = df.dropna(subset=["Close"]).copy()
    if x.empty:
        raise ValueError("No hay cierres válidos para analizar.")
    last = x.iloc[-1]
    close = float(last.Close)
    horizon_key = horizon.lower()

    if fibonacci is None:
        fibonacci = analizar_fibonacci(x, levels)
    if pattern is None:
        pattern = analizar_patrones(x)
    _integrar_confluencia_chartista(fibonacci, pattern)

    # El score se conserva como índice heurístico comparable con la versión previa.
    if "corto" in horizon_key:
        weights = {"trend": 25, "ma": 20, "momentum": 20, "volume": 15, "dmi": 10, "structure": 10}
        slope_n = 8
    elif "largo" in horizon_key:
        weights = {"trend": 30, "ma": 30, "momentum": 10, "volume": 8, "dmi": 7, "structure": 15}
        slope_n = 30
    else:
        weights = {"trend": 28, "ma": 25, "momentum": 15, "volume": 10, "dmi": 10, "structure": 12}
        slope_n = 15

    long_mas = [c for c in ["WMA150", "SMA200"] if c in x and pd.notna(last[c])]
    short_mas = [c for c in ["SMA21", "WMA30"] if c in x and pd.notna(last[c])]
    above_long = np.mean([close > float(last[c]) for c in long_mas]) if long_mas else 0.5
    above_short = np.mean([close > float(last[c]) for c in short_mas]) if short_mas else 0.5
    long_slopes = np.mean([_slope(x[c], slope_n) > 0 for c in long_mas]) if long_mas else 0.5
    short_slopes = np.mean([_slope(x[c], slope_n) > 0 for c in short_mas]) if short_mas else 0.5

    vals = {c: float(last[c]) for c in ["SMA21", "WMA30", "WMA150", "SMA200"] if c in x and pd.notna(last[c])}
    ma_order_bull = len(vals) == 4 and vals["SMA21"] > vals["WMA30"] > vals["WMA150"] > vals["SMA200"]
    ma_order_bear = len(vals) == 4 and vals["SMA21"] < vals["WMA30"] < vals["WMA150"] < vals["SMA200"]

    trend_raw = 0.55 * above_long + 0.25 * long_slopes + 0.20 * above_short
    if trend_raw >= 0.72 and not ma_order_bear:
        trend = "Alcista"
    elif trend_raw <= 0.28 and not ma_order_bull:
        trend = "Bajista"
    else:
        w = x.tail(40)
        ret = close / float(w.Close.iloc[0]) - 1 if len(w) > 1 else 0
        trend = "Lateral" if abs(ret) < 0.05 else "Transición / indefinida"

    trend_score = np.clip(100 * (0.6 * above_long + 0.4 * long_slopes), 0, 100)
    ma_score = np.clip(100 * (0.45 * above_long + 0.35 * above_short + 0.2 * ((long_slopes + short_slopes) / 2)), 0, 100)

    rsi = _valor_seguro(last, "RSI14")
    macd = _valor_seguro(last, "MACD")
    sig = _valor_seguro(last, "MACD_SIGNAL")
    hist = _valor_seguro(last, "MACD_HIST")
    prev_hist = _valor_seguro(x.iloc[-2], "MACD_HIST") if len(x) > 1 else np.nan
    mom_parts = []
    if np.isfinite(rsi):
        mom_parts.append(1.0 if 50 <= rsi <= 68 else 0.65 if 40 <= rsi < 50 or 68 < rsi <= 75 else 0.35)
    if np.isfinite(macd) and np.isfinite(sig):
        mom_parts.append(1.0 if macd > sig and hist >= 0 else 0.55 if macd > sig else 0.25)
    momentum_score = 100 * (np.mean(mom_parts) if mom_parts else 0.5)

    adx = _valor_seguro(last, "ADX14")
    pdi = _valor_seguro(last, "PLUS_DI14")
    mdi = _valor_seguro(last, "MINUS_DI14")
    if np.isfinite(adx) and np.isfinite(pdi) and np.isfinite(mdi):
        dmi_score = 80 if pdi > mdi and adx >= 20 else 60 if pdi > mdi else 35 if adx < 20 else 20
    else:
        dmi_score = 50

    relv = _valor_seguro(last, "REL_VOLUME20")
    if np.isfinite(relv):
        volume_score = 90 if relv >= 1.4 else 75 if relv >= 1.0 else 55 if relv >= 0.75 else 35
    else:
        volume_score = 50

    support, resistance = nearest_levels(levels)
    structure_score = 60
    if support and resistance:
        ds = (close - support["level"]) / close
        dr = (resistance["level"] - close) / close
        structure_score = 80 if ds < 0.06 and dr > 0.06 else 55 if dr < 0.03 else 65
    elif support:
        structure_score = 70

    comps = {
        "trend": float(trend_score), "ma": float(ma_score), "momentum": float(momentum_score),
        "volume": float(volume_score), "dmi": float(dmi_score), "structure": float(structure_score),
    }
    score = int(round(sum(comps[k] * weights[k] for k in weights) / 100))

    atr = _valor_seguro(last, "ATR14")
    if not np.isfinite(atr) or atr <= 0:
        atr = max(float((x.High - x.Low).tail(20).median()), close * 0.01)
    umbral_vol = _umbral_volumen(x)
    estado_patron_bajista = _estado_patron_bajista(x, pattern, atr, relv, umbral_vol)

    summary: list[str] = [
        f"Tendencia predominante: {trend.lower()}. El cierre está {'por encima' if above_long >= 0.5 else 'por debajo'} de la mayoría de las medias largas disponibles."
    ]
    if short_mas:
        summary.append(f"Medias cortas: el precio está sobre {sum(close > float(last[c]) for c in short_mas)}/{len(short_mas)} de SMA21/WMA30.")
    if np.isfinite(rsi):
        summary.append(f"RSI(14) = {rsi:.1f}: {'positivo sin sobreextensión extrema' if 50 <= rsi < 70 else 'sobrecompra relativa' if rsi >= 70 else 'débil' if rsi < 45 else 'neutral'}.")
    if np.isfinite(macd) and np.isfinite(sig):
        mejora_hist = np.isfinite(prev_hist) and hist > prev_hist
        summary.append(f"MACD {'sobre' if macd > sig else 'bajo'} señal; histograma {hist:+.3f}{' y mejorando' if mejora_hist else ''}.")
    if pattern and pattern.detectado:
        if pattern.direccion == "bajista":
            etiqueta_estado = {
                "confirmado": "ruptura bajista confirmada",
                "invalidado": "figura bajista invalidada por el precio",
                "potencial": "figura bajista potencial, aún sin ruptura confirmada",
            }.get(estado_patron_bajista, "en evaluación")
            summary.append(
                f"Chartismo: {pattern.nombre} con confianza geométrica {pattern.confianza:.0%}; {etiqueta_estado}."
            )
        else:
            summary.append(f"Chartismo: {pattern.nombre} con confianza geométrica {pattern.confianza:.0%}.")
    if fibonacci and fibonacci.disponible:
        summary.append(f"Fibonacci: {fibonacci.modo} {fibonacci.direccion}; {len(fibonacci.confluencias)} confluencia(s) técnica(s) detectada(s).")

    checklist: list[ChecklistItem] = []
    checklist.append(ChecklistItem("✅" if trend == "Alcista" else "❌", "Tendencia primaria alcista", f"Clasificación actual: {trend}.", trend == "Alcista"))
    if long_mas:
        ok = all(close > float(last[c]) for c in long_mas)
        checklist.append(ChecklistItem("✅" if ok else "❌", "Precio sobre medias largas", f"Evaluadas: {', '.join(long_mas)}.", ok))
    else:
        checklist.append(ChecklistItem("⚪", "Precio sobre medias largas", "N/D: histórico insuficiente para WMA150/SMA200.", None))
    if short_mas:
        ok = all(close > float(last[c]) for c in short_mas)
        pendientes = all(_slope(x[c], slope_n) > 0 for c in short_mas)
        checklist.append(ChecklistItem("✅" if ok and pendientes else "⚠️" if ok else "❌", "Recuperación de SMA21/WMA30", f"Precio {'sobre' if ok else 'no está sobre'} ambas; pendientes {'positivas' if pendientes else 'no confirmadas'}.", ok and pendientes))
    if np.isfinite(macd) and np.isfinite(sig):
        ok = macd > sig and hist >= 0
        detalle = f"MACD {macd:.3f}, señal {sig:.3f}, histograma {hist:+.3f}."
        checklist.append(ChecklistItem("✅" if ok else "⚠️" if np.isfinite(prev_hist) and hist > prev_hist else "❌", "MACD confirmado", detalle, ok))
    if np.isfinite(rsi):
        ok = 50 <= rsi <= 70
        checklist.append(ChecklistItem("✅" if ok else "⚠️", "RSI en zona operable", f"RSI(14) = {rsi:.1f}; se prioriza impulso >50 sin perseguir lecturas muy extendidas.", ok))
    if np.isfinite(adx) and np.isfinite(pdi) and np.isfinite(mdi):
        ok = pdi > mdi and adx >= 20
        checklist.append(ChecklistItem("✅" if ok else "⚠️" if pdi > mdi else "❌", "+DI > -DI con fuerza suficiente", f"+DI {pdi:.1f}, -DI {mdi:.1f}, ADX {adx:.1f}.", ok))
    if umbral_vol is None:
        checklist.append(ChecklistItem("⚪", "Volumen de confirmación", "N/D: el CSV no contiene volumen utilizable.", None))
    else:
        ok = np.isfinite(relv) and relv >= umbral_vol
        checklist.append(ChecklistItem("✅" if ok else "❌", "Volumen de confirmación", f"Volumen relativo {relv:.2f}×; umbral adaptativo actual {umbral_vol:.2f}×.", ok))

    trigger_futuro = _nivel_ruptura_futuro(x, levels, pattern, atr)
    ruptura_ok, nivel_ruptura_previo = _ruptura_confirmada(x, atr, relv, umbral_vol)
    if ruptura_ok:
        checklist.append(ChecklistItem("✅", "Ruptura estructural", f"Cierre sobre el máximo previo ajustado por volatilidad ({nivel_ruptura_previo:.2f}).", True))
    else:
        checklist.append(ChecklistItem("❌", "Ruptura estructural", f"Falta cierre diario > {trigger_futuro:.2f} para validar la ruptura relevante más cercana.", False))

    if fibonacci and fibonacci.disponible:
        if fibonacci.confluencias:
            checklist.append(ChecklistItem("✅", "Confluencia Fibonacci", fibonacci.confluencias[0], True))
        else:
            checklist.append(ChecklistItem("⚠️", "Confluencia Fibonacci", "Hay niveles Fibonacci válidos, pero sin coincidencia cercana con niveles/medias principales.", False))
    if pattern and pattern.detectado:
        if pattern.direccion == "alcista" and pattern.ruptura:
            patron_ok = close > pattern.ruptura
            checklist.append(ChecklistItem("✅" if patron_ok else "⚠️", f"Confirmación de {pattern.nombre.lower()}", f"Nivel de ruptura estimado: {pattern.ruptura:.2f}.", patron_ok))
        elif pattern.direccion == "bajista":
            if estado_patron_bajista == "confirmado":
                checklist.append(ChecklistItem(
                    "❌",
                    f"Ruptura bajista: {pattern.nombre.lower()}",
                    f"La figura confirmó ruptura por cierre debajo de {float(pattern.ruptura):.2f}.",
                    False,
                ))
            elif estado_patron_bajista == "invalidado":
                checklist.append(ChecklistItem(
                    "✅",
                    f"Invalidación de {pattern.nombre.lower()}",
                    f"El precio superó la invalidación de la figura ({float(pattern.invalidacion):.2f}).",
                    True,
                ))
            else:
                detalle = pattern.descripcion
                if pattern.ruptura is not None:
                    detalle += f" Aún no existe cierre bajista confirmado por debajo de {float(pattern.ruptura):.2f}."
                checklist.append(ChecklistItem(
                    "⚠️",
                    f"Patrón bajista potencial: {pattern.nombre.lower()}",
                    detalle,
                    None,
                ))

    strategy, confirmada_por_precio, trigger_futuro = _construir_estrategia(
        x, levels, fibonacci, pattern, trend, score, atr, relv, umbral_vol
    )

    riesgos: list[str] = []
    if resistance and 0 <= (float(resistance["level"]) - close) / close < 0.03:
        riesgos.append(f"Resistencia cercana en {float(resistance['level']):.2f}; sin cierre por encima el riesgo de rechazo sigue activo.")
    if support and 0 <= (close - float(support["level"])) / close < 0.025:
        riesgos.append(f"Soporte cercano en {float(support['level']):.2f}; perderlo deterioraría la estructura reciente.")
    if np.isfinite(rsi) and rsi >= 70:
        riesgos.append("RSI en sobrecompra relativa: la entrada puede estar extendida aunque la tendencia siga siendo alcista.")
    if umbral_vol is not None and (not np.isfinite(relv) or relv < umbral_vol):
        riesgos.append(f"El volumen no alcanza el umbral adaptativo de confirmación ({umbral_vol:.2f}×).")
    if short_mas:
        dist_atr = max(abs(close - float(last[c])) / max(atr, 1e-9) for c in short_mas)
        if dist_atr > 2.5:
            riesgos.append("El precio está a más de 2,5 ATR de una media corta; aumenta el riesgo de perseguir un movimiento extendido.")
    if pattern and pattern.detectado and pattern.direccion == "bajista":
        if estado_patron_bajista == "confirmado":
            riesgos.append(
                f"{pattern.nombre} confirmó ruptura bajista; mientras no recupere el nivel roto, contradice una entrada larga."
            )
        elif estado_patron_bajista == "potencial":
            riesgos.append(
                f"Se detecta {pattern.nombre.lower()} como riesgo potencial, pero todavía NO confirmó ruptura bajista. "
                "Se trata como advertencia y no como señal automática de salida/no entrada."
            )

    rr_aceptable = bool(strategy and strategy.rr_1 is not None and strategy.rr_1 >= RR_MINIMO_ATRACTIVO)
    senales_clave = [c for c in checklist if c.cumplida is not None]
    cumplidas = sum(c.cumplida is True for c in senales_clave)
    proporcion = cumplidas / max(len(senales_clave), 1)

    # Una figura bajista geométricamente plausible no basta para forzar NO ENTRAR.
    # Se exige confirmación por ruptura del nivel técnico correspondiente.
    if trend == "Bajista" or score < 40 or estado_patron_bajista == "confirmado":
        rating = "🔴 NO ENTRAR"
        entry = "No existe actualmente una configuración técnica saludable para una posición larga."
    elif (
        confirmada_por_precio
        and score >= 62
        and proporcion >= 0.55
        and rr_aceptable
        and estado_patron_bajista != "potencial"
    ):
        rating = "🟢 ENTRADA CONFIRMADA"
        entry = "La entrada reúne confirmación de precio y una relación riesgo/beneficio técnicamente aceptable."
    elif strategy and score >= 55 and trend in {"Alcista", "Transición / indefinida"}:
        rating = "🟡 ENTRADA CONDICIONADA"
        entry = "Existe una configuración potencial, pero la compra no queda habilitada hasta que se cumpla el trigger indicado."
    else:
        rating = "🟠 ESPERAR"
        entry = "La evidencia es incompleta o contradictoria; todavía no existe una entrada suficientemente confirmada."

    if strategy and strategy.rr_1 is not None and strategy.rr_1 < RR_MINIMO_ATRACTIVO:
        if rating == "🟢 ENTRADA CONFIRMADA":
            rating = "🟡 ENTRADA CONDICIONADA"
            entry = "La señal de precio existe, pero el primer objetivo ofrece una relación riesgo/beneficio insuficiente para tratarla como entrada confirmada."
        riesgos.append(f"R/R al objetivo 1 = {strategy.rr_1:.2f}:1, por debajo del mínimo operativo de {RR_MINIMO_ATRACTIVO:.1f}:1.")

    missing_conditions: list[str] = []
    for item in checklist:
        if item.cumplida is False:
            missing_conditions.append(f"{item.condicion}: {item.detalle}")
    if strategy and not confirmada_por_precio:
        missing_conditions.insert(0, strategy.trigger + ".")
    if strategy and strategy.rr_1 is not None and strategy.rr_1 < RR_MINIMO_ATRACTIVO:
        missing_conditions.append(f"Mejorar el punto de entrada, stop u objetivo hasta obtener R/R ≥ {RR_MINIMO_ATRACTIVO:.1f}:1.")

    scenarios: list[str] = []
    scenarios.append(f"Escenario alcista: cierre > {trigger_futuro:.2f}" + (f" con volumen relativo ≥ {umbral_vol:.2f}×." if umbral_vol is not None else "."))
    if support:
        scenarios.append(f"Escenario de pullback: sostener {float(support['level']):.2f} y recuperar el máximo de la vela de reacción puede ofrecer una invalidación más corta.")
        scenarios.append(f"Escenario adverso: pérdida de {float(support['level']):.2f} con expansión de presión vendedora debilita la tesis larga.")
    elif strategy:
        scenarios.append(f"Escenario adverso: cierre por debajo de la invalidación propuesta ({strategy.stop:.2f}).")

    return AnalysisResult(
        trend=trend,
        rating=rating,
        entry=entry,
        score=score,
        summary=summary,
        risks=riesgos or ["No se detecta un riesgo técnico dominante; esto no elimina el riesgo de mercado."],
        scenarios=scenarios,
        components=comps,
        checklist=checklist,
        missing_conditions=missing_conditions,
        strategy=strategy,
        volume_threshold=umbral_vol,
        fibonacci=fibonacci,
        pattern=pattern,
        pattern_status=estado_patron_bajista if pattern and pattern.direccion == "bajista" else "no_aplica",
    )
