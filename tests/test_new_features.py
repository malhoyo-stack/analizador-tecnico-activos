import numpy as np
import pandas as pd

from src.indicators import add_indicators
from src.levels import detect_support_resistance
from src.fibonacci import detectar_swings, analizar_fibonacci
from src.patterns import analizar_patrones
from src.analysis_engine import analyze
from src.charts import recent_four_month_chart, fibonacci_chart, pattern_chart


def _ohlc_desde_cierre(close: np.ndarray, con_volumen: bool = True) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    fechas = pd.bdate_range("2025-01-02", periods=len(close))
    apertura = np.r_[close[0], close[:-1]]
    amplitud = np.maximum(0.8, close * 0.006)
    high = np.maximum(apertura, close) + amplitud
    low = np.minimum(apertura, close) - amplitud
    data = {
        "Date": fechas,
        "Open": apertura,
        "High": high,
        "Low": low,
        "Close": close,
    }
    if con_volumen:
        data["Volume"] = 1_000_000 * (1.0 + 0.25 * np.sin(np.arange(len(close)) / 7))
    return pd.DataFrame(data)


def _serie_impulso(continuacion: bool, direccion: str = "alcista") -> pd.DataFrame:
    if direccion == "alcista":
        segmentos = [
            np.linspace(112, 100, 16),
            np.linspace(100, 140, 31)[1:],
            np.linspace(140, 120, 26)[1:],
            np.linspace(120, 151 if continuacion else 132, 31)[1:],
        ]
    else:
        segmentos = [
            np.linspace(125, 140, 16),
            np.linspace(140, 100, 31)[1:],
            np.linspace(100, 120, 26)[1:],
            np.linspace(120, 89 if continuacion else 110, 31)[1:],
        ]
    close = np.concatenate(segmentos)
    # Oscilación muy pequeña para evitar una serie perfectamente lineal sin alterar la estructura principal.
    close = close + 0.25 * np.sin(np.arange(len(close)) / 2.5)
    return add_indicators(_ohlc_desde_cierre(close))


def test_deteccion_de_swings_relevantes():
    df = _serie_impulso(continuacion=False, direccion="alcista")
    swings = detectar_swings(df, ventana=len(df), factor_prominencia=0.6)
    assert len(swings) >= 3
    assert {s.tipo for s in swings} == {"maximo", "minimo"}


def test_fibonacci_retroceso_y_extension_alcista():
    retro = _serie_impulso(continuacion=False, direccion="alcista")
    fib_retro = analizar_fibonacci(retro, {"supports": [], "resistances": []}, ventana=len(retro))
    assert fib_retro.disponible
    assert fib_retro.direccion == "alcista"
    assert fib_retro.modo == "retroceso"
    assert set(["23,6 %", "38,2 %", "50,0 %", "61,8 %", "78,6 %"]).issubset(fib_retro.niveles)
    valores = list(fib_retro.niveles.values())
    assert valores[0] > valores[-1]

    ext = _serie_impulso(continuacion=True, direccion="alcista")
    fib_ext = analizar_fibonacci(ext, {"supports": [], "resistances": []}, ventana=len(ext))
    assert fib_ext.disponible
    assert fib_ext.direccion == "alcista"
    assert fib_ext.modo == "extension"
    assert fib_ext.niveles["161,8 %"] > fib_ext.niveles["127,2 %"] > fib_ext.niveles["100,0 %"]


def test_fibonacci_orientacion_bajista():
    df = _serie_impulso(continuacion=False, direccion="bajista")
    fib = analizar_fibonacci(df, {"supports": [], "resistances": []}, ventana=len(df))
    assert fib.disponible
    assert fib.direccion == "bajista"
    assert fib.modo == "retroceso"
    assert fib.niveles["23,6 %"] < fib.niveles["78,6 %"]


def test_ausencia_controlada_de_patron():
    close = np.full(90, 100.0)
    df = add_indicators(_ohlc_desde_cierre(close))
    patron = analizar_patrones(df)
    assert not patron.detectado
    assert "No se detecta" in patron.descripcion or "insuficiente" in patron.descripcion


def test_deteccion_de_canal_sintetico():
    n = 150
    i = np.arange(n)
    close = 80 + 0.18 * i + 2.2 * np.sin(2 * np.pi * i / 14)
    df = add_indicators(_ohlc_desde_cierre(close))
    patron = analizar_patrones(df, ventana=140)
    assert patron.detectado
    assert "Canal" in patron.nombre
    assert patron.direccion == "alcista"
    assert len(patron.lineas) >= 2


def test_estrategia_rr_y_graficos_nuevos():
    n = 260
    i = np.arange(n)
    close = 60 + 0.12 * i + 1.8 * np.sin(2 * np.pi * i / 18)
    close[-8:] += np.linspace(0, 4.5, 8)
    df = add_indicators(_ohlc_desde_cierre(close))
    levels = detect_support_resistance(df, lookback=220)
    fib = analizar_fibonacci(df, levels)
    patron = analizar_patrones(df)
    resultado = analyze(df, levels, "Mediano plazo", fib, patron)

    assert resultado.rating in {"🟢 ENTRADA CONFIRMADA", "🟡 ENTRADA CONDICIONADA", "🟠 ESPERAR", "🔴 NO ENTRAR"}
    assert len(resultado.checklist) >= 6
    if resultado.strategy is not None:
        s = resultado.strategy
        assert s.stop < s.entrada
        if s.objetivo_1 is not None and s.rr_1 is not None:
            esperado = (s.objetivo_1 - s.entrada) / (s.entrada - s.stop)
            assert np.isclose(s.rr_1, esperado)

    figuras = [
        recent_four_month_chart(df, ["SMA21", "WMA30", "WMA150", "SMA200"], levels),
        fibonacci_chart(df, fib),
        pattern_chart(df, patron),
    ]
    assert all(len(fig.data) > 0 for fig in figuras)


def test_funcionamiento_sin_volumen_y_con_historico_corto():
    i = np.arange(55)
    close = 100 + 0.15 * i + np.sin(i / 4)
    df = add_indicators(_ohlc_desde_cierre(close, con_volumen=False))
    levels = detect_support_resistance(df, lookback=55)
    resultado = analyze(df, levels, "Corto plazo")
    assert resultado.volume_threshold is None
    assert np.isnan(df.WMA150.iloc[-1])
    assert np.isnan(df.SMA200.iloc[-1])
    fig = recent_four_month_chart(df, ["SMA21", "WMA30", "WMA150", "SMA200"], levels)
    assert len(fig.data) > 0


def test_patron_bajista_potencial_no_fuerza_no_entrar():
    """Una figura bajista sin ruptura confirmada debe ser advertencia, no veto automático."""
    n = 280
    i = np.arange(n)
    close = 70 + 0.16 * i + 1.6 * np.sin(2 * np.pi * i / 22)
    df = add_indicators(_ohlc_desde_cierre(close))
    levels = detect_support_resistance(df, lookback=240)
    ultimo = float(df.Close.iloc[-1])

    from src.patterns import PatternAnalysis

    patron = PatternAnalysis(
        detectado=True,
        nombre="Doble techo",
        direccion="bajista",
        confianza=0.90,
        descripcion="Patrón bajista sintético todavía sin ruptura de cuello.",
        lineas=[],
        ruptura=ultimo * 0.92,      # cuello claramente por debajo: no se rompió
        objetivo=ultimo * 0.84,
        invalidacion=ultimo * 1.05, # techo por encima: figura aún potencial
    )

    resultado = analyze(df, levels, "Mediano plazo", pattern=patron)
    assert resultado.pattern_status == "potencial"
    assert resultado.rating != "🔴 NO ENTRAR"
    assert any("todavía NO confirmó ruptura bajista" in r for r in resultado.risks)


def test_patron_bajista_confirmado_si_puede_bloquear_entrada_larga():
    """Tras una ruptura bajista persistente, el patrón sí puede bloquear una entrada larga."""
    n = 280
    i = np.arange(n)
    close = 90 + 0.10 * i + 1.2 * np.sin(2 * np.pi * i / 20)
    # Fuerza dos cierres finales por debajo de un nivel de ruptura conocido.
    close[-3] = 118.0
    close[-2] = 112.0
    close[-1] = 110.0
    df = add_indicators(_ohlc_desde_cierre(close))
    levels = detect_support_resistance(df, lookback=240)

    from src.patterns import PatternAnalysis

    patron = PatternAnalysis(
        detectado=True,
        nombre="Doble techo",
        direccion="bajista",
        confianza=0.90,
        descripcion="Patrón bajista sintético con cuello roto.",
        lineas=[],
        ruptura=114.0,
        objetivo=104.0,
        invalidacion=125.0,
    )

    resultado = analyze(df, levels, "Mediano plazo", pattern=patron)
    assert resultado.pattern_status == "confirmado"
    assert resultado.rating == "🔴 NO ENTRAR"
