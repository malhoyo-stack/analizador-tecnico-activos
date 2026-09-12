# Analizador técnico de activos

Aplicación Streamlit para análisis técnico de series diarias OHLCV. La versión actual conserva las funciones originales e incorpora una capa operativa destinada a responder dos preguntas:

1. **¿Existe una oportunidad técnica razonable de entrada?**
2. **Si todavía no existe, qué debe ocurrir exactamente para habilitarla?**

## Funcionalidades

- Carga flexible de CSV con reconocimiento de columnas en español e inglés.
- Validación OHLCV y funcionamiento degradado cuando no existe volumen.
- SMA21, WMA30, WMA150 y SMA200.
- RSI(14), MACD(12,26,9), ADX/DMI(14) y ATR(14).
- Volumen relativo y SMA20 de volumen.
- Soportes y resistencias automáticos.
- Score técnico heurístico dependiente del horizonte.
- Gráfico específico de velas de los últimos cuatro meses.
- Checklist de condiciones de entrada con estados cumplido/pendiente/N-D.
- Trigger de entrada cuantificado, invalidación, objetivos y relación riesgo/beneficio.
- Estados operativos: **ENTRADA CONFIRMADA**, **ENTRADA CONDICIONADA**, **ESPERAR** y **NO ENTRAR**.
- Fibonacci automático mediante swings significativos:
  - retrocesos 23,6 / 38,2 / 50 / 61,8 / 78,6 %;
  - extensiones 100 / 127,2 / 161,8 / 261,8 %;
  - selección automática entre retroceso y extensión;
  - confluencias con soportes, resistencias, medias y niveles chartistas.
- Análisis chartista conservador de estructuras simples y defendibles:
  - canales alcistas, bajistas y laterales;
  - líneas de tendencia;
  - banderas;
  - triángulos;
  - cuñas;
  - doble techo y doble suelo.
- Gráficos Plotly independientes para Fibonacci y chartismo.

## Instalación

Se recomienda Python 3.11 o 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```text
analizador_activos_financieros/
├── app.py
├── requirements.txt
├── README.md
├── sample_data/
│   └── vist_us_d.csv
├── src/
│   ├── analysis_engine.py
│   ├── charts.py
│   ├── data_loader.py
│   ├── fibonacci.py
│   ├── indicators.py
│   ├── levels.py
│   └── patterns.py
└── tests/
    ├── test_core.py
    └── test_new_features.py
```

## Lógica de entrada

El score sigue siendo un índice heurístico y **no activa por sí solo una compra**. El motor analiza adicionalmente:

- tendencia y posición frente a medias;
- MACD y evolución del histograma;
- RSI;
- ADX/DMI;
- volumen relativo frente a un umbral adaptativo calculado con el propio histórico;
- ruptura estructural ajustada por ATR;
- soportes/resistencias;
- Fibonacci y sus confluencias;
- estructuras chartistas;
- punto de invalidación y relación riesgo/beneficio.

Cuando la ruptura todavía no ocurrió, la aplicación construye una **entrada condicionada** e indica el cierre requerido y, cuando hay volumen, el volumen relativo mínimo requerido. Si la estructura es bajista o claramente desfavorable, no fuerza una estrategia.

## Fibonacci

Los swings se detectan con `scipy.signal.find_peaks`, usando separación temporal y prominencia escalada por ATR/rango reciente. El algoritmo busca el impulso relevante más reciente y evalúa si el precio sigue corrigiendo ese movimiento o si ya existe una secuencia impulso → corrección → reanudación. En el primer caso usa retrocesos; en el segundo, extensiones.

Los anclajes seleccionados se muestran explícitamente en el gráfico para que el cálculo sea auditable.

## Chartismo

La detección automática prioriza estructuras geométricas simples. Las directrices se construyen a partir de swings y regresiones lineales, con umbrales mínimos de ajuste y paralelismo/convergencia. Si no hay evidencia suficiente, la aplicación devuelve que no se detecta una figura robusta en lugar de forzar un patrón.

Patrones complejos como HCH/HCH invertido no se fuerzan actualmente: su automatización fiable requiere validaciones geométricas adicionales para reducir falsos positivos.

## Formato de CSV

Mínimo: fecha, apertura, máximo, mínimo y cierre. Volumen es opcional pero recomendable. El cargador reconoce nombres habituales en inglés y español, elimina fechas duplicadas conservando la última, ordena cronológicamente y descarta filas con OHLC inválido o inconsistente. No rellena precios faltantes.

## Tests

```bash
pytest -q
```

La suite cubre, entre otros casos:

- carga y ordenamiento;
- indicadores;
- swings;
- orientación de Fibonacci alcista/bajista;
- selección retroceso/extensión;
- ausencia controlada de patrones;
- detección de canal sintético;
- estrategia y cálculo R/R;
- construcción de gráficos Plotly;
- CSV sin volumen;
- históricos cortos.

## Advertencia

La aplicación es una herramienta de apoyo al análisis técnico. Las clasificaciones y estrategias representan configuraciones basadas en datos históricos y no constituyen garantías de rendimiento futuro.
