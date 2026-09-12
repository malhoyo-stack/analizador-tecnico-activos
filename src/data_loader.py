from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable
import io
import re
import pandas as pd
import numpy as np

REQUIRED = ("Date", "Open", "High", "Low", "Close")
ALIASES = {
    "Date": {"date", "fecha", "datetime", "time", "timestamp", "dia", "día"},
    "Open": {"open", "apertura", "abrir", "o"},
    "High": {"high", "max", "maximo", "máximo", "alto", "h"},
    "Low": {"low", "min", "minimo", "mínimo", "bajo", "l"},
    "Close": {"close", "cierre", "ultimo", "último", "last", "precio", "c"},
    "Volume": {"volume", "volumen", "vol", "v"},
}

@dataclass
class ValidationReport:
    column_map: dict[str, str]
    missing_required: list[str]
    duplicate_rows: int
    missing_values: dict[str, int]
    inconsistent_ohlc_rows: int
    invalid_numeric_rows: int
    invalid_date_rows: int
    rows_input: int
    rows_output: int
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_required and self.rows_output > 0


def _norm(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s_\-.]+", "", s)
    return s


def detect_columns(columns: Iterable[str]) -> dict[str, str]:
    normalized = {_norm(c): c for c in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases | {canonical.lower()}:
            key = _norm(alias)
            if key in normalized:
                mapping[canonical] = normalized[key]
                break
    return mapping


def _read_csv(source) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    if hasattr(source, "read"):
        raw = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        if isinstance(raw, str):
            raw = raw.encode()
        # Try common separators while preserving decimal dots/commas as best effort.
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep)
                if df.shape[1] >= 5:
                    return df
            except Exception:
                pass
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    raise TypeError("Fuente CSV no soportada")


def load_market_csv(source) -> tuple[pd.DataFrame, ValidationReport]:
    raw = _read_csv(source)
    rows_input = len(raw)
    mapping = detect_columns(raw.columns)
    missing_required = [c for c in REQUIRED if c not in mapping]
    warnings: list[str] = []

    if missing_required:
        report = ValidationReport(mapping, missing_required, 0, {}, 0, 0, 0, rows_input, 0, warnings)
        return pd.DataFrame(), report

    keep = [mapping[c] for c in mapping]
    df = raw[keep].copy().rename(columns={v: k for k, v in mapping.items()})

    date_before = df["Date"].notna().sum()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    invalid_date_rows = int(date_before - df["Date"].notna().sum())

    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df]
    invalid_numeric_rows = 0
    for c in numeric_cols:
        before = df[c].notna().sum()
        # Remove thousand separators/spaces only when values are strings.
        s = df[c].astype(str).str.strip().str.replace(" ", "", regex=False)
        # If comma occurs without dot, interpret comma as decimal separator.
        comma_decimal = s.str.contains(",", regex=False) & ~s.str.contains(".", regex=False)
        s.loc[comma_decimal] = s.loc[comma_decimal].str.replace(",", ".", regex=False)
        s.loc[~comma_decimal] = s.loc[~comma_decimal].str.replace(",", "", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce")
        invalid_numeric_rows += int(before - df[c].notna().sum())

    duplicate_rows = int(df.duplicated(subset=["Date"], keep="last").sum())
    if duplicate_rows:
        warnings.append(f"Se detectaron {duplicate_rows} fechas duplicadas; se conservó la última fila de cada fecha.")
    df = df.drop_duplicates(subset=["Date"], keep="last")

    missing_values = {c: int(df[c].isna().sum()) for c in df.columns}
    critical = ["Date", "Open", "High", "Low", "Close"]
    dropped_critical = int(df[critical].isna().any(axis=1).sum())
    if dropped_critical:
        warnings.append(f"Se descartaron {dropped_critical} filas con fecha u OHLC inválido/faltante.")
    df = df.dropna(subset=critical)

    inconsistent_mask = (
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)) |
        (df["Low"] > df[["Open", "Close", "High"]].min(axis=1)) |
        (df[["Open", "High", "Low", "Close"]].le(0).any(axis=1))
    )
    inconsistent_ohlc_rows = int(inconsistent_mask.sum())
    if inconsistent_ohlc_rows:
        warnings.append(f"Se descartaron {inconsistent_ohlc_rows} filas con OHLC internamente inconsistente o no positivo.")
        df = df.loc[~inconsistent_mask]

    if "Volume" in df:
        neg_vol = int((df["Volume"] < 0).sum())
        if neg_vol:
            warnings.append(f"Se marcaron como faltantes {neg_vol} volúmenes negativos.")
            df.loc[df["Volume"] < 0, "Volume"] = np.nan
    else:
        warnings.append("El archivo no contiene volumen: la aplicación funcionará, pero limitará el análisis de confirmación por volumen.")

    df = df.sort_values("Date").reset_index(drop=True)
    report = ValidationReport(
        column_map=mapping,
        missing_required=missing_required,
        duplicate_rows=duplicate_rows,
        missing_values=missing_values,
        inconsistent_ohlc_rows=inconsistent_ohlc_rows,
        invalid_numeric_rows=invalid_numeric_rows,
        invalid_date_rows=invalid_date_rows,
        rows_input=rows_input,
        rows_output=len(df),
        warnings=warnings,
    )
    return df, report
