from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .config import RAW_DIR, FILES


class DataFileNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class LoadResult:
    name: str
    path: Path
    df: pd.DataFrame


def load_parquet(name: str, columns: Optional[list[str]] = None) -> LoadResult:
    """
    Carga un parquet desde data/raw por nombre lógico (ej: 'bureau').
    """
    if name not in FILES:
        raise KeyError(f"Nombre '{name}' no está en config.FILES. Opciones: {list(FILES.keys())}")

    path = RAW_DIR / FILES[name]
    if not path.exists():
        raise DataFileNotFoundError(f"No existe el archivo: {path}")

    df = pd.read_parquet(path, columns=columns)
    return LoadResult(name=name, path=path, df=df)


def require_columns(df: pd.DataFrame, required: Iterable[str], df_name: str = "df") -> None:
    """
    Lanza error si faltan columnas obligatorias.
    """
    required = list(required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{df_name}] faltan columnas requeridas: {missing}")


def report_basic(df: pd.DataFrame, df_name: str = "df", max_cols: int = 10) -> str:
    """
    Resumen rápido para consola.
    """
    cols_preview = list(df.columns[:max_cols])
    return (
        f"{df_name}: shape={df.shape}, cols_preview={cols_preview}, "
        f"nulls_total={int(df.isna().sum().sum())}"
    )