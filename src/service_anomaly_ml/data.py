"""Load a NAB CSV and annotation windows without modifying original data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_KEY = "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"


def default_paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "data" / "raw" / DATASET_KEY,
        root / "data" / "labels" / "combined_windows.json",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_series(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != ["timestamp", "value"]:
        raise ValueError("O dataset precisa conter somente 'timestamp' e 'value', nessa ordem.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise", utc=False)
    df["value"] = pd.to_numeric(df["value"], errors="raise")
    if df.empty or df.isna().any().any():
        raise ValueError("Dataset vazio ou contendo valores ausentes.")
    if not np.isfinite(df["value"].to_numpy(dtype=float)).all():
        raise ValueError("Valores infinitos não são suportados.")
    if not df["timestamp"].is_monotonic_increasing or df["timestamp"].duplicated().any():
        raise ValueError("Timestamps devem estar em ordem crescente, sem duplicações.")
    if df["timestamp"].dt.tz is not None:
        raise ValueError("Esta versão espera timestamps locais sem timezone (formato NAB).")
    return df.reset_index(drop=True)


def load_windows(path: Path, key: str = DATASET_KEY) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not path.is_file():
        raise FileNotFoundError(f"Anotações NAB não encontradas: {path}")
    with path.open(encoding="utf-8") as handle:
        content = json.load(handle)
    if key not in content:
        raise ValueError(f"Série temporal ausente de combined_windows.json: {key}")
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for pair in content[key]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Janela NAB inválida")
        start, end = pd.Timestamp(pair[0]), pd.Timestamp(pair[1])
        if start > end or start.tz is not None or end.tz is not None:
            raise ValueError("Janela NAB com limites inválidos")
        windows.append((start, end))
    windows.sort(key=lambda w: w[0])
    for previous, next_window in zip(windows, windows[1:]):
        if next_window[0] <= previous[1]:
            raise ValueError("Janelas NAB sobrepostas, esperava-se intervalos disjuntos")
    return windows
