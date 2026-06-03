"""
Shared data-loading helpers for the DAP391m attribution notebooks.

The notebooks intentionally use the already-cleaned CSV files from
Social-Media-Campaign-Attribution-for-E-commerce-Sales-main.
"""
from pathlib import Path

import pandas as pd


NOTEBOOK_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = NOTEBOOK_DIR.parent
PROJECT_ROOT = ANALYSIS_DIR.parent
OUTPUT_DIR = ANALYSIS_DIR / "outputs"
REQUIRED_DATA_FILES = ("data_touchpoints.csv", "data_journeys.csv", "data_encoded.csv")


def _find_data_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "data_preparation" / "processed",
        PROJECT_ROOT,
        PROJECT_ROOT / "Social-Media-Campaign-Attribution-for-E-commerce-Sales-main",
        PROJECT_ROOT.parent / "Social-Media-Campaign-Attribution-for-E-commerce-Sales-main",
    ]
    for candidate in candidates:
        if all((candidate / filename).exists() for filename in REQUIRED_DATA_FILES):
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find cleaned data files. Checked: {checked}")


DATA_DIR = _find_data_dir()
RANDOM_SEED = 42


def _read_clean_csv(filename: str, path: Path | str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DATA_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_touchpoints(path: Path | str | None = None) -> pd.DataFrame:
    """Load cleaned touchpoint-level data with precomputed attribution columns."""
    return _read_clean_csv("data_touchpoints.csv", path)


def load_journeys(path: Path | str | None = None) -> pd.DataFrame:
    """Load cleaned user-level journey data."""
    return _read_clean_csv("data_journeys.csv", path)


def load_encoded(path: Path | str | None = None) -> pd.DataFrame:
    """Load cleaned channel one-hot encoded data for regression."""
    return _read_clean_csv("data_encoded.csv", path)
