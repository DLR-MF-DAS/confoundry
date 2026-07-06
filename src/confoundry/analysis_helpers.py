"""Reusable helpers for Confoundry analysis commands."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import click
import duckdb
import numpy as np
import pandas as pd

from confoundry.per_pixel_graph_discovery import quote_identifier


def ensure_identifier(identifier: str) -> str:
    """Validate and quote a DuckDB table or column identifier."""
    return quote_identifier(identifier)


def write_dataframe_table(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    """Create or replace a DuckDB table from a pandas data frame."""
    table_sql = ensure_identifier(table_name)
    con.register("_analysis_df", df)
    try:
        con.execute(
            f"CREATE OR REPLACE TABLE {table_sql} "
            "AS SELECT * FROM _analysis_df"
        )
    finally:
        con.unregister("_analysis_df")


def read_duckdb_table(db_path: Path, table_name: str) -> pd.DataFrame:
    """Read one DuckDB table, raising a Click error when it is absent."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if table_name not in tables:
            raise click.ClickException(
                f"{table_name!r} not found in {db_path}. "
                f"Available tables: {sorted(tables)}"
            )
        return con.execute(f"SELECT * FROM {ensure_identifier(table_name)}").fetchdf()
    finally:
        con.close()


def require_files(paths: Iterable[Path]) -> None:
    """Raise a user-facing error when any required input file is absent."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise click.ClickException(
            "Required input files are missing:\n" + formatted
        )


def safe_filename(value: str) -> str:
    """Convert arbitrary text to a conservative filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


def safe_float(value: Any) -> float:
    """Return a finite float or NaN for invalid/non-finite values."""
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def display_label(value: Any) -> str:
    """Format code-style variable names for figure text."""
    text = str(value).strip()
    if not text:
        return text
    text = text.replace("_", " ").replace("-", " ")
    acronym_tokens = {
        "ci": "CI",
        "db": "DB",
        "lst": "LST",
        "ndvi": "NDVI",
        "sd": "SD",
        "spei": "SPEI",
        "vpd": "VPD",
    }
    unit_tokens = {
        "cm": "cm",
        "m": "m",
        "mm": "mm",
    }
    word_tokens = {
        "abs": "Absolute",
        "boot": "Bootstrap",
        "corr": "Correlation",
        "gt": "Greater Than",
        "lt": "Less Than",
        "prob": "Probability",
    }
    lowercase_tokens = {
        "and",
        "as",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "to",
        "with",
    }
    words: list[str] = []
    for idx, raw_word in enumerate(text.split()):
        word = raw_word.strip()
        lower = word.lower()
        if lower in acronym_tokens:
            words.append(acronym_tokens[lower])
        elif lower in unit_tokens:
            words.append(unit_tokens[lower])
        elif lower in word_tokens:
            words.extend(word_tokens[lower].split())
        elif idx > 0 and lower in lowercase_tokens:
            words.append(lower)
        elif word.replace(".", "", 1).isdigit():
            words.append(word)
        else:
            words.append(lower.capitalize())
    return " ".join(words)
