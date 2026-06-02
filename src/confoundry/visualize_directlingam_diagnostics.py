"""Visualize pixel-wise DirectLiNGAM diagnostics stored in DuckDB.

This script reads the ``pixel_graph_diagnostics`` table produced by
``graph_discovery_directlingam_duckdb_diagnostics.py`` and creates an
understandable diagnostics report consisting of:

* distribution plots for global per-pixel metrics,
* spatial heatmaps when ``row`` and ``col`` are present,
* aggregated tables/plots for JSON columns such as top residual-correlation
  pairs, top bootstrap edges, bidirectional instability, lag-1 residual
  autocorrelation variables, and residual moment diagnostics,
* a compact HTML report linking all generated figures and CSV summaries.

The script intentionally avoids refitting causal models. It only visualizes the
statistics already written during graph discovery.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import click
import duckdb
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class MetricSpec:
    """Description of a scalar diagnostics metric."""

    column: str
    title: str
    explanation: str
    transform_name: str | None = None
    transform: Callable[[pd.Series], pd.Series] | None = None


def neg_log10(series: pd.Series) -> pd.Series:
    """Transform p-values into ``-log10(p)`` while avoiding infinities."""
    numeric = pd.to_numeric(series, errors="coerce")
    clipped = numeric.clip(lower=1e-300, upper=1.0)
    return -np.log10(clipped)


def log10_positive(series: pd.Series) -> pd.Series:
    """Transform positive values into ``log10(value)``."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric > 0)
    return np.log10(numeric)


METRICS: list[MetricSpec] = [
    MetricSpec(
        "directlingam_assumption_warning",
        "DirectLiNGAM assumption warning",
        "Fraction of pixels flagged by the cheap warning rule: high residual correlation, high residual lag-1 autocorrelation, or near-constant variables.",
    ),
    MetricSpec(
        "residual_max_abs_corr",
        "Maximum absolute residual correlation",
        "Large values suggest remaining dependence between estimated errors. This is a warning for hidden confounding, nonlinearity, missing lags, measurement artifacts, or other misspecification.",
    ),
    MetricSpec(
        "residual_median_abs_corr",
        "Median absolute residual correlation",
        "Typical pairwise residual dependence within a pixel/window.",
    ),
    MetricSpec(
        "residual_corr_pairs_ge_threshold",
        "Residual-correlation pairs above threshold",
        "Number of residual variable pairs whose absolute correlation exceeds the configured residual-correlation threshold.",
    ),
    MetricSpec(
        "residual_jb_min_p",
        "Minimum residual Jarque-Bera p-value",
        "Small p-values indicate at least one clearly non-Gaussian residual, which is useful for DirectLiNGAM identifiability. Raw p-values are shown in the summary; the plot uses -log10(p).",
        transform_name="-log10",
        transform=neg_log10,
    ),
    MetricSpec(
        "residual_jb_median_p",
        "Median residual Jarque-Bera p-value",
        "Median residual normality p-value across variables. The plot uses -log10(p).",
        transform_name="-log10",
        transform=neg_log10,
    ),
    MetricSpec(
        "residual_nongaussian_fraction",
        "Fraction of non-Gaussian residuals",
        "Fraction of variables whose residual Jarque-Bera p-value was below the configured diagnostic alpha.",
    ),
    MetricSpec(
        "residual_max_abs_skew",
        "Maximum absolute residual skew",
        "Largest absolute residual skewness across variables.",
    ),
    MetricSpec(
        "residual_max_abs_excess_kurtosis",
        "Maximum absolute residual excess kurtosis",
        "Largest absolute residual excess kurtosis across variables.",
    ),
    MetricSpec(
        "residual_lag1_max_median_abs_autocorr",
        "Maximum median absolute residual lag-1 autocorrelation",
        "Large values suggest remaining temporal dependence, which weakens the i.i.d. row assumption for ordinary DirectLiNGAM.",
    ),
    MetricSpec(
        "residual_lag1_median_abs_autocorr",
        "Median residual lag-1 autocorrelation",
        "Typical lag-1 residual autocorrelation across variables.",
    ),
    MetricSpec(
        "residual_lag1_variables_ge_threshold",
        "Variables above lag-1 autocorrelation threshold",
        "Number of variables whose median absolute residual lag-1 autocorrelation exceeds the configured threshold.",
    ),
    MetricSpec(
        "bootstrap_probability_entropy_mean",
        "Mean bootstrap edge-probability entropy",
        "Higher entropy means bootstrap probabilities are more ambiguous; probabilities near 0.5 contribute most.",
    ),
    MetricSpec(
        "bootstrap_edges_near_threshold",
        "Bootstrap edges near threshold",
        "Number of edges whose bootstrap probability is close to the configured consensus threshold; these are fragile decisions.",
    ),
    MetricSpec(
        "bootstrap_bidirectional_instability_max",
        "Maximum bidirectional bootstrap instability",
        "Largest value of min(P(A→B), P(B→A)) across variable pairs. Large values mean bootstrap samples often disagree about direction.",
    ),
    MetricSpec(
        "bootstrap_probability_mean",
        "Mean bootstrap edge probability",
        "Average bootstrap probability across all possible directed off-diagonal edges.",
    ),
    MetricSpec(
        "bootstrap_probability_max",
        "Maximum bootstrap edge probability",
        "Strongest edge support in each pixel/window.",
    ),
    MetricSpec(
        "raw_edge_count",
        "Raw edge count",
        "Number of raw DirectLiNGAM coefficients above the effect-size threshold.",
    ),
    MetricSpec(
        "consensus_edge_count",
        "Consensus edge count",
        "Number of edges retained after both bootstrap-probability and effect-size thresholding.",
    ),
    MetricSpec(
        "bootstrap_edges_ge_min_prob",
        "Bootstrap-supported edge count",
        "Number of directed edges whose bootstrap probability exceeds the configured minimum probability.",
    ),
    MetricSpec(
        "sample_to_variable_ratio",
        "Sample-to-variable ratio",
        "Number of complete observations divided by number of variables for each pixel/window.",
    ),
    MetricSpec(
        "condition_number",
        "Design matrix condition number",
        "Large values indicate an ill-conditioned data matrix. The plot uses log10(condition number).",
        transform_name="log10",
        transform=log10_positive,
    ),
    MetricSpec(
        "x_max_abs_corr",
        "Maximum absolute input correlation",
        "Largest absolute correlation between input variables before fitting.",
    ),
    MetricSpec(
        "x_median_abs_corr",
        "Median absolute input correlation",
        "Typical pairwise input-variable correlation before fitting.",
    ),
    MetricSpec(
        "near_constant_variable_count",
        "Near-constant variable count",
        "Number of variables with essentially zero variance in the pixel/window.",
    ),
]

HEATMAP_COLUMNS = [
    "directlingam_assumption_warning",
    "residual_max_abs_corr",
    "residual_nongaussian_fraction",
    "residual_lag1_max_median_abs_autocorr",
    "bootstrap_probability_entropy_mean",
    "bootstrap_edges_near_threshold",
    "bootstrap_bidirectional_instability_max",
    "consensus_edge_count",
    "sample_to_variable_ratio",
]

JSON_AGGREGATION_COLUMNS = [
    "residual_corr_top_pairs_json",
    "bootstrap_top_edges_json",
    "bootstrap_bidirectional_top_pairs_json",
    "residual_lag1_top_variables_json",
    "residual_moments_json",
]


def quote_identifier(identifier: str) -> str:
    """Return a safely quoted DuckDB identifier for simple names."""
    if not identifier.replace("_", "").isalnum() or identifier[0].isdigit():
        raise click.BadParameter(
            f"Invalid DuckDB identifier {identifier!r}; use letters, numbers, and underscores."
        )
    return f'"{identifier}"'


def load_table(db_path: Path, table_name: str, metadata_table: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load diagnostics and optional metadata from DuckDB."""
    if not db_path.exists():
        raise click.ClickException(f"Diagnostics database not found: {db_path}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").df()["name"])
        if table_name not in tables:
            raise click.ClickException(
                f"Table {table_name!r} not found in {db_path}. Available tables: {sorted(tables)}"
            )
        df = con.execute(f"SELECT * FROM {quote_identifier(table_name)}").fetchdf()
        metadata = None
        if metadata_table in tables:
            metadata = con.execute(f"SELECT * FROM {quote_identifier(metadata_table)}").fetchdf()
    finally:
        con.close()

    if df.empty:
        raise click.ClickException(f"Diagnostics table {table_name!r} is empty.")
    return df, metadata


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric Series for a diagnostics column."""
    if column not in df.columns:
        return pd.Series(dtype=float)
    series = df[column]
    if series.dtype == bool:
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def clean_values(series: pd.Series) -> np.ndarray:
    """Return finite values from a numeric Series."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def save_histogram(values: np.ndarray, title: str, xlabel: str, output_path: Path) -> None:
    """Save a histogram for one metric."""
    if len(values) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(60, max(10, int(np.sqrt(len(values))))))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel/window count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
    """Save a spatial heatmap for a metric when row/col are available."""
    if not {"row", "col", column}.issubset(df.columns):
        return

    work = df[["row", "col", column]].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["row", "col", column])
    if work.empty:
        return

    grid = work.pivot_table(index="row", columns="col", values=column, aggfunc="mean")
    grid = grid.sort_index(ascending=True).sort_index(axis=1, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid.to_numpy(dtype=float), origin="upper", aspect="auto")
    fig.colorbar(im, ax=ax, label=column)
    ax.set_title(title)
    #ax.set_xlabel("col")
    #ax.set_ylabel("row")

    # Keep tick labels sparse so large grids remain readable.
    if len(grid.columns) > 0:
        xtick_positions = np.linspace(0, len(grid.columns) - 1, min(8, len(grid.columns)), dtype=int)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([str(grid.columns[i]) for i in xtick_positions])
    if len(grid.index) > 0:
        ytick_positions = np.linspace(0, len(grid.index) - 1, min(8, len(grid.index)), dtype=int)
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([str(grid.index[i]) for i in ytick_positions])

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_barh(table: pd.DataFrame, label_col: str, value_col: str, title: str, output_path: Path) -> None:
    """Save a horizontal bar plot from a summary table."""
    if table.empty or label_col not in table.columns or value_col not in table.columns:
        return
    work = table[[label_col, value_col]].dropna().head(25).iloc[::-1]
    if work.empty:
        return
    fig_height = max(4.0, 0.35 * len(work) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(work[label_col].astype(str), pd.to_numeric(work[value_col], errors="coerce"))
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact statistical summary of scalar diagnostics columns."""
    rows: list[dict[str, Any]] = []
    for spec in METRICS:
        if spec.column not in df.columns:
            continue
        series = numeric_series(df, spec.column)
        values = clean_values(series)
        rows.append(
            {
                "metric": spec.column,
                "title": spec.title,
                "n": int(len(values)),
                "missing_fraction": float(series.isna().mean()),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "std": float(np.std(values)) if len(values) else np.nan,
                "min": float(np.min(values)) if len(values) else np.nan,
                "q05": float(np.quantile(values, 0.05)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "q95": float(np.quantile(values, 0.95)) if len(values) else np.nan,
                "max": float(np.max(values)) if len(values) else np.nan,
                "explanation": spec.explanation,
            }
        )
    return pd.DataFrame(rows)


def iter_json_list(value: Any) -> Iterable[dict[str, Any]]:
    """Yield dictionaries from a JSON-list cell."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return
    if isinstance(value, list):
        records = value
    else:
        try:
            records = json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            return
    if not isinstance(records, list):
        return
    for record in records:
        if isinstance(record, dict):
            yield record


def aggregate_residual_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate strongest residual-correlation pairs across pixels."""
    if "residual_corr_top_pairs_json" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cell in df["residual_corr_top_pairs_json"]:
        for record in iter_json_list(cell):
            var1 = record.get("var1")
            var2 = record.get("var2")
            if var1 is None or var2 is None:
                continue
            rows.append(
                {
                    "pair": f"{var1} ↔ {var2}",
                    "abs_value": pd.to_numeric(record.get("abs_value"), errors="coerce"),
                    "value": pd.to_numeric(record.get("value"), errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame()
    work = pd.DataFrame(rows)
    return (
        work.groupby("pair", as_index=False)
        .agg(
            n_pixels=("pair", "size"),
            median_abs_residual_corr=("abs_value", "median"),
            mean_abs_residual_corr=("abs_value", "mean"),
            max_abs_residual_corr=("abs_value", "max"),
        )
        .sort_values(["n_pixels", "median_abs_residual_corr"], ascending=False)
    )


def aggregate_bootstrap_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate top bootstrap-supported directed edges across pixels."""
    if "bootstrap_top_edges_json" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cell in df["bootstrap_top_edges_json"]:
        for record in iter_json_list(cell):
            parent = record.get("parent")
            child = record.get("child")
            if parent is None or child is None:
                continue
            rows.append(
                {
                    "edge": f"{parent} → {child}",
                    "probability": pd.to_numeric(record.get("probability"), errors="coerce"),
                    "abs_coefficient": pd.to_numeric(record.get("abs_coefficient"), errors="coerce"),
                    "in_consensus": bool(record.get("in_consensus", False)),
                }
            )
    if not rows:
        return pd.DataFrame()
    work = pd.DataFrame(rows)
    return (
        work.groupby("edge", as_index=False)
        .agg(
            n_top_pixels=("edge", "size"),
            consensus_count=("in_consensus", "sum"),
            median_probability=("probability", "median"),
            mean_probability=("probability", "mean"),
            median_abs_coefficient=("abs_coefficient", "median"),
        )
        .sort_values(["n_top_pixels", "median_probability"], ascending=False)
    )


def aggregate_bidirectional_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bidirectional bootstrap-instability pairs across pixels."""
    if "bootstrap_bidirectional_top_pairs_json" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cell in df["bootstrap_bidirectional_top_pairs_json"]:
        for record in iter_json_list(cell):
            var1 = record.get("var1")
            var2 = record.get("var2")
            if var1 is None or var2 is None:
                continue
            rows.append(
                {
                    "pair": f"{var1} ↔ {var2}",
                    "bidirectional_instability": pd.to_numeric(
                        record.get("bidirectional_instability"), errors="coerce"
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    work = pd.DataFrame(rows)
    return (
        work.groupby("pair", as_index=False)
        .agg(
            n_pixels=("pair", "size"),
            median_bidirectional_instability=("bidirectional_instability", "median"),
            max_bidirectional_instability=("bidirectional_instability", "max"),
        )
        .sort_values(["n_pixels", "median_bidirectional_instability"], ascending=False)
    )


def aggregate_lag1_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate residual lag-1 autocorrelation variables across pixels."""
    if "residual_lag1_top_variables_json" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cell in df["residual_lag1_top_variables_json"]:
        for record in iter_json_list(cell):
            variable = record.get("variable")
            if variable is None:
                continue
            rows.append(
                {
                    "variable": str(variable),
                    "median_abs_lag1_autocorr": pd.to_numeric(
                        record.get("median_abs_lag1_autocorr"), errors="coerce"
                    ),
                    "max_abs_lag1_autocorr": pd.to_numeric(
                        record.get("max_abs_lag1_autocorr"), errors="coerce"
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    work = pd.DataFrame(rows)
    return (
        work.groupby("variable", as_index=False)
        .agg(
            n_top_pixels=("variable", "size"),
            median_abs_lag1_autocorr=("median_abs_lag1_autocorr", "median"),
            max_abs_lag1_autocorr=("max_abs_lag1_autocorr", "max"),
        )
        .sort_values(["n_top_pixels", "median_abs_lag1_autocorr"], ascending=False)
    )


def aggregate_residual_moments(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate residual non-Gaussianity diagnostics by variable."""
    if "residual_moments_json" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cell in df["residual_moments_json"]:
        for record in iter_json_list(cell):
            variable = record.get("variable")
            if variable is None:
                continue
            rows.append(
                {
                    "variable": str(variable),
                    "abs_skew": abs(pd.to_numeric(record.get("skew"), errors="coerce")),
                    "abs_excess_kurtosis": abs(
                        pd.to_numeric(record.get("excess_kurtosis"), errors="coerce")
                    ),
                    "jarque_bera_p": pd.to_numeric(record.get("jarque_bera_p"), errors="coerce"),
                    "nongaussian_at_alpha": bool(record.get("nongaussian_at_alpha", False)),
                }
            )
    if not rows:
        return pd.DataFrame()
    work = pd.DataFrame(rows)
    return (
        work.groupby("variable", as_index=False)
        .agg(
            n_pixels=("variable", "size"),
            nongaussian_fraction=("nongaussian_at_alpha", "mean"),
            median_abs_skew=("abs_skew", "median"),
            median_abs_excess_kurtosis=("abs_excess_kurtosis", "median"),
            median_jb_p=("jarque_bera_p", "median"),
        )
        .sort_values(["nongaussian_fraction", "median_abs_excess_kurtosis"], ascending=False)
    )


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    """Render a small dataframe as an HTML table."""
    if df is None or df.empty:
        return "<p>No data available.</p>"
    return df.head(max_rows).to_html(index=False, border=0, classes="summary-table")


def create_report(
    df: pd.DataFrame,
    metadata: pd.DataFrame | None,
    metric_summary: pd.DataFrame,
    aggregate_tables: dict[str, pd.DataFrame],
    figure_paths: list[Path],
    table_paths: list[Path],
    output_dir: Path,
    report_path: Path,
) -> None:
    """Create a compact HTML report linking plots and tables."""
    n_pixels = len(df)
    warning_fraction = None
    if "directlingam_assumption_warning" in df.columns:
        warning_fraction = float(numeric_series(df, "directlingam_assumption_warning").mean())

    headline_rows = [
        {"quantity": "pixels/windows", "value": f"{n_pixels:,}"},
    ]
    if warning_fraction is not None and np.isfinite(warning_fraction):
        headline_rows.append({"quantity": "assumption-warning fraction", "value": f"{warning_fraction:.1%}"})
    for col in [
        "residual_max_abs_corr",
        "residual_lag1_max_median_abs_autocorr",
        "bootstrap_edges_near_threshold",
        "consensus_edge_count",
    ]:
        if col in df.columns:
            values = clean_values(numeric_series(df, col))
            if len(values):
                headline_rows.append({"quantity": f"median {col}", "value": f"{np.median(values):.4g}"})
                headline_rows.append({"quantity": f"95th percentile {col}", "value": f"{np.quantile(values, 0.95):.4g}"})

    metadata_html = "<p>No run metadata table found.</p>"
    if metadata is not None and not metadata.empty:
        metadata_view = metadata.tail(1).T.reset_index()
        metadata_view.columns = ["field", "value"]
        metadata_html = dataframe_to_html_table(metadata_view, max_rows=80)

    figure_items = "\n".join(
        f'<figure><img src="{html.escape(str(path.relative_to(output_dir)))}" alt="{html.escape(path.stem)}"><figcaption>{html.escape(path.stem.replace("_", " "))}</figcaption></figure>'
        for path in figure_paths
    )

    table_links = "\n".join(
        f'<li><a href="{html.escape(str(path.relative_to(output_dir)))}">{html.escape(path.name)}</a></li>'
        for path in table_paths
    )

    aggregate_sections = []
    for title, table in aggregate_tables.items():
        aggregate_sections.append(f"<h3>{html.escape(title)}</h3>{dataframe_to_html_table(table)}")

    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DirectLiNGAM diagnostics report</title>
<style>
body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; line-height: 1.45; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
figure {{ margin: 1.5rem 0; }}
figcaption {{ color: #555; font-size: 0.9rem; }}
table.summary-table {{ border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
table.summary-table th, table.summary-table td {{ border-bottom: 1px solid #ddd; padding: 0.35rem 0.6rem; text-align: left; vertical-align: top; }}
code {{ background: #f5f5f5; padding: 0.1rem 0.25rem; }}
</style>
</head>
<body>
<h1>DirectLiNGAM diagnostics report</h1>
<p>This report visualizes diagnostics already computed during pixel-wise graph discovery. It does not refit DirectLiNGAM or run additional causal tests.</p>

<h2>How to read this</h2>
<p><strong>Residual dependence</strong> is the main cheap warning signal for hidden confounding or misspecification. It is not a proof of a hidden confounder, because nonlinearity, missing temporal lags, measurement artifacts, cycles, or selection effects can produce similar patterns.</p>
<p><strong>Residual non-Gaussianity</strong> supports the DirectLiNGAM identifiability assumption. <strong>Residual lag-1 autocorrelation</strong> warns that rows may not be i.i.d. <strong>Bootstrap entropy and near-threshold edge counts</strong> summarize graph stability.</p>

<h2>Headline summary</h2>
{dataframe_to_html_table(pd.DataFrame(headline_rows), max_rows=80)}

<h2>Run metadata</h2>
{metadata_html}

<h2>Metric summary</h2>
{dataframe_to_html_table(metric_summary[["metric", "median", "q95", "max", "explanation"]] if not metric_summary.empty else metric_summary, max_rows=80)}

<h2>Aggregated JSON diagnostics</h2>
{''.join(aggregate_sections)}

<h2>Generated CSV tables</h2>
<ul>{table_links}</ul>

<h2>Plots</h2>
{figure_items}
</body>
</html>
"""
    report_path.write_text(content, encoding="utf-8")


@click.command()
@click.option(
    "--diagnostics-db",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="DuckDB file containing the pixel_graph_diagnostics table.",
)
@click.option("--table", default="pixel_graph_diagnostics", show_default=True)
@click.option("--metadata-table", default="graph_discovery_run_metadata", show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory. Defaults to <diagnostics-db-stem>_diagnostics_report next to the database.",
)
@click.option("--top-n", default=25, show_default=True, type=int, help="Number of aggregate items to show in bar plots.")
def visualize_diagnostics(
    diagnostics_db: Path,
    table: str,
    metadata_table: str,
    output_dir: Path | None,
    top_n: int,
) -> None:
    """Create plots and an HTML report for DirectLiNGAM diagnostics."""
    if output_dir is None:
        output_dir = diagnostics_db.with_name(f"{diagnostics_db.stem}_diagnostics_report")
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df, metadata = load_table(diagnostics_db, table, metadata_table)

    # Normalize boolean warning columns for summaries/plots.
    if "directlingam_assumption_warning" in df.columns:
        df["directlingam_assumption_warning"] = df["directlingam_assumption_warning"].astype(float)

    figure_paths: list[Path] = []
    table_paths: list[Path] = []

    metric_summary = summarize_metrics(df)
    metric_summary_path = tables_dir / "metric_summary.csv"
    metric_summary.to_csv(metric_summary_path, index=False)
    table_paths.append(metric_summary_path)

    # Distribution plots.
    for spec in METRICS:
        if spec.column not in df.columns:
            continue
        series = numeric_series(df, spec.column)
        plot_series = spec.transform(series) if spec.transform is not None else series
        values = clean_values(plot_series)
        if len(values) == 0:
            continue
        suffix = f"_{spec.transform_name}" if spec.transform_name else ""
        path = figures_dir / f"hist_{spec.column}{suffix}.png"
        xlabel = f"{spec.transform_name}({spec.column})" if spec.transform_name else spec.column
        save_histogram(values, spec.title, xlabel, path)
        figure_paths.append(path)

    # Spatial heatmaps.
    if {"row", "col"}.issubset(df.columns):
        for column in HEATMAP_COLUMNS:
            if column not in df.columns:
                continue
            path = figures_dir / f"heatmap_{column}.png"
            save_heatmap(df, column, f"Spatial heatmap: {column}", path)
            if path.exists():
                figure_paths.append(path)

    aggregate_tables: dict[str, pd.DataFrame] = {}

    aggregators = {
        "Residual-correlation pairs": (aggregate_residual_pairs, "pair", "n_pixels"),
        "Bootstrap-supported edges": (aggregate_bootstrap_edges, "edge", "n_top_pixels"),
        "Bidirectional bootstrap-instability pairs": (
            aggregate_bidirectional_pairs,
            "pair",
            "n_pixels",
        ),
        "Residual lag-1 autocorrelation variables": (
            aggregate_lag1_variables,
            "variable",
            "n_top_pixels",
        ),
        "Residual moment diagnostics by variable": (
            aggregate_residual_moments,
            "variable",
            "nongaussian_fraction",
        ),
    }

    for title, (func, label_col, value_col) in aggregators.items():
        table_df = func(df)
        aggregate_tables[title] = table_df
        if table_df.empty:
            continue
        safe_name = title.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        csv_path = tables_dir / f"{safe_name}.csv"
        table_df.to_csv(csv_path, index=False)
        table_paths.append(csv_path)

        plot_path = figures_dir / f"bar_{safe_name}.png"
        save_barh(table_df.head(top_n), label_col, value_col, title, plot_path)
        if plot_path.exists():
            figure_paths.append(plot_path)

    report_path = output_dir / "diagnostics_report.html"
    create_report(
        df=df,
        metadata=metadata,
        metric_summary=metric_summary,
        aggregate_tables=aggregate_tables,
        figure_paths=figure_paths,
        table_paths=table_paths,
        output_dir=output_dir,
        report_path=report_path,
    )

    click.echo(f"Wrote diagnostics report: {report_path}")
    click.echo(f"Wrote figures: {figures_dir}")
    click.echo(f"Wrote summary tables: {tables_dir}")


if __name__ == "__main__":
    visualize_diagnostics()
