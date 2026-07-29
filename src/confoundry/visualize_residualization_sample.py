"""Create a paper-ready before/after residualization example.

The command reads the residualized ARD table produced by
``residualize_timeseries.py`` and selects one pixel with sufficient complete
observations. It visualizes, for each requested variable:

* the original observations and fitted seasonal/trend expectation,
* the residual anomaly passed to causal graph discovery, and
* the monthly climatology before and after residualization.

No model is refit here: the figure uses the expectation and residual columns
already stored by the residualization command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import yaml


SEASONAL_COLUMNS = {"month_sin", "month_cos"}


def read_config(config_path: Path) -> dict[str, Any]:
    """Read and minimally validate an experiment YAML file."""
    with config_path.open("r", encoding="utf-8") as fd:
        config = yaml.safe_load(fd) or {}
    if not isinstance(config, dict):
        raise click.ClickException("Experiment YAML must contain a mapping.")
    if "name" not in config:
        raise click.ClickException("Config must contain 'name'.")
    return config


def resolve_path(base_dir: Path, value: str | Path | None, default: Path) -> Path:
    """Resolve a possibly relative path against the experiment directory."""
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    try:
        cwd_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return base_dir / path
    return cwd_path


def quote_identifier(value: str) -> str:
    """Quote a DuckDB identifier."""
    return '"' + str(value).replace('"', '""') + '"'


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """Return the columns of ``table`` after checking that it exists."""
    tables = set(con.sql("SHOW TABLES").df()["name"])
    if table not in tables:
        raise click.ClickException(
            f"{table!r} not found. Available tables: {sorted(tables)}"
        )
    description = con.execute(
        f"DESCRIBE {quote_identifier(table)}"
    ).fetchdf()
    return description["column_name"].astype(str).tolist()


def infer_variables(
    config: Mapping[str, Any],
    available_columns: Sequence[str],
    requested_variables: Sequence[str],
    suffix: str,
    expected_suffix: str,
) -> list[str]:
    """Resolve original variable names represented in the residualized table."""
    available = set(available_columns)
    residualization = config.get("residualization") or {}
    configured = [str(value) for value in residualization.get("variables", [])]

    if not configured:
        configured = [
            column[: -len(suffix)]
            for column in available_columns
            if suffix
            and column.endswith(suffix)
            and f"{column[: -len(suffix)]}{expected_suffix}" in available
            and column[: -len(suffix)] in available
        ]

    if not configured and "columns" in config:
        configured = [
            str(spec["name"])
            for spec in config["columns"]
            if str(spec.get("name")) not in SEASONAL_COLUMNS
            and str(spec.get("name")) in available
            and f"{spec['name']}{suffix}" in available
            and f"{spec['name']}{expected_suffix}" in available
        ]

    variables = (
        list(map(str, requested_variables))
        if requested_variables
        else configured
    )
    if not variables:
        raise click.ClickException(
            "Could not infer residualized variables. Pass one or more --variable "
            "options or use the generated residualized YAML config."
        )

    missing: list[str] = []
    for variable in variables:
        required = {
            variable,
            f"{variable}{suffix}",
            f"{variable}{expected_suffix}",
        }
        if not required.issubset(available):
            missing.append(variable)
    if missing:
        raise click.ClickException(
            "Missing original/residual/expectation columns for: " + ", ".join(missing)
        )
    return variables


def complete_condition(
    variables: Sequence[str],
    suffix: str,
    expected_suffix: str,
) -> str:
    """Return a SQL condition requiring complete plotting observations."""
    required = ["row", "col", "year", "month"]
    for variable in variables:
        required.extend(
            [
                variable,
                f"{variable}{suffix}",
                f"{variable}{expected_suffix}",
            ]
        )
    return " AND ".join(
        f"{quote_identifier(column)} IS NOT NULL" for column in required
    )


def candidate_pixels(
    con: duckdb.DuckDBPyConnection,
    table: str,
    variables: Sequence[str],
    suffix: str,
    expected_suffix: str,
    min_samples: int,
) -> pd.DataFrame:
    """Summarize complete candidate pixels without loading the full table."""
    condition = complete_condition(variables, suffix, expected_suffix)
    explained_terms = []
    for variable in variables:
        original = quote_identifier(variable)
        residual = quote_identifier(f"{variable}{suffix}")
        explained_terms.append(
            f"1.0 - var_pop({residual}) / nullif(var_pop({original}), 0.0)"
        )
    mean_explained = " + ".join(f"({term})" for term in explained_terms)
    mean_explained = f"({mean_explained}) / {len(explained_terms)}"

    query = f"""
        WITH complete AS (
            SELECT *
            FROM {quote_identifier(table)}
            WHERE {condition}
        )
        SELECT
            CAST(row AS BIGINT) AS row,
            CAST(col AS BIGINT) AS col,
            count(*) AS n_complete,
            {mean_explained} AS mean_explained_fraction
        FROM complete
        GROUP BY row, col
        HAVING count(*) >= ?
        ORDER BY row, col
    """
    candidates = con.execute(query, [min_samples]).fetchdf()
    if candidates.empty:
        raise click.ClickException(
            "No pixel contains the required complete original, expected, and "
            f"residual values for at least {min_samples} observations."
        )
    return candidates


def select_pixel(
    candidates: pd.DataFrame,
    selection: str,
    seed: int,
    row: int | None,
    col: int | None,
) -> tuple[int, int, dict[str, Any]]:
    """Select an explicit, representative, random, or high-signal pixel."""
    if (row is None) != (col is None):
        raise click.ClickException("--row and --col must be supplied together.")

    if row is not None and col is not None:
        match = candidates[(candidates["row"] == row) & (candidates["col"] == col)]
        if match.empty:
            raise click.ClickException(
                f"Pixel row={row}, col={col} does not satisfy --min-samples."
            )
        selected = match.iloc[0]
        strategy = "explicit"
    elif selection == "random":
        rng = np.random.default_rng(seed)
        selected = candidates.iloc[int(rng.integers(0, len(candidates)))]
        strategy = "random"
    else:
        finite = candidates[
            np.isfinite(candidates["mean_explained_fraction"].astype(float))
        ].copy()
        if finite.empty:
            finite = candidates.copy()
            finite["mean_explained_fraction"] = 0.0

        if selection == "highest-explained":
            selected = finite.sort_values(
                ["mean_explained_fraction", "n_complete"],
                ascending=[False, False],
            ).iloc[0]
            strategy = "highest-explained"
        else:
            median_score = float(
                np.nanmedian(finite["mean_explained_fraction"].astype(float))
            )
            finite["distance_to_median"] = (
                finite["mean_explained_fraction"].astype(float) - median_score
            ).abs()
            selected = finite.sort_values(
                ["distance_to_median", "n_complete", "row", "col"],
                ascending=[True, False, True, True],
            ).iloc[0]
            strategy = "representative"

    metadata = {
        "selection_strategy": strategy,
        "candidate_pixel_count": int(len(candidates)),
        "selected_n_complete": int(selected["n_complete"]),
        "selected_mean_explained_fraction": float(
            selected["mean_explained_fraction"]
        ),
        "seed": int(seed) if strategy == "random" else None,
    }
    return int(selected["row"]), int(selected["col"]), metadata


def load_pixel_data(
    con: duckdb.DuckDBPyConnection,
    table: str,
    row: int,
    col: int,
    variables: Sequence[str],
    suffix: str,
    expected_suffix: str,
) -> pd.DataFrame:
    """Load complete plotting data for one pixel."""
    columns = ["row", "col", "year", "month"]
    for optional in ["month_sin", "month_cos"]:
        columns.append(optional)
    for variable in variables:
        columns.extend(
            [
                variable,
                f"{variable}{expected_suffix}",
                f"{variable}{suffix}",
            ]
        )

    available = set(table_columns(con, table))
    columns = [column for column in columns if column in available]
    condition = complete_condition(variables, suffix, expected_suffix)
    query = f"""
        SELECT {', '.join(quote_identifier(column) for column in columns)}
        FROM {quote_identifier(table)}
        WHERE row = ? AND col = ? AND {condition}
        ORDER BY year, month
    """
    frame = con.execute(query, [row, col]).fetchdf()
    if frame.empty:
        raise click.ClickException(
            f"No complete observations found for row={row}, col={col}."
        )
    frame["date"] = pd.to_datetime(
        {
            "year": frame["year"].astype(int),
            "month": frame["month"].astype(int),
            "day": 1,
        }
    )
    return frame


def coefficient_of_determination(observed: np.ndarray, fitted: np.ndarray) -> float:
    """Return R² for observed and fitted values."""
    observed = np.asarray(observed, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    if denominator <= 0.0:
        return np.nan
    return 1.0 - float(np.sum((observed - fitted) ** 2)) / denominator


def harmonic_statistics(values: np.ndarray, months: np.ndarray) -> tuple[float, float]:
    """Return annual harmonic R² and fitted peak-to-peak amplitude."""
    values = np.asarray(values, dtype=float)
    angle = 2.0 * np.pi * (np.asarray(months, dtype=float) - 1.0) / 12.0
    design = np.column_stack([np.ones(len(values)), np.sin(angle), np.cos(angle)])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    r_squared = coefficient_of_determination(values, fitted)
    amplitude = 2.0 * float(np.hypot(coefficients[1], coefficients[2]))
    return r_squared, amplitude


def lag1_autocorrelation(values: np.ndarray) -> float:
    """Return lag-one Pearson autocorrelation."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def summarize_variable(
    frame: pd.DataFrame,
    variable: str,
    suffix: str,
    expected_suffix: str,
) -> dict[str, Any]:
    """Compute transparent before/after diagnostics for one variable."""
    original = frame[variable].astype(float).to_numpy()
    expected = frame[f"{variable}{expected_suffix}"].astype(float).to_numpy()
    residual = frame[f"{variable}{suffix}"].astype(float).to_numpy()
    months = frame["month"].astype(int).to_numpy()

    seasonal_r2_before, seasonal_amplitude_before = harmonic_statistics(
        original, months
    )
    seasonal_r2_after, seasonal_amplitude_after = harmonic_statistics(
        residual, months
    )
    amplitude_reduction = (
        1.0 - seasonal_amplitude_after / seasonal_amplitude_before
        if seasonal_amplitude_before > 0.0
        else np.nan
    )
    return {
        "variable": variable,
        "n_observations": int(len(frame)),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
        "original_mean": float(np.mean(original)),
        "original_std": float(np.std(original, ddof=1)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=1)),
        "expected_r2": coefficient_of_determination(original, expected),
        "seasonal_r2_before": seasonal_r2_before,
        "seasonal_r2_after": seasonal_r2_after,
        "seasonal_amplitude_before": seasonal_amplitude_before,
        "seasonal_amplitude_after": seasonal_amplitude_after,
        "seasonal_amplitude_reduction": amplitude_reduction,
        "lag1_autocorrelation_before": lag1_autocorrelation(original),
        "lag1_autocorrelation_after": lag1_autocorrelation(residual),
        "identity_max_abs_error": float(
            np.max(np.abs(original - expected - residual))
        ),
    }


def standardized_monthly_climatology(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Return monthly mean and standard error after global standardization."""
    values = frame[column].astype(float)
    std = float(values.std(ddof=1))
    standardized = (values - float(values.mean())) / std if std > 0.0 else values * 0.0
    climatology = pd.DataFrame(
        {"month": frame["month"].astype(int), "value": standardized}
    ).groupby("month")["value"]
    result = climatology.agg(["mean", "count", "std"]).reset_index()
    result["sem"] = result["std"] / np.sqrt(result["count"])
    return result


def match_vertical_scale(ax_original: plt.Axes, ax_residual: plt.Axes) -> None:
    """Give original and residual panels equal y-axis spans.

    The original observations retain their absolute level, while the residual
    panel stays centered on zero. Equal spans make magnitudes visually
    comparable without forcing both panels to share inappropriate absolute
    limits.
    """
    original_bottom, original_top = ax_original.get_ylim()
    residual_bottom, residual_top = ax_residual.get_ylim()
    half_span = max(
        (original_top - original_bottom) / 2.0,
        abs(residual_bottom),
        abs(residual_top),
    )
    original_center = (original_bottom + original_top) / 2.0
    ax_original.set_ylim(original_center - half_span, original_center + half_span)
    ax_residual.set_ylim(-half_span, half_span)


def create_figure(
    frame: pd.DataFrame,
    summaries: pd.DataFrame,
    variables: Sequence[str],
    suffix: str,
    expected_suffix: str,
    row: int,
    col: int,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
) -> None:
    """Create the three-panel-per-variable residualization figure."""
    n_variables = len(variables)
    figure, axes = plt.subplots(
        n_variables,
        3,
        figsize=(15.0, 2.9 * n_variables + 1.0),
        squeeze=False,
        constrained_layout=True,
    )

    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    summary_lookup = summaries.set_index("variable")

    for index, variable in enumerate(variables):
        original_col = variable
        expected_col = f"{variable}{expected_suffix}"
        residual_col = f"{variable}{suffix}"
        summary = summary_lookup.loc[variable]

        ax_original, ax_residual, ax_cycle = axes[index]
        ax_original.plot(
            frame["date"], frame[original_col], linewidth=1.1, label="Observed"
        )
        ax_original.plot(
            frame["date"],
            frame[expected_col],
            linewidth=1.8,
            label="Seasonal + trend expectation",
        )
        ax_original.set_ylabel(variable.replace("_", " "))
        ax_original.grid(alpha=0.25)
        if index == 0:
            ax_original.set_title("Before: observed and fitted component")
            ax_original.legend(frameon=False, fontsize=8, loc="best")

        ax_residual.plot(
            frame["date"], frame[residual_col], linewidth=1.1, label="Residual"
        )
        ax_residual.axhline(0.0, linewidth=0.8, linestyle="--")
        ax_residual.grid(alpha=0.25)
        if index == 0:
            ax_residual.set_title("After: anomaly used for causal discovery")
        ax_residual.text(
            0.02,
            0.96,
            f"fitted component $R^2$ = {summary['expected_r2']:.2f}",
            transform=ax_residual.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        match_vertical_scale(ax_original, ax_residual)

        original_cycle = standardized_monthly_climatology(frame, original_col)
        residual_cycle = standardized_monthly_climatology(frame, residual_col)
        ax_cycle.errorbar(
            original_cycle["month"],
            original_cycle["mean"],
            yerr=original_cycle["sem"],
            marker="o",
            linewidth=1.4,
            capsize=2,
            label="Original",
        )
        ax_cycle.errorbar(
            residual_cycle["month"],
            residual_cycle["mean"],
            yerr=residual_cycle["sem"],
            marker="o",
            linewidth=1.4,
            capsize=2,
            label="Residual",
        )
        ax_cycle.axhline(0.0, linewidth=0.8, linestyle="--")
        ax_cycle.set_xticks(range(1, 13), month_labels)
        ax_cycle.set_ylabel("Standardized monthly mean")
        ax_cycle.grid(alpha=0.25)
        if index == 0:
            ax_cycle.set_title("Annual cycle before and after")
            ax_cycle.legend(frameon=False, fontsize=8, loc="best")
        reduction = summary["seasonal_amplitude_reduction"]
        reduction_text = (
            f"annual amplitude reduction = {reduction:.0%}"
            if np.isfinite(reduction)
            else "annual amplitude reduction = n/a"
        )
        ax_cycle.text(
            0.02,
            0.04,
            reduction_text,
            transform=ax_cycle.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )

    for axis in axes[-1]:
        axis.set_xlabel("Date" if axis is not axes[-1, 2] else "Month")

    date_start = frame["date"].min().strftime("%Y-%m")
    date_end = frame["date"].max().strftime("%Y-%m")
    figure.suptitle(
        "Residualization example from the analysis-ready data\n"
        f"pixel row={row}, col={col}; {date_start} to {date_end}",
        fontsize=14,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def write_caption(
    output_path: Path,
    row: int,
    col: int,
    variables: Sequence[str],
    selection_strategy: str,
) -> None:
    """Write a reusable paper-caption draft."""
    variable_text = ", ".join(variable.replace("_", " ") for variable in variables)
    caption = (
        "Illustration of the residualization applied before causal graph "
        f"discovery for {variable_text} at pixel row {row}, column {col}. "
        "The left panels show the original monthly observations and the fitted "
        "deterministic seasonal-plus-trend component. The middle panels show "
        "the residual anomalies supplied to the causal model. The right panels "
        "compare standardized monthly climatologies before and after "
        "residualization; the collapse of the annual cycle toward zero shows "
        "that deterministic seasonality has been removed while short-term "
        "departures are retained. The pixel was selected using the "
        f"'{selection_strategy}' strategy."
    )
    output_path.write_text(caption + "\n", encoding="utf-8")


@click.command()
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Residualized experiment YAML generated by residualize_timeseries.py. "
        "The original YAML also works with the residualizer's default outputs."
    ),
)
@click.option("--input-db", default=None, type=click.Path(path_type=Path))
@click.option("--input-table", default=None)
@click.option(
    "--variable",
    "variables",
    multiple=True,
    help="Original variable to show. Repeat to control figure rows and order.",
)
@click.option("--suffix", default=None, help="Residual column suffix.")
@click.option(
    "--expected-suffix",
    default=None,
    help="Fitted seasonal/trend expectation column suffix.",
)
@click.option("--row", default=None, type=int, help="Explicit pixel row.")
@click.option("--col", default=None, type=int, help="Explicit pixel column.")
@click.option(
    "--selection",
    type=click.Choice(
        ["representative", "random", "highest-explained"],
        case_sensitive=False,
    ),
    default="representative",
    show_default=True,
    help=(
        "Automatic pixel selection. 'representative' chooses the pixel nearest "
        "the median mean explained fraction; 'highest-explained' is useful for "
        "a deliberately illustrative example but should be reported as such."
    ),
)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--min-samples",
    default=60,
    show_default=True,
    type=click.IntRange(min=12),
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--dpi", default=300, show_default=True, type=click.IntRange(min=72))
def visualize_residualization_sample(
    config_path: Path,
    input_db: Path | None,
    input_table: str | None,
    variables: tuple[str, ...],
    suffix: str | None,
    expected_suffix: str | None,
    row: int | None,
    col: int | None,
    selection: str,
    seed: int,
    min_samples: int,
    output_dir: Path | None,
    dpi: int,
) -> None:
    """Export a transparent paper-ready residualization example."""
    config = read_config(config_path)
    experiment_dir = config_path.parent
    experiment_name = str(config["name"])
    residualization = config.get("residualization") or {}
    suffix = suffix or str(residualization.get("suffix", "_resid"))
    expected_suffix = expected_suffix or str(
        residualization.get("expected_suffix", "_seasonal_trend")
    )

    input_db = resolve_path(
        experiment_dir,
        input_db or config.get("timeseries_db"),
        experiment_dir / f"{experiment_name}_ard.duckdb",
    )
    input_table = input_table or str(
        config.get("timeseries_table", f"{experiment_name}_residualized")
    )
    if output_dir is None:
        output_dir = experiment_dir / f"{experiment_name}_residualization_sample"
    if not input_db.exists():
        raise click.ClickException(f"Input database does not exist: {input_db}")

    con = duckdb.connect(input_db, read_only=True)
    try:
        available_columns = table_columns(con, input_table)
        selected_variables = infer_variables(
            config=config,
            available_columns=available_columns,
            requested_variables=variables,
            suffix=suffix,
            expected_suffix=expected_suffix,
        )
        candidates = candidate_pixels(
            con=con,
            table=input_table,
            variables=selected_variables,
            suffix=suffix,
            expected_suffix=expected_suffix,
            min_samples=min_samples,
        )
        selected_row, selected_col, selection_metadata = select_pixel(
            candidates=candidates,
            selection=selection.lower(),
            seed=seed,
            row=row,
            col=col,
        )
        frame = load_pixel_data(
            con=con,
            table=input_table,
            row=selected_row,
            col=selected_col,
            variables=selected_variables,
            suffix=suffix,
            expected_suffix=expected_suffix,
        )
    finally:
        con.close()

    summaries = pd.DataFrame(
        [
            summarize_variable(
                frame=frame,
                variable=variable,
                suffix=suffix,
                expected_suffix=expected_suffix,
            )
            for variable in selected_variables
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"residualization_sample_row{selected_row}_col{selected_col}"
    output_png = output_dir / f"{stem}.png"
    output_pdf = output_dir / f"{stem}.pdf"
    data_path = output_dir / f"{stem}_data.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"
    caption_path = output_dir / f"{stem}_caption.txt"

    export_columns = ["row", "col", "year", "month", "date"]
    for variable in selected_variables:
        export_columns.extend(
            [
                variable,
                f"{variable}{expected_suffix}",
                f"{variable}{suffix}",
            ]
        )
    frame[export_columns].to_csv(data_path, index=False)
    summaries.to_csv(summary_path, index=False)
    create_figure(
        frame=frame,
        summaries=summaries,
        variables=selected_variables,
        suffix=suffix,
        expected_suffix=expected_suffix,
        row=selected_row,
        col=selected_col,
        output_png=output_png,
        output_pdf=output_pdf,
        dpi=dpi,
    )
    write_caption(
        output_path=caption_path,
        row=selected_row,
        col=selected_col,
        variables=selected_variables,
        selection_strategy=selection_metadata["selection_strategy"],
    )

    metadata = {
        "experiment": experiment_name,
        "config_path": str(config_path),
        "input_db": str(input_db),
        "input_table": input_table,
        "variables": list(selected_variables),
        "suffix": suffix,
        "expected_suffix": expected_suffix,
        "selected_row": selected_row,
        "selected_col": selected_col,
        "min_samples": min_samples,
        "n_exported_observations": int(len(frame)),
        **selection_metadata,
        "outputs": {
            "figure_png": str(output_png),
            "figure_pdf": str(output_pdf),
            "sample_data_csv": str(data_path),
            "summary_csv": str(summary_path),
            "caption_txt": str(caption_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    click.echo(f"Selected pixel: row={selected_row}, col={selected_col}")
    click.echo(f"Selection strategy: {selection_metadata['selection_strategy']}")
    click.echo("Variables: " + ", ".join(selected_variables))
    click.echo(f"Observations: {len(frame)}")
    click.echo(f"PNG figure: {output_png}")
    click.echo(f"PDF figure: {output_pdf}")
    click.echo(f"Sample data: {data_path}")
    click.echo(f"Summary: {summary_path}")
    click.echo(f"Caption draft: {caption_path}")
    click.echo(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    visualize_residualization_sample()
