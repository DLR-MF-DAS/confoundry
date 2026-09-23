"""Prepare diagnostics-qualified VAR-LiNGAM populations for publication."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import click
import duckdb
import pandas as pd

from confoundry.analysis_helpers import ensure_identifier, write_dataframe_table
from confoundry.per_pixel_varlingam_analysis import (
    aggregate_effects,
    plot_effect_trajectories,
)


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    source: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise click.ClickException(
            f"{source} is missing required columns: {missing}"
        )


def build_quality_populations(
    diagnostics: pd.DataFrame,
    *,
    row_columns: Sequence[str],
    residual_corr_threshold: float,
    residual_crosslag_corr_threshold: float,
    autocorr_threshold: float,
    bootstrap_stable_fraction_min: float,
) -> pd.DataFrame:
    """Build primary and nested analytical-sensitivity pixel masks."""
    required = [
        *row_columns,
        "var_lags",
        "residual_whiteness_p",
        "residual_whiteness_rejected",
        "residual_whiteness_bootstrap_p",
        "residual_whiteness_bootstrap_rejected",
        "residual_whiteness_bootstrap_status",
        "residual_whiteness_bootstrap_samples_valid",
        "residual_max_abs_corr",
        "residual_lag1_max_median_abs_autocorr",
        "residual_crosslag_max_abs_corr",
        "residual_nongaussian_fraction",
        "near_constant_variable_count",
        "var_stability_radius",
        "var_stable",
        "var_bootstrap_stable_fraction",
    ]
    _require_columns(diagnostics, required, "diagnostics table")
    if diagnostics.duplicated(list(row_columns)).any():
        raise click.ClickException(
            f"Diagnostics contain duplicate pixel keys for {list(row_columns)}."
        )

    qc = diagnostics[required].copy()
    other_assumptions = (
        qc["var_stable"].fillna(False).astype(bool)
        & qc["var_bootstrap_stable_fraction"]
        .fillna(0.0)
        .ge(bootstrap_stable_fraction_min)
        & qc["near_constant_variable_count"].fillna(0).eq(0)
        & qc["residual_max_abs_corr"].lt(residual_corr_threshold)
        & qc["residual_lag1_max_median_abs_autocorr"].lt(
            autocorr_threshold
        )
        & qc["residual_crosslag_max_abs_corr"].lt(
            residual_crosslag_corr_threshold
        )
    )
    calibrated_whiteness = (
        qc["residual_whiteness_bootstrap_status"].eq("ok")
        & qc["residual_whiteness_bootstrap_rejected"]
        .fillna(True)
        .eq(False)
    )
    qc["other_assumptions_pass"] = other_assumptions
    qc["primary_eligible"] = other_assumptions & calibrated_whiteness
    qc["analytical_sensitivity_eligible"] = (
        qc["primary_eligible"]
        & qc["residual_whiteness_rejected"].fillna(True).eq(False)
    )
    return qc.sort_values(list(row_columns)).reset_index(drop=True)


def _read_table(path: Path, table: str) -> pd.DataFrame:
    table_sql = ensure_identifier(table)
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = set(con.execute("SHOW TABLES").fetchdf()["name"])
        if table not in tables:
            raise click.ClickException(
                f"Table {table!r} not found in {path}; "
                f"available tables: {sorted(tables)}"
            )
        return con.execute(f"SELECT * FROM {table_sql}").fetchdf()
    finally:
        con.close()


def _materialize_population(
    con: duckdb.DuckDBPyConnection,
    *,
    qc: pd.DataFrame,
    row_columns: Sequence[str],
    eligibility_column: str,
    input_table: str,
    output_table: str,
    summary_table: str,
) -> tuple[pd.DataFrame, int, int]:
    eligible = qc.loc[
        qc[eligibility_column].eq(True),
        list(row_columns),
    ].drop_duplicates()
    input_sql = ensure_identifier(input_table)
    output_sql = ensure_identifier(output_table)
    using_sql = ", ".join(ensure_identifier(column) for column in row_columns)
    con.register("_eligible_pixels", eligible)
    try:
        con.execute(
            f"CREATE OR REPLACE TABLE {output_sql} AS "
            f"SELECT effects.* FROM {input_sql} AS effects "
            f"INNER JOIN _eligible_pixels USING ({using_sql})"
        )
    finally:
        con.unregister("_eligible_pixels")
    effects = con.execute(f"SELECT * FROM {output_sql}").fetchdf()
    summary = aggregate_effects(effects)
    effect_rows = int(len(effects))
    write_dataframe_table(con, summary, summary_table)
    return summary, int(len(eligible)), effect_rows


def prepare_outputs(
    *,
    diagnostics_db: Path,
    diagnostics_table: str,
    effects_db: Path,
    effects_table: str,
    output_dir: Path,
    row_columns: Sequence[str],
    residual_corr_threshold: float,
    residual_crosslag_corr_threshold: float,
    autocorr_threshold: float,
    bootstrap_stable_fraction_min: float,
    make_plots: bool,
) -> dict[str, Any]:
    """Create QC masks, effect subsets, summaries, and primary trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = _read_table(diagnostics_db, diagnostics_table)
    qc = build_quality_populations(
        diagnostics,
        row_columns=row_columns,
        residual_corr_threshold=residual_corr_threshold,
        residual_crosslag_corr_threshold=residual_crosslag_corr_threshold,
        autocorr_threshold=autocorr_threshold,
        bootstrap_stable_fraction_min=bootstrap_stable_fraction_min,
    )
    qc_path = output_dir / "pixel_qc.csv"
    qc.to_csv(qc_path, index=False)

    con = duckdb.connect(str(effects_db))
    try:
        tables = set(con.execute("SHOW TABLES").fetchdf()["name"])
        if effects_table not in tables:
            raise click.ClickException(
                f"Table {effects_table!r} not found in {effects_db}; "
                f"available tables: {sorted(tables)}"
            )
        primary_summary, primary_pixels, primary_effect_rows = (
            _materialize_population(
                con,
                qc=qc,
                row_columns=row_columns,
                eligibility_column="primary_eligible",
                input_table=effects_table,
                output_table="pixel_varlingam_effects_primary",
                summary_table="varlingam_effect_summary_primary",
            )
        )
        analytical_summary, analytical_pixels, analytical_effect_rows = (
            _materialize_population(
                con,
                qc=qc,
                row_columns=row_columns,
                eligibility_column="analytical_sensitivity_eligible",
                input_table=effects_table,
                output_table=(
                    "pixel_varlingam_effects_analytical_sensitivity"
                ),
                summary_table=(
                    "varlingam_effect_summary_analytical_sensitivity"
                ),
            )
        )
        all_summary = con.execute(
            "SELECT * FROM varlingam_effect_summary"
        ).fetchdf()
        metadata = pd.DataFrame(
            [
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "diagnostics_db": str(diagnostics_db),
                    "diagnostics_table": diagnostics_table,
                    "effects_db": str(effects_db),
                    "effects_table": effects_table,
                    "residual_corr_threshold": residual_corr_threshold,
                    "residual_crosslag_corr_threshold": (
                        residual_crosslag_corr_threshold
                    ),
                    "autocorr_threshold": autocorr_threshold,
                    "bootstrap_stable_fraction_min": (
                        bootstrap_stable_fraction_min
                    ),
                    "all_diagnostic_pixels": int(len(qc)),
                    "primary_pixels": primary_pixels,
                    "analytical_sensitivity_pixels": analytical_pixels,
                    "primary_effect_rows": primary_effect_rows,
                    "analytical_sensitivity_effect_rows": (
                        analytical_effect_rows
                    ),
                }
            ]
        )
        write_dataframe_table(
            con,
            metadata,
            "varlingam_population_run_metadata",
        )
    finally:
        con.close()

    summaries = []
    for name, summary in [
        ("all_stable", all_summary),
        ("primary", primary_summary),
        ("analytical_sensitivity", analytical_summary),
    ]:
        summary.to_csv(output_dir / f"effect_summary_{name}.csv", index=False)
        labelled = summary.copy()
        labelled.insert(0, "analysis_population", name)
        summaries.append(labelled)
    pd.concat(summaries, ignore_index=True).to_csv(
        output_dir / "effect_summary_sensitivity_comparison.csv",
        index=False,
    )
    metadata.to_csv(
        output_dir / "varlingam_population_run_metadata.csv",
        index=False,
    )
    if make_plots:
        plot_effect_trajectories(
            primary_summary,
            output_dir / "primary_effect_plots",
            {},
        )
    return {
        "qc_path": qc_path,
        "all_pixels": int(len(qc)),
        "primary_pixels": primary_pixels,
        "analytical_sensitivity_pixels": analytical_pixels,
        "primary_effect_rows": primary_effect_rows,
        "analytical_sensitivity_effect_rows": analytical_effect_rows,
    }


@click.command()
@click.option(
    "--diagnostics-db",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--diagnostics-table",
    default="pixel_graph_diagnostics",
    show_default=True,
)
@click.option(
    "--effects-db",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--effects-table",
    default="pixel_varlingam_effects",
    show_default=True,
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
)
@click.option(
    "--row-column",
    "row_columns",
    multiple=True,
    default=("row", "col"),
    show_default=True,
)
@click.option(
    "--residual-corr-threshold",
    default=0.20,
    show_default=True,
    type=click.FloatRange(0.0, None),
)
@click.option(
    "--residual-crosslag-corr-threshold",
    default=0.30,
    show_default=True,
    type=click.FloatRange(0.0, None),
)
@click.option(
    "--autocorr-threshold",
    default=0.30,
    show_default=True,
    type=click.FloatRange(0.0, None),
)
@click.option(
    "--bootstrap-stable-fraction-min",
    default=0.95,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
)
@click.option("--no-plots", is_flag=True)
def prepare_varlingam_paper_outputs(
    diagnostics_db: Path,
    diagnostics_table: str,
    effects_db: Path,
    effects_table: str,
    output_dir: Path,
    row_columns: tuple[str, ...],
    residual_corr_threshold: float,
    residual_crosslag_corr_threshold: float,
    autocorr_threshold: float,
    bootstrap_stable_fraction_min: float,
    no_plots: bool,
) -> None:
    """Create paper populations and summaries from diagnostics and effects."""
    if len(row_columns) < 1:
        raise click.BadParameter(
            "At least one coordinate column is required.",
            param_hint="--row-column",
        )
    result = prepare_outputs(
        diagnostics_db=diagnostics_db,
        diagnostics_table=diagnostics_table,
        effects_db=effects_db,
        effects_table=effects_table,
        output_dir=output_dir,
        row_columns=row_columns,
        residual_corr_threshold=residual_corr_threshold,
        residual_crosslag_corr_threshold=(
            residual_crosslag_corr_threshold
        ),
        autocorr_threshold=autocorr_threshold,
        bootstrap_stable_fraction_min=bootstrap_stable_fraction_min,
        make_plots=not no_plots,
    )
    click.echo(f"Wrote pixel QC: {result['qc_path']}")
    click.echo(
        f"Pixels: all={result['all_pixels']}; "
        f"primary={result['primary_pixels']}; analytical sensitivity="
        f"{result['analytical_sensitivity_pixels']}"
    )
    click.echo(
        f"Effect rows: primary={result['primary_effect_rows']}; "
        f"analytical sensitivity="
        f"{result['analytical_sensitivity_effect_rows']}"
    )


if __name__ == "__main__":
    prepare_varlingam_paper_outputs()
