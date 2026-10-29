# Computational reproduction of the per-pixel VAR–LiNGAM analysis

This guide gives the commands required to reproduce the production analysis from data acquisition through the paper figures.  Commands are run from the repository root with `uv`.  The commands write new source, analysis-ready, graph, diagnostic, and effect databases; they do not require previously generated analysis outputs.

## Required inputs

Reproduction requires the v0.0.1 release of Confoundry. It can be found by following this link <https://github.com/DLR-MF-DAS/confoundry/releases/tag/v0.0.1>. The download covers January 2005 through December 2024.  ERA5-Land monthly means are retrieved for 2-m air temperature, total precipitation, potential evaporation, and volumetric soil water in layers 2 and 3.  Monthly MODIS MOD13C2 NDVI is retrieved on its 0.05-degree grid, which is subsequently used as the reference grid during data gathering.  ERA5 retrieval requires a valid Copernicus Climate Data Store account, acceptance of the ERA5-Land licence, and a working `.cdsapirc`; both downloaders require network access.

Because upstream providers can revise hosted products, an exact archival reproduction should preserve the downloaded GeoTIFFs and their source-catalog database.  Re-running the download command reconstructs the dataset exposed by the providers at the time of execution.

## Environment and paths

Install the environment.

```bash
uv sync
```

Define the paths.

```bash
EXP="data/iberian_drought_experiment"
RUN="$EXP/paper_var3_reproduction"
DOWNLOAD_CFG="$EXP/download.yaml"
RAW_CFG="$EXP/experiment.yaml"
ARD_DB="$EXP/iberian_droughts_ard.duckdb"
CFG="$EXP/iberian_droughts_residualized_monthly.yaml"
GRAPH_DB="$RUN/varlingam_var3_graphs.duckdb"
DIAG_DB="$RUN/varlingam_var3_diagnostics.duckdb"
EFFECT_DB="$RUN/varlingam_effects.duckdb"
EFFECT_CSV="$RUN/varlingam_effects.csv"
QC_CSV="$RUN/pixel_qc.csv"
REPORT_DIR="$RUN/diagnostics_report"
MAP_DIR="$RUN/per_pixel_effect_maps"
WORKERS=160
mkdir -p "$RUN"
```

## Step 1: download the monthly source rasters

Download and spatially clip all configured ERA5-Land and MODIS NDVI products. The downloader writes GeoTIFFs below the experiment directory and records one catalog row per variable and month in `iberian_droughts_source_db.duckdb`.

```bash
uv run python \
  -m confoundry.download_to_db \
  --config-path "$DOWNLOAD_CFG"
```

## Step 2: gather the analysis-ready monthly table

Align every source raster to the monthly NDVI reference grid, apply raster scale and offset metadata, convert nodata values to missing values, and write the long-form analysis-ready table `iberian_droughts`.

```bash
uv run python \
  -m confoundry.gather \
  --config-path "$RAW_CFG" \
  --max-workers "$WORKERS"
```

## Step 3: remove deterministic seasonality and trend

Residualize each of the six environmental variables independently at every pixel. Residuals are written back to the ARD database, and a generated YAML restricts graph discovery to the six residual columns.

```bash
uv run python \
  -m confoundry.residualize_timeseries \
  --config-path "$RAW_CFG" \
  --input-db "$ARD_DB" \
  --input-table iberian_droughts \
  --output-db "$ARD_DB" \
  --output-table iberian_droughts_residualized_monthly \
  --output-config "$CFG" \
  --graph-db "$GRAPH_DB" \
  --seasonal-model monthly-fixed-effects \
  --trend \
  --min-fit-samples 60 \
  --min-month-samples 3
```

Produce the residualization audit figure and its plotted data table:

```bash
uv run python \
  -m confoundry.visualize_residualization_sample \
  --config-path "$CFG" \
  --selection representative \
  --output-dir "$RUN/residualization_validation"
```

## Step 4: fit the production graphs

Fit a separate pruned VAR–LiNGAM model at every eligible pixel.  The lag order is fixed at three for the complete six-variable VAR. Five hundred paired bootstrap graph fits are stored for uncertainty analysis.

```bash
uv run python \
  -m confoundry.per_pixel_graph_discovery \
  --config-path "$CFG" \
  --model varlingam \
  --var-lags 3 \
  --var-criterion none \
  --var-prune \
  --window-size 0 \
  --bootstrap-samples 500 \
  --min-edge-prob 0.70 \
  --min-abs-effect 0.01 \
  --workers "$WORKERS" \
  --output-db "$GRAPH_DB"
```

## Step 5: calculate calibrated diagnostics

Calculate structural-error independence, reduced-form innovation dependence, companion-matrix stability, graph-bootstrap stability, and innovation whiteness through lag 12.  The analytical portmanteau statistic is calibrated at every pixel using 499 joint residual-vector bootstrap simulations.  Each simulation uses a 200-month burn-in and refits the same VAR order before the diagnostic statistic is recomputed.

```bash
uv run python \
  -m confoundry.per_pixel_graph_diagnostics \
  --config-path "$CFG" \
  --graphs-db-path "$GRAPH_DB" \
  --diagnostics-db-path "$DIAG_DB" \
  --window-size 0 \
  --diagnostic-alpha 0.05 \
  --whiteness-lags 12 \
  --whiteness-bootstrap-samples 499 \
  --whiteness-bootstrap-burnin 200 \
  --whiteness-bootstrap-seed 0 \
  --whiteness-bootstrap-pixel-limit 0 \
  --residual-corr-threshold 0.20 \
  --residual-crosslag-corr-threshold 0.30 \
  --autocorr-threshold 0.30 \
  --stability-bootstrap-limit 0 \
  --workers "$WORKERS"
```

Generate the diagnostic report, including the innovation-lag profile and its underlying CSV tables:

```bash
uv run python \
  -m confoundry.visualize_directlingam_diagnostics \
  --diagnostics-db "$DIAG_DB" \
  --output-dir "$REPORT_DIR"
```

## Step 6: estimate all per-pixel dynamic effects

Estimate responses of NDVI to each of the other five variables at horizons 0--12.  Point estimates use the raw pruned matrices.  Setting `--bootstrap-limit 0` uses all 500 paired graph-bootstrap replicates for the reported effect intervals.

```bash
uv run python \
  -m confoundry.per_pixel_varlingam_analysis \
  --config-path "$CFG" \
  --graphs-db "$GRAPH_DB" \
  --graphs-table pixel_graphs \
  --target ndvi_resid \
  --horizon 12 \
  --point-matrix raw \
  --low-quantile 0.1 \
  --high-quantile 0.9 \
  --ci 0.95 \
  --stability-threshold 1.0 \
  --bootstrap-limit 0 \
  --output-csv "$EFFECT_CSV" \
  --output-db "$EFFECT_DB" \
  --plot-dir "$RUN/all_stable_effect_plots" \
  --jobs "$WORKERS"
```

## Step 7: construct the paper analysis populations

Constructs the pixel QC mask from diagnostic fields only, then write the primary and nested analytical-sensitivity effect tables.  The thresholds are given explicitly in the command.

```bash
uv run python \
  -m confoundry.prepare_varlingam_paper_outputs \
  --diagnostics-db "$DIAG_DB" \
  --diagnostics-table pixel_graph_diagnostics \
  --effects-db "$EFFECT_DB" \
  --effects-table pixel_varlingam_effects \
  --output-dir "$RUN" \
  --residual-corr-threshold 0.20 \
  --residual-crosslag-corr-threshold 0.30 \
  --autocorr-threshold 0.30 \
  --bootstrap-stable-fraction-min 0.95
```

## Step 8: render the per-pixel effect maps

Render a five-panel map of cumulative effects through month 12, a primary-QC coverage map, and supplementary horizon-specific maps.

```bash
uv run python \
  -m confoundry.visualize_varlingam_effect_maps \
  --effects-db "$EFFECT_DB" \
  --effects-table pixel_varlingam_effects_primary \
  --qc-csv "$QC_CSV" \
  --output-dir "$MAP_DIR" \
  --target ndvi_resid \
  --main-horizon 12 \
  --supplement-horizons 0,1,3,6,12 \
  --color-quantile 0.98 \
  --stipple-stride 2
```

## Step 9: run the validation on synthetic data

```bash
uv run python -m confoundry.validate_varlingam_synthetic \
  --config-path "$CFG" \
  --input-db "$ARD_DB" \
  --input-table iberian_droughts_residualized_monthly \
  --graphs-db "$GRAPH_DB" \
  --graphs-table pixel_graphs \
  --row 120 \
  --col 31 \
  --target ndvi_resid \
  --n-samples 240 \
  --replicates 500 \
  --burnin 200 \
  --noise-mode empirical-independent \
  --seed 0 \
  --var-lags 3 \
  --var-criterion none \
  --var-prune \
  --edge-threshold 0.01 \
  --horizon 12 \
  --low-quantile 0.10 \
  --high-quantile 0.90 \
  --workers 32 \
  --output-dir "$RUN/synthetic_validation/row_120_col_31"
```

## Location of generated figures

All figure paths below are relative to the run directory `data/iberian_drought_experiment/paper_var3_reproduction`. Each plotting command writes both a 300-dpi PNG file and a vector PDF file with the same basename.  The PDF version should normally be used in the manuscript.

- **Residualization audit.** The representative-pixel audit is written to `residualization_validation/`.  Its filename is `residualization_sample_row<ROW>_col<COL>.pdf`, where `<ROW>` and `<COL>` are the automatically selected grid coordinates printed by the command.  The same directory contains the plotted data, summary statistics, autocorrelation table, metadata, and a draft caption with the same filename prefix.
- **Model diagnostics.** Diagnostic figures are written to `diagnostics_report/figures/`.  The compact four-panel diagnostic figure is `varlingam_minimal_diagnostics.pdf`, and the lag-by-lag innovation diagnostic is `innovation_lag_profile.pdf`.  Additional diagnostic histograms, spatial heatmaps, and bar charts are stored in the same directory. Their underlying tables are in `diagnostics_report/tables/`.
- **Primary effect trajectories.** The response trajectories for pixels passing the primary quality-control criteria are written to `primary_effect_plots/`.  The five PDF files are:
  -  `varlingam_effect_temperature_resid_to_ndvi_resid.pdf`;
  -  `varlingam_effect_precipitation_resid_to_ndvi_resid.pdf`;
  -  `varlingam_effect_evaporation_resid_to_ndvi_resid.pdf`;
  -  `varlingam_effect_soil_moisture_7_to_28_cm_resid_to_ndvi_resid.pdf`;
  -  `varlingam_effect_soil_moisture_28_to_100_cm_resid_to_ndvi_resid.pdf`.

  Each figure shows the spatial median and the 5th–95th percentile range of the horizon-specific and cumulative scaled effects over the primary analysis population.
- **Main cumulative-effect map.** The principal five-panel spatial result is `per_pixel_effect_maps/figures/cumulative_effects_main.pdf`. It maps the primary-population cumulative effect through month 12 for each of the five source variables.
- **Quality-control coverage map.** The fitted footprint and the pixels retained in the primary population are shown in `per_pixel_effect_maps/figures/primary_qc_coverage.pdf`.
- **Horizon-specific supplementary maps.** The month-specific effects are stored in `per_pixel_effect_maps/figures/` as `horizon_effects_month_xx.pdf`. The corresponding plotted values and figure metadata are in `per_pixel_effect_maps/tables/`.

The directory `all_stable_effect_plots/` contains preliminary trajectories over all dynamically stable fitted pixels.  These are retained as sensitivity outputs and should not be substituted for the primary-quality-control trajectories in `primary_effect_plots/`.

## Related analysis

The separate [external aridity-stratification guide](varlingam_aridity_stratification.md) documents the independent climatic-gradient analysis.
