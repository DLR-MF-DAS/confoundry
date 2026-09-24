# Single-pixel synthetic VAR--LiNGAM validation

This workflow tests whether the complete analysis can recover known parameters
from a time series with the same length as the observations. It uses one fitted
pixel as the data-generating model, rather than inventing coefficients unrelated
to the Iberian analysis.

For each Monte Carlo replicate, the command:

1. reads the selected pixel's raw pruned contemporaneous and lag-1 to lag-3
   structural matrices from `pixel_graphs`;
2. calculates that pixel's structural errors from the observed residual series;
3. independently resamples each error component, preserving its empirical and
   potentially non-Gaussian distribution while enforcing the structural-error
   independence assumed by VAR--LiNGAM;
4. simulates a stable structural VAR for a burn-in period and retains 240 months;
5. evaluates and adds the pixel's stored calendar-month effects and linear trend;
6. re-estimates and removes those deterministic components using the production
   residualization code;
7. refits the same pruned VAR--LiNGAM model; and
8. compares the recovered coefficients, edge support, coefficient signs,
   stability, innovations, and NDVI dynamic effects with their known values.

The pixel must have a successful residualization model, a fitted VAR--LiNGAM
graph, and a point-estimate spectral radius below one. For a paper, select the
pixel by a rule fixed before viewing the recovery result. A defensible simple
choice is the primary-QC pixel nearest the spatial centre of the study area. Do
not search across pixels and then report only the one with the best recovery.

With the mamba environment active, define the production paths and the chosen
pixel coordinates:

```bash
EXP="data/iberian_drought_experiment"
RUN="$EXP/paper_var3_prod_v1"
CFG="$EXP/iberian_droughts_residualized_monthly.yaml"
ARD_DB="$EXP/iberian_droughts_ard.duckdb"
GRAPH_DB="$RUN/varlingam_var3_graphs.duckdb"
QC_CSV="$RUN/pixel_qc.csv"
ROW=REPLACE_WITH_GRID_ROW
COL=REPLACE_WITH_GRID_COLUMN
OUT="$RUN/synthetic_validation/row_${ROW}_col_${COL}"
```

For example, the following returns the primary-QC pixel nearest the mean grid
position of the eligible population. Copy its `row` and `col` into the two
variables above and record this selection rule in the paper:

```bash
duckdb -c "
WITH eligible AS (
    SELECT row, col
    FROM read_csv_auto('$QC_CSV', header = true)
    WHERE primary_eligible
), centre AS (
    SELECT avg(row) AS centre_row, avg(col) AS centre_col
    FROM eligible
)
SELECT row, col
FROM eligible, centre
ORDER BY pow(row - centre_row, 2) + pow(col - centre_col, 2), row, col
LIMIT 1;
"
```

Run the primary experiment with the production lag order and pruning setting:

```bash
python -m confoundry.validate_varlingam_synthetic \
  --config-path "$CFG" \
  --input-db "$ARD_DB" \
  --input-table iberian_droughts_residualized_monthly \
  --graphs-db "$GRAPH_DB" \
  --graphs-table pixel_graphs \
  --row "$ROW" \
  --col "$COL" \
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
  --output-dir "$OUT"
```

`--workers 32` is only an example. Set it to a value appropriate for the memory
and CPU allocation. Every replicate has a deterministic child seed derived from
`--seed`, so changing the number of workers does not change the simulated data.

The main outputs are:

- `coefficient_recovery.pdf`: known standardized coefficients against median
  recovered coefficients; vertical ranges show the 5th--95th percentiles over
  the simulated records;
- `graph_support_recovery.pdf`: distributions of edge precision, recall, F1,
  and sign agreement;
- `dynamic_effect_recovery.pdf`: known and recovered non-cumulative NDVI effects
  at horizons 0--12;
- `cumulative_effect_recovery.pdf`: the corresponding cumulative effects;
- `residualization_recovery.pdf`: error in recovering the deterministic baseline,
  expressed relative to the original residual standard deviation;
- `summary_metrics.csv`: numerical recovery summaries for all matrices and for
  the contemporaneous and individual lag matrices;
- `dynamic_effect_summary.csv`: known effect, recovered interval, bias, and RMSE
  for every source and horizon; and
- `validation_report.md`: a short interpretation plus the experiment's main
  limitations.

The long-form CSV files contain every replicate. `replicate_status.csv` records
failed fits rather than dropping them silently. `simulation_parameters.json`,
`source_causal_coefficients.csv`, `source_residualization_parameters.csv`, and
`source_structural_innovations.csv` provide the exact audit trail needed to
reproduce the data-generating process. `source_innovation_moments.csv` reports
the skewness and excess kurtosis of each empirical shock distribution, so its
departure from Gaussianity is explicit rather than assumed.

The command refits the same pruned point-estimate model but does not run 500
graph bootstraps inside every replicate. The outer Monte Carlo replicates are
the relevant sampling distribution for parameter recovery; nesting the
production graph bootstrap inside them would multiply the cost without changing
the point coefficients being evaluated. Production bootstrap support and this
synthetic recovery rate answer different questions and should be reported
separately.

Interpret precision and recall together. High precision with low recall means
that recovered edges are usually correct but many true edges are missed. High
recall with low precision means that most true edges are found but many extra
edges are introduced. Coefficient correlation measures relative recovery,
whereas standardized RMSE measures the size of the errors. Report both the
all-coefficient RMSE and the true-edge-only RMSE: the former can look small
simply because the coefficient matrices contain many true zeros. Dynamic and
cumulative-effect recovery should also be reported because small errors in
several lagged coefficients can accumulate along a response pathway.

Two optional sensitivity runs are useful:

```bash
python -m confoundry.validate_varlingam_synthetic ... \
  --noise-mode empirical-joint \
  --output-dir "$RUN/synthetic_validation_joint_errors/row_${ROW}_col_${COL}"

python -m confoundry.validate_varlingam_synthetic ... \
  --noise-mode gaussian-independent \
  --output-dir "$RUN/synthetic_validation_gaussian/row_${ROW}_col_${COL}"
```

The joint-error run preserves observed dependence between the estimated shocks
and therefore deliberately relaxes the LiNGAM independence assumption. The
Gaussian run is a negative control for LiNGAM's non-Gaussian identification.
They should not replace the primary independent empirical-error experiment.

This experiment validates software behavior and finite-sample recoverability
conditional on the selected fitted graph. It does not independently prove that
the real-data graph is causally correct. A single pixel also cannot establish
spatial generality; if the result is intended as more than an illustrative
validation, repeat the preregistered procedure for several pixels spanning
different climates, graph densities, and regions.
