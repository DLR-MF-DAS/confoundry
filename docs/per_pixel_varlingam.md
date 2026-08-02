# Per-pixel VAR-LiNGAM

VAR-LiNGAM is an opt-in alternative in the existing per-pixel graph-discovery
command. DirectLiNGAM remains the default.

## Configuration

The existing residualized YAML can be reused. Add these keys to its
`graph_discovery` section:

```yaml
graph_discovery:
  model: varlingam
  var_lags: 1
  var_criterion: none
  var_prune: true
```

`var_lags` must be at least 1. `var_criterion: none` fits exactly that many
lags. `aic`, `fpe`, `hqic`, or `bic` instead select between lag orders 1 and
`var_lags`.

Run graph discovery as before:

```bash
PYTHONPATH=src python -m confoundry.per_pixel_graph_discovery \
  --config-path path/to/residualized.yaml \
  --bootstrap-samples 200 \
  --workers 4
```

The same settings can be supplied without editing the YAML:

```bash
PYTHONPATH=src python -m confoundry.per_pixel_graph_discovery \
  --config-path path/to/residualized.yaml \
  --model varlingam \
  --var-lags 1 \
  --var-criterion none \
  --var-prune \
  --bootstrap-samples 200 \
  --workers 4
```

VAR-LiNGAM requires `--window-size 0`. Variables are kept aligned at the same
month, so nonzero `shift` values in the YAML are deliberately ignored for this
model. A pixel is skipped if its complete observations do not form a unique,
uninterrupted monthly sequence or do not provide `min_samples` observations
after accounting for the fitted lags.

## Output

Unless `--output-db` or `graph_discovery.var_output_db` is supplied, a VAR run
does not replace the DirectLiNGAM database. For example,
`demo_residualized_graphs.duckdb` becomes
`demo_residualized_varlingam_graphs.duckdb`.

The existing adjacency fields retain their meaning for contemporaneous effects:

- `adjacency_raw_json`, `edge_probability_json`, and
  `adjacency_consensus_json` contain the contemporaneous matrix \(B_0\).
- `adjacency_lagged_raw_json`, `edge_probability_lagged_json`, and
  `adjacency_lagged_consensus_json` contain \(B_1,\ldots,B_p\). Array element
  zero is lag 1.
- Matrix entries follow LiNGAM's child-by-parent convention:
  `B[child, parent]` is the effect `parent → child`.
- The existing GML graph and `order_graphs` workflow use \(B_0\), preserving
  their current behavior.

The `graph_discovery_run_metadata` table records the model, requested lag
settings, whether configured shifts were applied, and how many pixel tasks were
skipped.

## Diagnostics

When `graph_discovery.model: varlingam` is present in the YAML, diagnostics find
the separate VAR graph database automatically:

```bash
PYTHONPATH=src python -m confoundry.per_pixel_graph_diagnostics \
  --config-path path/to/residualized.yaml \
  --whiteness-lags 12 \
  --workers 4
```

If VAR-LiNGAM was selected only on the graph-discovery command line, identify
the graph database explicitly:

```bash
PYTHONPATH=src python -m confoundry.per_pixel_graph_diagnostics \
  --config-path path/to/residualized.yaml \
  --graphs-db-path path/to/demo_residualized_varlingam_graphs.duckdb \
  --workers 4
```

For VAR graphs, two error sequences serve different diagnostic purposes. The
post-pruning structural errors are computed from the saved contemporaneous and
lagged matrices:

\[
e_t = x_t - B_0 x_t - \sum_{\tau=1}^{p} B_\tau x_{t-\tau}.
\]

These errors are used for the contemporaneous independence and non-Gaussianity
diagnostics. Temporal diagnostics instead refit an intercept-free reduced-form
VAR at the graph's saved lag order and use its innovations

\[
u_t = x_t - \sum_{\tau=1}^{p} M_\tau x_{t-\tau}.
\]

This distinction matters when `var_prune` is enabled: VAR-LiNGAM estimates its
lag order and contemporaneous model from the reduced-form innovations, then
re-estimates the final structural matrices during pruning. Residuals
reconstructed from those final matrices need not reproduce the original
reduced-form innovations and are not the right basis for testing lag-order
adequacy. `residual_temporal_basis` records which sequence was used. Neither
sequence is the set of residualized environmental variables produced by the
earlier seasonal/trend residualization procedure.

The VAR diagnostics additionally calculate:

- all current-versus-lagged innovation correlations through
  `--whiteness-lags` months, including cross-variable relationships;
- an adjusted multivariate portmanteau whiteness test;
- contemporaneous and lagged bootstrap edge stability separately;
- the point-estimate companion-matrix spectral radius;
- the fraction of paired bootstrap VAR models with a spectral radius below
  `--stability-threshold`.

The default maximum whiteness lag is 12 months. Important output fields are:

- `residual_max_abs_corr`: maximum same-month correlation between different
  post-pruning structural errors;
- `residual_crosslag_max_abs_corr`: maximum absolute correlation between any
  current and lagged reduced-form VAR innovation;
- `residual_crosslag_top_pairs_json`: source, target, lag, and correlation
  for the strongest temporal innovation relationships. These are diagnostic
  correlations, not causal edges, and the stored top values are subject to
  selection bias;
- `residual_crosslag_by_lag_json`: complete lag-by-lag summaries over every
  variable pair, separating diagonal autocorrelation from cross-variable
  correlation without top-pair selection;
- `residual_whiteness_p` and `residual_whiteness_rejected`: adjusted
  multivariate portmanteau results for reduced-form VAR innovations;
- `residual_whiteness_bootstrap_p` and
  `residual_whiteness_bootstrap_rejected`: optional finite-sample calibrated
  results from a joint residual-vector bootstrap. These fields are added only
  when calibration is requested; the analytical fields remain unchanged;
- `residual_whiteness_by_lag_json`: each lag's contribution to the adjusted
  portmanteau statistic, its share of the total statistic, and the cumulative
  test result;
- `residual_nongaussian_fraction`: fraction of innovations rejecting
  normality in the post-pruning structural errors;
- `lagged_bootstrap_probability_entropy_mean`: ambiguity of lagged-edge
  support, including autoregressive diagonal edges;
- `var_stability_radius` and `var_stable`: point-model dynamic stability;
- `var_bootstrap_stable_fraction`: bootstrap dynamic-stability support.

`bootstrap_probability_entropy_mean` continues to describe contemporaneous
\(B_0\) edges. `lagged_bootstrap_probability_entropy_mean` describes all
\(B_1,\ldots,B_p\) entries, including their meaningful autoregressive
diagonals. Bidirectional direction instability is calculated only for
\(B_0\), because opposite cross-lag relationships are not mutually
exclusive.

Use `--stability-bootstrap-limit` to reduce exploratory runtime. Zero, the
default, evaluates all paired bootstrap matrices.

### Bootstrap calibration of innovation whiteness

The analytical portmanteau p-value uses an asymptotic chi-squared reference
distribution. To check its finite-sample calibration while preserving the
joint, potentially non-Gaussian innovation distribution, enable the optional
residual bootstrap:

```bash
PYTHONPATH=src python -m confoundry.per_pixel_graph_diagnostics \
  --config-path path/to/residualized.yaml \
  --graphs-db-path path/to/varlingam_graphs.duckdb \
  --diagnostics-db-path path/to/bootstrap_pilot_diagnostics.duckdb \
  --whiteness-lags 12 \
  --whiteness-bootstrap-samples 499 \
  --whiteness-bootstrap-burnin 200 \
  --whiteness-bootstrap-seed 0 \
  --whiteness-bootstrap-pixel-limit 500 \
  --workers 4
```

For each selected pixel, diagnostics refit the intercept-free reduced-form VAR
at the graph's stored order, center and jointly resample complete innovation
vectors, simulate from the fitted VAR after the configured burn-in, refit the
same VAR order, and recompute the adjusted portmanteau statistic. The empirical
p-value uses the usual plus-one correction. Joint vector resampling preserves
contemporaneous innovation dependence and non-Gaussianity; variables are never
resampled independently.

`--whiteness-bootstrap-pixel-limit` makes calibration practical as a pilot.
Zero, its default, calibrates every graph pixel when bootstrap samples are
positive. Pixels outside a pilot retain all analytical diagnostics and have
null bootstrap-calibration fields. Pixel selection and per-pixel simulation
are reproducible across worker counts from `--whiteness-bootstrap-seed`.

Additional output fields record requested and valid simulation counts, the
bootstrap-statistic median and 95th percentile, the refitted reduced-form VAR
stability radius, method, burn-in, and status. The separate
`lingam_assumption_warning_bootstrap_calibrated` field substitutes the
bootstrap whiteness decision while leaving every other warning component
unchanged. The original `lingam_assumption_warning` remains unchanged for
backward compatibility.

The run-metadata table records every calibration option and the number of
selected pilot pixels. Summarize a pilot with:

```sql
SELECT
  count(residual_whiteness_bootstrap_p) AS calibrated_pixels,
  round(
    100.0 * count_if(residual_whiteness_bootstrap_rejected)
    / count(residual_whiteness_bootstrap_p),
    2
  ) AS bootstrap_rejected_pct,
  median(residual_whiteness_bootstrap_p) AS median_bootstrap_p,
  round(
    100.0 * count_if(
      residual_whiteness_rejected
      AND residual_whiteness_bootstrap_p IS NOT NULL
    ) / count(residual_whiteness_bootstrap_p),
    2
  ) AS analytical_rejected_pct_same_pixels
FROM pixel_graph_diagnostics;
```

Use at least 199 bootstrap samples for a 0.05 decision. A 499-sample pilot on
spatially sampled pixels is recommended before calibrating the full raster.
Calibration is VAR-LiNGAM-only and requires a stable refitted reduced-form VAR;
the per-pixel status explains skipped calibrations.

Create the publication report and minimal four-panel diagnostic figure with:

```bash
PYTHONPATH=src python -m \
  confoundry.visualize_directlingam_diagnostics \
  --diagnostics-db \
  path/to/demo_residualized_varlingam_graph_diagnostics.duckdb \
  --output-dir path/to/varlingam_diagnostics_report
```

The report includes individual distributions and spatial maps, aggregated
cross-lag innovation pairs, contemporaneous and lagged bootstrap-edge
tables, `innovation_lag_profile.csv`, a matching publication-ready PNG/PDF,
and `varlingam_minimal_diagnostics.png` plus a vector PDF. The lag-profile
figure uses all evaluated variable-lag combinations rather than the selected
top-pair records. The
minimal figure contains:

1. multivariate temporal-whiteness p-values;
2. contemporaneous innovation correlations;
3. innovation non-Gaussianity;
4. companion-matrix spectral radii.

To produce the optional DirectLiNGAM-versus-VAR-LiNGAM temporal comparison,
first run both models on the same pixels, time range, variables, and
`--window-size 0`. Then pass the DirectLiNGAM diagnostics database as:

```bash
PYTHONPATH=src python -m \
  confoundry.visualize_directlingam_diagnostics \
  --diagnostics-db path/to/var_diagnostics.duckdb \
  --comparison-diagnostics-db path/to/direct_diagnostics.duckdb \
  --output-dir path/to/model_diagnostics_comparison
```

This writes a paired temporal-dependence figure and pixel-level comparison
CSV files. A comparison against an earlier windowed DirectLiNGAM run would
conflate model choice with spatial pooling and should not be used as evidence
for the VAR specification.

## Dynamic causal-effect analysis

Run the VAR-specific effect command after graph discovery:

```bash
PYTHONPATH=src python -m confoundry.per_pixel_varlingam_analysis \
  --config-path path/to/residualized.yaml \
  --target ndvi_resid \
  --horizon 12 \
  --point-matrix raw \
  --jobs 4
```

If `analysis.target`, `analysis.outcome`, or `reference_var` names the target
in the YAML, `--target` can be omitted. Use
`--sources temperature_resid,precipitation_resid` to restrict the source
variables.

For the structural VAR

\[
x_t=B_0x_t+\sum_{\ell=1}^{p}B_\ell x_{t-\ell}+e_t,
\]

the command computes

\[
C=(I-B_0)^{-1},\qquad A_\ell=CB_\ell,
\]

and the dynamic response matrices

\[
G_0=C,\qquad
G_h=\sum_{\ell=1}^{p}A_\ell G_{h-\ell}.
\]

For a source `X` and target `Y`, `total_effect` at horizon `h` is
`G_h[Y, X]`: the response of `Y` after a one-unit pulse intervention on `X`,
including contemporaneous paths, lagged paths, and repeated propagation
through the VAR. `cumulative_total_effect` is the sum from horizon zero
through `h`.

The output also keeps two more local quantities:

- `direct_effect` is the saved structural coefficient: `B0[Y, X]` at
  horizon zero and `Bh[Y, X]` at lag `h`.
- `lag_slice_total_effect` is the within-slice total effect:
  `C[Y, X]` at horizon zero and `(C Bh)[Y, X]` for a saved lag. This is the
  quantity corresponding to VAR-LiNGAM's lag-specific, time-unrolled total
  effect; unlike `total_effect`, it does not recursively propagate the
  intervention through subsequent months.

The `scaled_*` columns multiply an effect by the source 10th-to-90th
percentile contrast and divide by the corresponding target contrast. This
makes effects more comparable across variables while leaving the unscaled
physical-unit effects available.

The point estimate uses `raw`, `consensus`, or `bootstrap_mean` matrices.
Bootstrap intervals always use paired contemporaneous and lagged matrices
from the same bootstrap replicate. Singular and dynamically unstable
replicates are excluded and counted. A VAR is treated as stable when the
spectral radius of its reduced-form companion matrix is below
`--stability-threshold` (one by default). Unstable point models remain in the
pixel table with `point_stable=false`, but are excluded from spatial
summaries.

Default outputs are:

- `<name>_varlingam_effects.csv`: pixel-, source-, target-, and
  horizon-level estimates;
- `<name>_varlingam_effects.duckdb::pixel_varlingam_effects`: the same
  estimates;
- `<name>_varlingam_effects.duckdb::varlingam_effect_summary`: spatial
  summaries over stable pixels;
- `<name>_varlingam_effect_plots/`: 300-dpi PNG and vector PDF response
  plots with human-readable variable names.

Known residual suffixes are rendered as “anomaly”, and underscores are
converted to readable text. Use a repeatable option such as
`--variable-label 'ndvi_resid=Vegetation greenness anomaly'` when the paper
requires a specific display name. Raw variable names are retained in the
machine-readable tables.

Pass `--graphs-db` if graph discovery was run with an explicit graph path.
Pass `--bootstrap-limit 500` to use only the first 500 saved paired
replicates during exploratory runs. Zero, the default, uses all of them.

## Dynamic interventions and counterfactuals

The intervention command replaces the structural equation of each
intervened variable for the requested duration and then propagates the
result through both \(B_0\) and all lag matrices:

```bash
PYTHONPATH=src python -m confoundry.per_pixel_varlingam_interventions \
  --config-path path/to/residualized.yaml \
  --target ndvi_resid \
  --intervention wetting soil_moisture_7_to_28_cm_resid qdelta:1 \
  --mode interventional_mean \
  --intervention-duration 1 \
  --horizon 12 \
  --jobs 4
```

An intervention is passed as `SCENARIO VARIABLE SPEC`. Repeat the option with
the same scenario name for a simultaneous joint intervention:

```bash
--intervention joint_wetting soil_moisture_7_to_28_cm_resid qdelta:1 \
--intervention joint_wetting soil_moisture_28_to_100_cm_resid qdelta:1
```

Supported value specifications are:

- `fixed:v`: set the residualized variable to anomaly value `v`;
- `delta:v`: add `v` to its factual or zero-baseline value;
- `quantile:q`: set it to the per-pixel empirical quantile `q`;
- `zdelta:v`: add `v` per-pixel standard deviations;
- `qdelta:v`: add `v` times the per-pixel 10th-to-90th percentile contrast.

A bare number is shorthand for `fixed`. For residualized variables, all
fixed and delta values are on the residualized anomaly scale, not the scale
of the original unresidualized variable.

`interventional_mean` starts from zero history and sets future structural
innovations to zero. It therefore describes the model-implied mean anomaly
response to the intervention. This is usually the clearest mode for a
population-level paper figure based on residualized data.

`counterfactual` instead answers what the fitted model predicts would have
happened during one observed event:

```bash
PYTHONPATH=src python -m confoundry.per_pixel_varlingam_interventions \
  --config-path path/to/residualized.yaml \
  --target ndvi_resid \
  --intervention wetting soil_moisture_7_to_28_cm_resid delta:0.05 \
  --mode counterfactual \
  --start-year 2022 \
  --start-month 7 \
  --horizon 5 \
  --jobs 4
```

For every fitted model, this mode reconstructs the structural innovations
from the factual trajectory and reuses exactly those innovations under the
intervention. The reported `effect` is
`counterfactual_value - factual_value`; `cumulative_effect` sums that
difference through the selected horizon.

Default outputs are:

- `<name>_varlingam_interventions.csv`: pixel-, scenario-, target-, and
  horizon-level trajectories;
- `<name>_varlingam_interventions.duckdb::pixel_varlingam_interventions`:
  the same results;
- `<name>_varlingam_interventions.duckdb::varlingam_intervention_summary`:
  spatial summaries over stable pixels;
- `<name>_varlingam_intervention_plots/`: publication-oriented PNG and PDF
  trajectory plots.

Both commands deliberately ignore nonzero YAML `shift` values, matching
VAR-LiNGAM graph discovery. They require the residualized time-series
database as well as the VAR graph database because scaling, intervention
values, and event counterfactuals depend on the observed per-pixel series.

These computations are model-based causal estimates. Bootstrap intervals
describe graph-estimation uncertainty conditional on the model and data
pipeline; they do not account for residualization uncertainty, measurement
error, omitted causes, spatial dependence between pixels, or extrapolation
beyond the observed intervention range.

See the
[official VAR-LiNGAM tutorial](https://lingam.readthedocs.io/en/latest/tutorial/var.html)
and [Hyvärinen et al. (2010)](https://www.jmlr.org/papers/v11/hyvarinen10a.html)
for the underlying model.
