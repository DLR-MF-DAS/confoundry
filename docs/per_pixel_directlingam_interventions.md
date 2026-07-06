# Per-pixel DirectLiNGAM interventions

Run per-pixel interventions, counterfactuals, mechanism changes, and
goal-seeking analyses for saved DirectLiNGAM structural causal models.

This module is a post-processing companion to
`per_pixel_graph_discovery.py` and
`per_pixel_directlingam_analysis.py`. It reads the shifted per-pixel time
series and graph tables configured by a Confoundry experiment YAML file,
evaluates one or more causal scenarios for every available pixel, propagates
the saved DirectLiNGAM bootstrap adjacency matrices, and writes tabular and
optional raster-like map outputs.

The command-line entry point is normally invoked as:

```bash
python -m confoundry.per_pixel_directlingam_interventions [OPTIONS]
```

The script supports three scenario components:

- hard interventions, such as `do(soil_moisture=0.30)`;
- mechanism interventions, such as halving a fitted edge coefficient;
- goal seeking, such as solving for the soil-moisture value required for
  NDVI to reach a specified target.

Scenario components with the same `SCENARIO` name are evaluated together.
Several distinct scenarios can be supplied in one invocation.

## Mathematical model

For each pixel, the fitted DirectLiNGAM adjacency matrix is interpreted as
the centered linear structural causal model


$$
z = Bz + e,
$$

where `z = x - mean(x)` is the vector of variables centered by their
per-pixel complete-case sample means, `B[child, parent]` is the fitted
direct causal coefficient, and `e` is the vector of exogenous
disturbances.

A hard intervention on variables indexed by `J` replaces their structural
equations with fixed values. For the remaining variables `R`, the script
solves


$$
z_R = (I - B_{RR})^{-1}
      \left(e_R + B_{RJ}z_J\right).
$$

The solve is performed independently for the selected point adjacency
matrix and for every saved bootstrap adjacency matrix that can be evaluated
successfully.

## Analysis modes

### `counterfactual`

Perform observation-specific abduction--action--prediction. For each
selected factual event observation, the disturbance vector is inferred
under the original adjacency matrix as


$$
e = z_{factual} - B_{original}z_{factual}.
$$

The requested mechanism and hard interventions are then applied while
preserving that inferred disturbance vector. This mode answers questions
of the form "What would the target have been for this particular
observation if the intervention had occurred?"

### `interventional_mean`

Set exogenous disturbances to zero and evaluate the intervention around
the per-pixel sample mean. This mode estimates a population-style mean
response under the fitted linear SCM rather than a counterfactual for a
particular factual observation.

Without `--event-filter`, one population context is evaluated. With
event filters, the selected rows can provide climatology contexts, but
the factual vector used by the structural solve remains the per-pixel
mean.

## Required input data

The experiment YAML must identify the time-series and graph DuckDB files,
their tables, the configured variables, pixel coordinate columns, and
ordering columns. Paths are interpreted by the shared Confoundry analysis
loader.

The graph table must contain, for every pixel:

- variable names;
- a raw adjacency matrix;
- a consensus adjacency matrix;
- an edge-probability matrix;
- the complete saved bootstrap adjacency matrices.

An edge-probability matrix alone is not sufficient for uncertainty
propagation. Supported bootstrap JSON fields are determined by
`per_pixel_directlingam_analysis.py` and include names such as
`adjacency_bootstrap_json`.

All graph variables required by a scenario and all requested targets must
be present in the graph's variable list. Before analysis, time-series rows
with missing values in any graph variable are removed. The resulting
complete-case sample must contain at least `--min-samples` rows.

## Command-line parameters

### `config_path`

**Type:** `path`

Existing Confoundry experiment YAML. This option is required and may be
written as `-c PATH` or `--config-path PATH`.

### `targets_raw`

**Type:** `str`

Target variable or variables. Supply `--target` repeatedly or pass a
comma-separated list. At least one target is required. Duplicate target
names are removed while preserving first occurrence.

### `mode`

**Type:** `{"counterfactual", "interventional_mean"}, default="counterfactual"`

Causal evaluation mode.

### `intervention`

**Type:** `tuple of (scenario, variable, value_spec), optional`

Repeatable hard-intervention definition:

```bash
--intervention SCENARIO VARIABLE SPEC
```

All hard interventions sharing a scenario name are applied
simultaneously. A variable cannot be hard-intervened more than once
within the same scenario.

### `mechanism`

**Type:** `tuple of (scenario, edge, mechanism_spec), optional`

Repeatable structural-coefficient intervention:

```bash
--mechanism SCENARIO PARENT->CHILD OPERATION:VALUE
```

The reverse notation `CHILD<-PARENT` is also accepted. Supported
operations are `scale`, `set`, and `add`.

### `goal_seek`

**Type:** `tuple of (scenario, variable, target, goal_spec), optional`

Repeatable goal-seeking definition:

```bash
--goal-seek SCENARIO VARIABLE TARGET GOAL
```

The script solves jointly for the hard-intervention value of `VARIABLE`
required to make `TARGET` equal `GOAL`. Within one scenario, goal
variables must be unique, goal targets must be unique, and a variable
cannot be both explicitly hard-intervened and goal-seeked.

### `event_filter`

**Type:** `str, optional`

Repeatable filter selecting factual or contextual observations. Filters
are combined with logical AND. Supported operators are `=`, `==`,
`!=`, `<`, `<=`, `>`, and `>=`.

Examples:

```bash
--event-filter year=2022
--event-filter month>=6
--event-filter season='summer'
```

Values are parsed as integers, floats, booleans, null-like values, or
strings. `none`, `null`, and `nan` select missing values for
equality and non-missing values for inequality.

### `reference_filter`

**Type:** `str, optional`

Repeatable filter defining the reference pool used by `mean`,
`median`, `quantile`, climatology-based specifications, and
`zdelta`. Filters are combined with logical AND.

`qdelta` is an important exception: its quantile range is calculated
from the full complete-case per-pixel series, not from the
reference-filtered pool.

### `climatology_by`

**Type:** `str, default="month"`

Comma-separated columns used to match climatology reference observations
to the current context. For example, `month` compares an August event
only with August rows in the already reference-filtered pool.

Pass an empty string to disable grouping. In that case,
`climatology_*` specifications reduce to statistics over the complete
reference pool.

### `event_aggregation`

**Type:** `{"none", "mean", "median", "sum"}, default="mean"`

Controls how multiple selected event observations are represented.

#### `none`

Produce one result row per selected context and pixel. The
`event_unit` column contains a JSON identifier made from available
order columns and an event ordinal. Built-in maps are skipped when
several rows exist for the same pixel, scenario, and target.

#### `mean`, `median`, `sum`

Aggregate factual values, counterfactual values, target changes,
    contributions, goal solutions, and bootstrap values across selected
    contexts using the requested operation.

`sum` is mathematically supported but should be used only when summing
the underlying target values is scientifically meaningful.

### `point_matrix`

**Type:** `{"raw", "consensus", "bootstrap_mean"}, default="consensus"`

Matrix used for the reported point estimate, decompositions, and paths.

#### `raw`

Original DirectLiNGAM adjacency matrix.

#### `consensus`

Consensus adjacency matrix generated by graph discovery.

#### `bootstrap_mean`

Elementwise mean of all saved bootstrap adjacency matrices.

Bootstrap uncertainty is always evaluated from the individual saved
bootstrap matrices, independently of this option.

### `low_quantile`

**Type:** `float, default=0.10`

Lower quantile used to define each target's robust scale and the
`qdelta` value specification.

### `high_quantile`

**Type:** `float, default=0.90`

Upper quantile used to define each target's robust scale and the
`qdelta` value specification. The script requires

`0 <= low_quantile < high_quantile <= 1`.

### `min_samples`

**Type:** `int, default=5`

Minimum number of complete-case rows required for a pixel.

### `ci`

**Type:** `float, default=0.95`

Central bootstrap confidence-interval mass. Must lie strictly between
zero and one.

### `allow_new_edges`

**Type:** `bool, default=False`

Permit a mechanism `set` or `add` operation to turn a zero
coefficient into a nonzero edge. Without this flag, attempts to create a
new edge are recorded as scenario errors.

### `top_paths`

**Type:** `int, default=5`

Maximum number of point-estimate intervention paths retained per target.
Set to zero to disable path decomposition.

### `min_path_abs_coefficient`

**Type:** `float, default=0.0`

Exclude point-matrix edges whose absolute coefficient is not greater
than this threshold during path enumeration.

### `max_paths_per_pair`

**Type:** `int, default=5000`

Stop enumerating simple paths for a source--target pair after this many
paths. Because enumeration stops before sorting, this limit is a
computational safeguard and does not guarantee that omitted paths have
smaller contributions than retained paths.

### `output_csv`

**Type:** `path, optional`

Main result CSV. Relative paths are resolved against the directory
containing the experiment YAML. The default is
`<location>_directlingam_interventions.csv`.

### `components_csv`

**Type:** `path, optional`

Long-format contribution and path CSV. Relative paths are resolved
against the experiment directory. The default is
`<location>_directlingam_intervention_components.csv`.

### `output_db`

**Type:** `path, optional`

DuckDB receiving the main and component tables. Relative paths are
resolved against the experiment directory. The default is
`<location>_directlingam_interventions.duckdb`.

### `output_table`

**Type:** `str, default="pixel_directlingam_interventions"`

Name of the main output table in `output_db`.

### `components_table`

**Type:** `str, default="pixel_directlingam_intervention_components"`

Name of the component output table in `output_db`.

### `plot_dir`

**Type:** `path, optional`

Directory for automatically generated PNG maps. Relative paths are
resolved against the experiment directory. The default is
`<location>_directlingam_intervention_plots`.

### `no_plots`

**Type:** `bool, default=False`

Disable built-in map generation.

### `plots_only`

**Type:** `bool, default=False`

Skip causal recomputation, read the existing main CSV, and regenerate
maps. Cannot be combined with `--no-plots`. The component CSV is read
if present but is not required for the built-in maps.

### `figure_width`

**Type:** `float, default=8.0`

Built-in map width in inches.

### `figure_height`

**Type:** `float, default=8.0`

Built-in map height in inches.

### `plot_dpi`

**Type:** `int, default=600`

PNG resolution. Minimum accepted value is 72.

### `show_title`

**Type:** `bool, default=True`

Controlled by `--title` and `--no-title`. Include or omit titles in
built-in maps.

### `show`

**Type:** `bool, default=False`

Display figures interactively in addition to writing PNG files.

### `no_progress`

**Type:** `bool, default=False`

Disable progress bars and the initial loading message.

### `jobs`

**Type:** `int, default=max(1, cpu_count - 1)`

Number of worker processes. `-j 1` evaluates pixels serially and is
recommended for debugging and memory-constrained jobs.

The current implementation materializes all pixel bundles and task
tuples before evaluation. When `jobs > 1`, it also submits all tasks to
a `ProcessPoolExecutor` at once. Large analyses may therefore require
substantial memory.

### `chunksize`

**Type:** `int, default=1`

Accepted for command-line compatibility with other per-pixel scripts.
The current implementation immediately discards this value; it does not
control task batching.

## Hard-intervention value specifications

A hard intervention or goal may use any of the following specifications.
All numeric values are interpreted in the units of the shifted variable
actually loaded from the experiment configuration.

### `NUMBER`

Numeric shorthand for `fixed:NUMBER`.

### `fixed:VALUE`

Set the variable to the absolute value `VALUE`. Aliases:
`value:VALUE` and `set:VALUE`.

### `mean`

Set the variable to its mean in the reference-filtered pool.

### `median`

Set the variable to its median in the reference-filtered pool.

### `quantile:Q`

Set the variable to quantile `Q` of the reference-filtered pool, where
`0 <= Q <= 1`. Alias: `q:Q`.

### `climatology_mean`

Set the variable to the mean of the reference-filtered rows matching the
current context on every `--climatology-by` column. Alias:
`clim_mean`.

### `climatology_median`

Set the variable to the median of the matching climatology subset.
Alias: `clim_median`.

### `climatology_quantile:Q`

Set the variable to quantile `Q` of the matching climatology subset.
Alias: `clim_q:Q`.

### `delta:D`

Set the variable to `base_value + D`. In counterfactual mode,
`base_value` is the factual event value. In interventional-mean mode,
it is the complete-case per-pixel mean.

### `scale:F`

Set the variable to `base_value * F`.

### `qdelta:F`

Set the variable to


$$
base\_value + F\left(Q_{high} - Q_{low}\right),
$$

where the quantiles are calculated from the full complete-case
per-pixel series using `--low-quantile` and `--high-quantile`.
Unlike `quantile` and `zdelta`, this range is not calculated from
the reference-filtered pool.

### `zdelta:F`

Set the variable to `base_value + F * standard_deviation`, where the
sample standard deviation uses `ddof=1` and is calculated from the
reference-filtered pool.

### `fraction_to:F:REFERENCE`

Move fraction `F` of the way from `base_value` to another supported
    value specification:


$$
value = base\_value + F(reference - base\_value).

For example, `fraction_to:0.5:quantile:0.90` moves halfway toward the
reference-pool 90th percentile. Fractions are not restricted to
`[0, 1]`; values outside that range extrapolate.
$$

No value specification automatically clips to an observed range, field
capacity, saturation, a nonnegative bound, or any other physical limit.

## Mechanism specifications

Mechanism interventions modify adjacency coefficients before solving the
SCM. For an edge `PARENT->CHILD`, the stored coefficient is
`B[child, parent]`.

### `scale:F`

Replace the coefficient `b` by `F * b`.

### `set:V`

Replace the coefficient by `V`.

### `add:D`

Replace the coefficient by `b + D`.

Mechanism interventions are applied to the selected point matrix and
separately to every bootstrap matrix. Self loops are forbidden. Creating a
nonzero edge from a fitted zero requires `--allow-new-edges`.

The script does not independently enforce acyclicity, stability, or
scientific plausibility after mechanism modification. A scenario fails for
a matrix if the required linear system cannot be solved or produces
non-finite values.

## Goal seeking

Each goal-seeking entry specifies an intervention variable, a target
variable, and a desired target value. For multiple entries sharing the same
scenario name, the script solves the joint linear system


$$
K u = y_{goal} - y_{base},
$$

where each column of `K` is the response of all goal targets to a unit
hard intervention on one goal variable, conditional on any explicitly fixed
hard interventions and mechanism changes.

The response matrix must be square and nonsingular. A singular or
non-finite solve is recorded as an error for that scenario and pixel.
Required intervention values are solved independently for the point matrix
and for each bootstrap matrix.

## Output files

The command writes a main CSV, a component CSV, and two DuckDB tables unless
`--plots-only` is used. It may also write PNG maps.

## Main result columns

The main output contains pixel coordinate columns from `row_col_cols` plus
the following fields for successful rows.

### `scenario`

Scenario name.

### `mode`

Selected causal mode.

### `target`

Target variable.

### `event_unit`

Aggregation label or a JSON event identifier when aggregation is
`none`.

### `event_aggregation`

Requested event aggregation.

### `n_samples`

Number of complete-case time-series rows for the pixel.

### `n_event_observations`

Number of selected contexts represented by the row.

### `point_matrix`

Matrix used for the point estimate.

### `low_quantile`, `high_quantile`

Quantile settings used for robust target scaling.

### `target_delta_qhi_qlo`

Per-pixel complete-case target range
`Q_high(target) - Q_low(target)`.

### `interventions_json`

JSON mapping hard-intervention variables to normalized value
specifications.

### `mechanisms_json`

JSON mapping modified edges to mechanism specifications.

### `goals_json`

JSON mapping `VARIABLE->TARGET` pairs to normalized goal
specifications.

### `factual_value`

Point factual target value after event aggregation. In
interventional-mean mode this is the per-pixel target mean.

### `counterfactual_value`

Point predicted target value after all scenario components.

### `target_change`

`counterfactual_value - factual_value`.

### `scaled_target_change`

`target_change / target_delta_qhi_qlo`.

### `mechanism_target_contribution`

Change in the target caused by mechanism modifications alone, before
hard interventions and goal-seek interventions.

### `hard_target_contributions_json`

Exact linear decomposition of the target change attributable to each
explicit and goal-seek hard intervention, relative to the
mechanism-only state.

### `required_intervention_values_json`

Goal-seek intervention values solved from the point matrix.

### `top_paths_json`

Point-matrix path contributions for hard and goal-seek intervention
variables after cutting incoming edges into all hard-intervened nodes.

### `n_bootstrap_total`

Number of saved bootstrap adjacency matrices.

### `n_bootstrap_successful`

Number of bootstrap matrices for which the entire scenario succeeded
for all selected contexts.

### `n_bootstrap_failed`

Number of bootstrap matrices discarded because any scenario/context
evaluation failed.

## Bootstrap summary columns

Each of the following prefixes is expanded with a standard bootstrap
summary:

- `counterfactual_value`;
- `target_change`;
- `scaled_target_change`;
- `mechanism_target_contribution`.

For each prefix `P`, the output includes:

### `P_boot_mean`

Arithmetic mean over finite successful bootstrap values.

### `P_boot_median`

Median over finite successful bootstrap values.

### `P_boot_sd`

Sample standard deviation with `ddof=1`; missing when fewer than two
values are available.

### `P_boot_ci_low`, `P_boot_ci_high`

Central interval bounds determined by `--ci`.

### `P_boot_ci_width`

Upper minus lower bound.

### `P_boot_prob_gt_zero`

Fraction of finite bootstrap values greater than zero.

### `P_boot_prob_lt_zero`

Fraction of finite bootstrap values less than zero.

### `P_boot_prob_excludes_zero`

Boolean indicating whether the central interval lies entirely above or
below zero.

### `P_n_bootstrap_successful`

Number of finite values entering that particular summary.

The main row also contains an `error` column. Failed pixels or scenarios
may produce shorter rows containing coordinates, diagnostic metadata, and
the error representation rather than all successful-result fields.

## Component result columns

The component CSV and DuckDB table contain pixel coordinates, `scenario`,
`mode`, `target`, `event_unit`, `component_type`, `component`,
`point_value`, optional path fields, bootstrap summaries where
applicable, and `error`.

Possible `component_type` values are:

### `hard_intervention_target_contribution`

Exact point and bootstrap contribution of one hard or goal-seek
intervention to the selected target.

### `mechanism_target_contribution`

Combined contribution of all mechanism changes.

### `goal_required_intervention_value`

Point and bootstrap distribution of a solved goal-seek intervention
value.

### `point_path_contribution`

Point-matrix contribution of one enumerated simple path. Path rows
include `coefficient_product` and `source_change` but no bootstrap
path uncertainty.

## Built-in maps

When plotting is enabled and each pixel has at most one successful row per
scenario and target, the script writes exactly these map types when the
corresponding values are available:

### `target_change`

Diverging map with symmetric limits based on the 98th percentile of
absolute finite values.

### `scaled_target_change`

Diverging map with the same limit rule.

### `target_change_boot_prob_gt_zero`

Sequential map from zero to one.

Files are named approximately as:

```text
<scenario>__<target>__<column>.png
```

The plotting routine treats the configured row and column identifiers as a
regular image grid. It does not georeference the PNG, draw administrative
boundaries, add a map projection, or automatically plot confidence
interval bounds or widths.

Maps are skipped when `event_aggregation='none'` produces duplicate
pixel/scenario/target rows. Use `mean`, `median`, or `sum` to create
one row per combination, or create custom maps from the CSV.

## Contribution interpretation

For simultaneous hard interventions, the script computes an exact
source-wise linear decomposition relative to the mechanism-only state. If
`J` denotes the intervened variables and `R` the remaining variables,
the response matrix is


$$
H = (I - B_{RR})^{-1}B_{RJ}.
$$

Each source contribution is its intervention change multiplied by the
corresponding column of `H`, with the source's own change placed at its
intervened coordinate. Contributions sum linearly to the hard-intervention
part of the scenario response, subject to numerical precision.

Mechanism contribution is kept separate. Consequently,


$$
total\ target\ change =
mechanism\ contribution +
\sum_j hard\ contribution_j
$$

for a successfully solved linear scenario, up to floating-point error.

## Path interpretation

Path decomposition is calculated only from the selected point matrix.
Before path enumeration, incoming edges into all explicit and goal-seek
hard-intervention nodes are removed. For a simple source-to-target path, the
reported contribution is


$$
source\ change \times \prod_{edge \in path} coefficient_{edge}.
$$

The path list is explanatory rather than a separate causal estimator.
Individual path contributions may cancel, and the list may be incomplete
because of coefficient thresholds, `--top-paths`, or
`--max-paths-per-pair`.

## Returns

### `None`

Results are written to CSV, DuckDB, and optionally PNG files. A concise
summary of modes, targets, scenarios, output paths, failed rows, and
written plots is printed to standard output.

## Raises

### `click.BadParameter`

Raised for malformed CLI values, invalid quantile ordering, missing
files or columns, unsupported matrix choices, absent bootstrap matrices,
or invalid filters and scenario syntax.

### `click.UsageError`

Raised when no scenario is defined, when incompatible flags are
combined, or when scenario components violate uniqueness and overlap
constraints.

### `click.ClickException`

Raised when `--plots-only` cannot find the main CSV or when no
intervention rows are produced.

### `numpy.linalg.LinAlgError`

Normally converted into per-pixel scenario errors when a structural or
goal-seeking linear system is singular.

### `concurrent.futures.process.BrokenProcessPool`

May surface when a multiprocessing worker is terminated abruptly, for
example by an out-of-memory kill or a native-library crash.

## Notes

The analysis inherits the assumptions of the fitted DirectLiNGAM models:
linearity, acyclicity of the learned SCM, suitable non-Gaussian independent
disturbances, correct temporal alignment, and adequate control of common
causes.

Counterfactual validity additionally requires that the structural
mechanisms are stable under the requested intervention. Large values,
mechanism edits, or unusual combinations of variables may extrapolate far
outside the observed support even when the linear solve succeeds.

All per-pixel means and robust target scaling ranges are calculated from
the complete-case series after configured shifts. Reference filters do not
change the centering means.

The script does not simulate physical processes such as infiltration,
runoff, drainage, crop management, energy balance, or conservation laws
unless those mechanisms are already represented by variables and edges in
the fitted SCM.

The script does not clip hard interventions. Users must impose or
post-filter domain bounds such as field capacity, saturation, feasible
temperature ranges, or nonnegative concentrations.

Bootstrap intervals describe sensitivity to the saved bootstrap graph
estimates. They do not automatically include uncertainty from measurement
error, interpolation, reanalysis bias, intervention implementation, omitted
variables, or model-class misspecification.

## Examples

Display the command-line help:

```bash
python -m confoundry.per_pixel_directlingam_interventions --help
```

Evaluate a fixed observation-specific hard intervention for a known event:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  --config-path data/experiment.yaml \
  --target ndvi \
  --mode counterfactual \
  --intervention wet_soil soil_moisture fixed:0.30 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --event-aggregation mean \
  --point-matrix consensus \
  --output-csv august_2022_wet_soil.csv \
  --output-db august_2022_wet_soil.duckdb \
  --plot-dir august_2022_wet_soil_maps \
  -j 1
```

A bare numeric value is equivalent to `fixed:`:

```bash
--intervention wet_soil soil_moisture 0.30
```

Increase one variable by an amount in its loaded data units:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --intervention plus_005 soil_moisture delta:0.05 \
  --event-filter year=2022 \
  --event-filter month=8 \
  -j 1
```

Apply simultaneous hard interventions by reusing the same scenario name:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --intervention joint_wetting shallow_soil_moisture delta:0.05 \
  --intervention joint_wetting deep_soil_moisture delta:0.05 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --event-aggregation mean \
  -j 1
```

For unscaled ERA5 volumetric soil-water variables, a delta is in
`m3 m-3`. The following example tests three hypothetical increases in the
mean 7--100 cm root-zone water state. The physical interpretation and bounds
must be checked separately:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/iberian_drought_experiment/experiment.yaml \
  --target ndvi \
  --mode counterfactual \
  --intervention storage_20mm soil_moisture_7_to_28_cm delta:0.021505376 \
  --intervention storage_20mm soil_moisture_28_to_100_cm delta:0.021505376 \
  --intervention storage_40mm soil_moisture_7_to_28_cm delta:0.043010753 \
  --intervention storage_40mm soil_moisture_28_to_100_cm delta:0.043010753 \
  --intervention storage_60mm soil_moisture_7_to_28_cm delta:0.064516129 \
  --intervention storage_60mm soil_moisture_28_to_100_cm delta:0.064516129 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --event-aggregation mean \
  --point-matrix consensus \
  --ci 0.95 \
  --output-csv iberia_august_2022_storage_response.csv \
  --components-csv iberia_august_2022_storage_components.csv \
  --output-db iberia_august_2022_storage_response.duckdb \
  --plot-dir iberia_august_2022_storage_maps \
  -j 1
```

The nominal water-storage conversion in that example is


$$
storage\ change\ [mm] =
1000 \sum_l depth_l\ [m] \Delta\theta_l.
$$

It assumes raw volumetric fractions and does not mean that the same amount
of surface irrigation would maintain the resulting monthly mean state.

Set shallow and deep soil moisture to the same-month 75th percentile of
reference years:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/iberian_drought_experiment/experiment.yaml \
  --target ndvi \
  --mode counterfactual \
  --intervention august_q75 soil_moisture_7_to_28_cm \
      climatology_quantile:0.75 \
  --intervention august_q75 soil_moisture_28_to_100_cm \
      climatology_quantile:0.75 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --reference-filter year!=2022 \
  --climatology-by month \
  --event-aggregation mean \
  -j 1
```

Set the same variables to annual/reference-period quantiles instead. The
ordinary `quantile` specification ignores `--climatology-by`:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/iberian_drought_experiment/experiment.yaml \
  --target ndvi \
  --intervention annual_q75 soil_moisture_7_to_28_cm quantile:0.75 \
  --intervention annual_q75 soil_moisture_28_to_100_cm quantile:0.75 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --reference-filter year!=2022 \
  -j 1
```

Compare several intervention definitions in one invocation:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --intervention fixed_030 soil_moisture fixed:0.30 \
  --intervention annual_q75 soil_moisture quantile:0.75 \
  --intervention monthly_q75 soil_moisture climatology_quantile:0.75 \
  --intervention plus_005 soil_moisture delta:0.05 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --reference-filter year!=2022 \
  --climatology-by month \
  -j 1
```

Move halfway toward the reference-pool 90th percentile:

```bash
--intervention half_to_q90 soil_moisture \
    fraction_to:0.5:quantile:0.90
```

Move 25 percent toward the same-month climatological median:

```bash
--intervention quarter_to_clim soil_moisture \
    fraction_to:0.25:climatology_median \
--climatology-by month
```

Apply a robust-range increment based on the full complete-case series:

```bash
--low-quantile 0.10 \
--high-quantile 0.90 \
--intervention robust_plus_half soil_moisture qdelta:0.5
```

Apply a one-standard-deviation increment based on the reference-filtered
pool:

```bash
--reference-filter year<2022 \
--intervention plus_one_sd soil_moisture zdelta:1.0
```

Evaluate a population interventional mean without selecting a factual
event:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --mode interventional_mean \
  --intervention population_wet soil_moisture quantile:0.75 \
  --reference-filter year<2022 \
  --event-aggregation mean \
  -j 1
```

Modify an existing causal mechanism by halving the soil-moisture-to-NDVI
edge:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --mode counterfactual \
  --mechanism buffered 'soil_moisture->ndvi' scale:0.5 \
  --event-filter year=2022 \
  --event-filter month=8 \
  -j 1
```

Remove an existing edge by setting it to zero:

```bash
--mechanism no_direct_effect 'soil_moisture->ndvi' set:0
```

Add 0.1 to an existing coefficient:

```bash
--mechanism stronger_effect 'soil_moisture->ndvi' add:0.1
```

Create a new edge that is zero in the fitted matrix:

```bash
--mechanism hypothetical_link 'irrigation->soil_moisture' set:0.5 \
--allow-new-edges
```

Combine mechanism and hard interventions in one scenario:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --mechanism managed 'vpd->ndvi' scale:0.5 \
  --intervention managed soil_moisture delta:0.05 \
  --event-filter year=2022 \
  --event-filter month=8 \
  -j 1
```

Solve for the soil-moisture intervention required to make NDVI reach the
reference-pool 75th percentile:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --mode counterfactual \
  --goal-seek ndvi_goal soil_moisture ndvi quantile:0.75 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --reference-filter year!=2022 \
  -j 1
```

Solve jointly for two intervention variables and two unique targets:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi,lst \
  --mode counterfactual \
  --goal-seek joint_goal soil_moisture ndvi quantile:0.75 \
  --goal-seek joint_goal albedo lst quantile:0.40 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --reference-filter year!=2022 \
  -j 1
```

Combine one fixed intervention with a separate goal-seek variable:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --intervention constrained_goal irrigation fixed:1.0 \
  --goal-seek constrained_goal soil_moisture ndvi fixed:0.65 \
  --event-filter year=2022 \
  --event-filter month=8 \
  -j 1
```

Analyze several target variables. Repeated and comma-separated forms may be
mixed:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi,lst \
  --target evapotranspiration \
  --intervention wet soil_moisture delta:0.05 \
  --event-filter year=2022 \
  --event-filter month=8 \
  -j 1
```

Retain one row per selected event rather than aggregating. Built-in maps
will normally be skipped because several rows exist per pixel:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --intervention wet soil_moisture delta:0.05 \
  --event-filter year=2022 \
  --event-filter month>=6 \
  --event-filter month<=8 \
  --event-aggregation none \
  --no-plots \
  -j 1
```

Aggregate a selected event window by its median:

```bash
--event-filter year=2022 \
--event-filter month>=6 \
--event-filter month<=8 \
--event-aggregation median
```

Write only tabular outputs:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --intervention wet soil_moisture delta:0.05 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --no-plots \
  -j 1
```

Regenerate the three built-in maps from an existing main CSV:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --intervention wet soil_moisture delta:0.05 \
  --output-csv existing_results.csv \
  --plot-dir regenerated_maps \
  --plots-only
```

Scenario definitions are still required in `--plots-only` mode because
the CLI constructs and reports scenarios before loading the existing CSV,
even though the scenarios are not recomputed.

Use custom DuckDB table names:

```bash
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --intervention wet soil_moisture delta:0.05 \
  --event-filter year=2022 \
  --event-filter month=8 \
  --output-db intervention_results.duckdb \
  --output-table main_results \
  --components-table component_results \
  -j 1
```

Use conservative settings on a memory-constrained HPC node:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
python -m confoundry.per_pixel_directlingam_interventions \
  -c data/experiment.yaml \
  --target ndvi \
  --intervention wet soil_moisture delta:0.05 \
  --event-filter year=2022 \
  --event-filter month=8 \
  -j 1
```

Inspect failed rows after a run:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv(
    "data/experiment/location_directlingam_interventions.csv"
)
print(df.loc[df["error"].notna(), ["row", "col", "scenario", "error"]])
PY
```

Inspect bootstrap uncertainty columns:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("interventions.csv")
columns = [
    c for c in df.columns
    if c.startswith("target_change_boot")
]
print(df[columns].describe(include="all"))
PY
```

Select robust positive-response pixels:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("interventions.csv")
candidates = df[
    df["error"].isna()
    & (df["target_change"] > 0)
    & (df["target_change_boot_ci_low"] > 0)
    & (df["target_change_boot_prob_gt_zero"] >= 0.95)
]
candidates.to_csv("robust_positive_candidates.csv", index=False)
PY
```

Create a confidence-interval-width map from the main CSV. This is not one
of the built-in maps:

```bash
python - <<'PY'
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("interventions.csv")
df = df[df["error"].isna()]
scenario = "wet"
target = "ndvi"
group = df[
    (df["scenario"] == scenario)
    & (df["target"] == target)
]
grid = group.pivot(
    index="row",
    columns="col",
    values="target_change_boot_ci_width",
).sort_index().sort_index(axis=1)

fig, ax = plt.subplots(figsize=(8, 8))
image = ax.imshow(grid.to_numpy(), origin="upper")
ax.set_axis_off()
fig.colorbar(image, ax=ax)
Path("custom_maps").mkdir(exist_ok=True)
fig.savefig(
    "custom_maps/wet__ndvi__target_change_boot_ci_width.png",
    dpi=300,
    bbox_inches="tight",
)
PY
```

## See Also

### `confoundry.per_pixel_graph_discovery`

Fit and bootstrap per-pixel DirectLiNGAM graphs.

### `confoundry.per_pixel_directlingam_analysis`

Estimate direct, total, scaled, path, and dominance effects from saved
adjacency matrices.

### `numpy.linalg.solve`

Linear-system solver used for structural and goal-seeking equations.

### `concurrent.futures.ProcessPoolExecutor`

Multiprocessing backend used when `--jobs` is greater than one.
