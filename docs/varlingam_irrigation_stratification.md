# Irrigation exposure and model diagnostics

`confoundry.varlingam_irrigation_stratification` tests whether externally mapped
irrigation infrastructure is associated with QC outcomes, remaining dependence,
and saved precipitation/soil-moisture responses in cropland. It reads the saved
models' results and does not refit or change them.

The primary question is **whether QC pass fraction decreases with increasing
irrigation exposure**. Such an association would support the plausibility of
an omitted-management explanation for some diagnostic failures. It would not
show that irrigation caused those failures, or validate the inferred causal
graphs.

## Run with the production directory

Run from the repository root, using your Confoundry Python environment:

```bash
EXP=data/iberian_drought_experiment
RUN="$EXP/paper_var3_prod_v1"

PYTHONPATH=src python -m confoundry.varlingam_irrigation_stratification \
  --effects-db "$RUN/varlingam_effects.duckdb" \
  --effects-table pixel_varlingam_effects \
  --qc-csv "$RUN/pixel_qc.csv" \
  --source-db "$EXP/iberian_droughts_source_db.duckdb" \
  --download-irrigation \
  --download-worldcover \
  --aridity-pixels "$RUN/aridity_stratification/pixel_aridity_effects.csv" \
  --output-dir "$RUN/irrigation_stratification"
```

Use `.venv/bin/python` instead of `python` if that is your environment.
`--source-db` locates an accessible original `modis_ndvi` raster in
`geotiff_catalog`, ordered by year/month. Change `--reference-variable` if your
source variable has a different name. Alternatively, replace `--source-db` with
`--reference-raster /path/to/original/ndvi.tif`. It must have the **same CRS,
transform, dimensions and row/column origin** used when gathering the series.
An independently cropped raster is unsuitable if it changed the indices.

`--aridity-pixels` reuses the **all-fitted** pixel audit table from the aridity
experiment. If you saved it elsewhere, change that path. Omit this option to
run unadjusted analyses only. Aridity coordinates, when present, are checked
against the reference grid. Missing AI never removes pixels from the main
unadjusted diagnostic analysis; a matched unadjusted analysis is also produced
for the same pixels used in the adjusted tests.

The first run downloads the small GMIA archive and the WorldCover tiles
intersecting the study pixels. WorldCover tiles are native **10 m** data and
can total several gigabytes. They are cached in `data/external/irrigation`.
Processing averages binary crop coverage in bounded windows, without loading
all of Iberia at 10 m into memory. Subsequent runs verify and reuse downloads.
Local SHA256 hashes detect changed cache files; they are not publisher-supplied
authenticity checksums. Output directories must be new or empty.

### Local rasters or a broader cropland mask

WorldCover class 40 mainly covers annual herbaceous crops. Perennial woody
crops, including many orchards and olive groves, fall under tree/shrub classes.
Thus a WorldCover-based experiment **does not represent all Iberian agriculture**.
Do not label all tree/shrub pixels as crops to compensate. For broader coverage,
supply a land-use mask that actually identifies permanent crops.

Replace `--download-worldcover` with, for example:

```bash
--cropland-raster /path/to/cropland_fraction.tif \
--cropland-kind fraction \
--cropland-scale 1 \
--cropland-description 'Product name, version, year and citation; includes permanent crops'
```

For percentages use scale `0.01`. For categorical rasters, use
`--cropland-kind categorical --cropland-classes 40` or your product's explicit
crop codes, separated by commas. Codes are classified **before** aggregation;
the script never averages class identifiers. Repeat `--cropland-raster` for
disjoint tiles sharing a CRS, or supply one mosaic/VRT. Overlapping tile
footprints are rejected. Nodata must be correctly encoded in the raster mask.

Replace `--download-irrigation` with
`--irrigation-raster /path/to/gmia_v5_aei_pct.asc` for a local GMIA v5 percentage
grid. Default `--irrigation-kind gmia` converts percentages and applies the
special FAO convention described below. For another irrigation product use
`--irrigation-kind fraction --irrigation-scale 1` (or `0.01` for percentages)
and give an accurate `--irrigation-description`. In fraction mode, ordinary
nodata remains missing; no special GMIA assumptions apply.

## External data and alignment

The module downloads the **area equipped for irrigation, percentage of total
grid-cell area** layer from [FAO GMIA version 5](https://www.fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas/latest-version/).
Its reference date is approximately 2005 and its resolution is 5 arcminutes.
It is not actual irrigation volume, timing, or the fraction irrigated during
the analysis years. The fraction is not divided by the cropland mask; these
products have different dates, resolutions, and classification definitions.

**GMIA's `-9` is special:** FAO documents these cells as having no irrigation,
despite encoding them as nodata. GMIA mode explicitly converts `-9` to zero
inside the raster's bounds. Dropping them would remove the comparison group.
Cells outside the raster remain missing. GMIA ASCII inputs lacking CRS metadata
are assigned the documented EPSG:4326. The download is checked against the
documented global grid dimensions and extent.

The optional cropland source is
[ESA WorldCover 2021 v200](https://esa-worldcover.org/en/data-access), with crop
class 40 as defined in the
[product manual](https://worldcover2021.esa.int/data/docs/WorldCover_PUM_V2.0.pdf).
GMIA, WorldCover and the observational record have different temporal support;
this experiment does not track changes in irrigation or crop extent over time.

Native irrigation fractions and native binary crop masks are averaged over
the original analysis footprints using GDAL overlap weights. Weights follow
the raster coordinate system; geographic-grid averages are not exact geodesic
area integrals. At these cell sizes this is a local approximation. Each layer
requires at least 50% valid footprint coverage by default. Fractions refer to
the valid portion, and valid coverage is exported separately. Alignment of
coarse irrigation data onto the NDVI grid does not create finer information.

The default population requires crop fraction at least `0.5`, configurable
with `--min-cropland-fraction`. Fixed irrigation bins, defined without examining
QC, are:

| Class | Equipped fraction of total grid-cell area |
| --- | --- |
| Little/no | `<0.01` |
| Moderate | `0.01` to `<0.20` |
| High | `>=0.20` |

These are study-specific operational choices, not FAO categories. Change them
with `--irrigation-cutpoints 0.01,0.20`. Keep the continuous association as the
primary analysis; do not tune cutpoints to obtain a preferred result. Crop
fraction cutoffs can also be varied in separate output directories.

## Inputs, endpoints and populations

The **complete** `pixel_qc.csv` from `prepare_varlingam_paper_outputs` provides
the diagnostics. A separate diagnostic database is not necessary. The module
requires the primary eligibility flag, point stability, bootstrap whiteness
status and p-value, and structural/innovation correlation screens. It verifies
booleans, ranges, unique pixel coordinates, and basic consistency of QC passes.

| Endpoint | Summary and interpretation | Population |
| --- | --- | --- |
| QC pass | Mean of saved eligibility flags; hypothesized to decrease | All fitted eligible cropland |
| Whiteness rejection | Fraction with valid bootstrap `p<0.05`; hypothesized to increase | All fitted eligible cropland with valid calibration |
| Structural-error dependence | Median maximum absolute same-month correlation | All fitted eligible cropland with a finite diagnostic |
| Lag-1 dependence | Median saved maximum absolute lag-1 innovation autocorrelation | Same |
| Cross-lag dependence | Median saved maximum absolute innovation correlation through the diagnostic lag range | Same |
| Precipitation coupling | Signed cumulative effect and its absolute magnitude; weaker coupling concerns magnitude | Primary QC; all-stable sensitivity |
| Soil-moisture coupling | Signed and absolute cumulative effects for both layers; direction unspecified | Primary QC; all-stable sensitivity |

Invalid whiteness calibrations are **not** rejections. The valid fraction and
p-value distribution are exported descriptively with their own denominators.
A p-value is not a measure of dependence magnitude. QC already contains these
diagnostic screens, so their associations are not independent validations.

Effect input can be `--effects-db` or `--effects-csv`. The selected table uses
`scaled_cumulative_total_effect` at horizon 12 by default: this already sums
months 0–12, and is not summed again. All three water-source effects must be
finite for a pixel to enter either effect population. Pruned zero effects
remain valid. Missing effects are never zero-filled. Use the **all-fitted**
`pixel_varlingam_effects` table to make the stable-all sensitivity informative;
missing stable effects produce an explicit warning and exported counts.

Precipitation magnitude is `abs(CE_precipitation)`, after signed temporal
summation. A smaller value can reflect temporal cancellation, not only uniformly
smaller monthly effects. Decreasing signed effects alone do not demonstrate
weaker coupling. All responses use local source and target quantile scaling,
so they do not compare equal physical pulses between regions.

## Statistical analysis

The default is 100 km blocks in Europe's equal-area EPSG:3035, with 50 and
150 km sensitivity runs. For another region set a suitable metric equal-area
`--block-crs`. Blocks have origin `(0,0)`. Each endpoint uses the same valid
pixels for its exposure, outcome and covariate block summaries.

- Block exposure: **mean** equipped irrigation fraction. Means retain sparse
  irrigated areas that a block median could round to zero.
- Binary outcomes: block pass/rejection fraction. Continuous outcomes: block
  median. Aridity control: block median AI. Crop control: block mean crop
  fraction. Every block receives equal weight in trend inference.
- Unadjusted association: Spearman correlation, with 4,999 two-sided block
  permutations and plus-one p-values. These are associations between block
  summaries, not fitted individual-pixel probabilities or odds ratios.
- Adjusted sensitivity: partial Spearman correlation after regressing the
  ranks of exposure and outcome on ranked AI and crop fraction. The
  Freedman–Lane test permutes reduced-model outcome residuals and residualizes
  them again. Constant controls are omitted and recorded; collinear controls
  or a fully explained exposure suppress inference.
- Confidence intervals: 999 paired block bootstrap samples, reranking and
  refitting the adjustment in each sample. At least 10 blocks and at least 95%
  valid bootstrap samples are required by default. Degenerate cases have an
  explicit status rather than a fabricated p-value or interval.
- Multiplicity: QC pass is the single primary test. Four secondary diagnostic
  tests form one Benjamini–Hochberg family; six signed/magnitude effect tests
  form another. Adjustment is separate for each block size, population and
  adjustment mode. It does not control multiplicity over all sensitivities.

Bin estimates use pixel means/medians. Bin intervals sample whole spatial
blocks **within each fixed irrigation bin** and recompute those estimates.
They are pointwise and conditional on saved estimates, not graph-bootstrap
uncertainty. Exported spatial percentiles describe pixel variation separately.

Blocking does not guarantee exchangeability or remove broad spatial gradients.
Adjusted residual permutation is approximate, and unequal block populations
can produce heterogeneous sampling variability. Check block sizes, coverage,
covariate overlap, and crop thresholds. AI and crop fraction do not account for
all crop types, soils, topography, management or observation-quality differences.

Effects among QC survivors can be affected by selection. Stable-all effects
are supplied to reveal sensitivity to that selection, not to rehabilitate
failed-QC causal interpretations. No result automatically proves that an
omitted irrigation driver explains failure, or that the diagnostics are
specific to omitted anthropogenic inputs.

## Outputs for the paper

| File | Contents |
| --- | --- |
| `irrigation_diagnostics.pdf` / `.png` | Pass/rejection fractions, dependence screens, coverage by irrigation class |
| `irrigation_effects.pdf` / `.png` | Signed and absolute precipitation/soil-moisture effects; primary and stable-all populations |
| `irrigation_associations.pdf` / `.png` | Unadjusted and adjusted block associations with 95% intervals |
| `irrigation_coverage.pdf` / `.png` | Exposure and QC maps for the selected crop population |
| `pixel_irrigation_diagnostics.csv` | All fitted pixels, original diagnostics, covariates, effects, population flags |
| `aligned_irrigation.tif`, `aligned_cropland.tif` | Cropped analysis-grid fractions and valid coverage; original grid offsets in tags |
| `bin_definitions.csv`, `qc_coverage.csv` | Exact bins and distinct population/calibration counts |
| `bin_summaries.csv` | Estimates, spatial spread, block intervals and denominators |
| `trend_tests.csv`, `block_summaries.csv` | All tests, validity status, controls, p-values and underlying block data |
| `irrigation_trends_unadjusted.tex`, `irrigation_trends_adjusted.tex` | Main-block-size tables including intervals; adjusted file requires AI input |
| `irrigation_diagnostics.tex` | Bin diagnostic table, with estimates in fraction/correlation units |
| `report.md`, `references.bib` | Run-specific methods, interpretation guidance, captions, BibTeX |
| `manifest.json` | Parameters, population counts, input/code hashes, grid metadata, software versions |

Figures have vector PDF text and 300 dpi PNG equivalents. LaTeX fragments
require `booktabs`; wrap them in a table environment and add the run's caption.
The diagnostics figure and QC/secondary-diagnostic rows are the central evidence
for the omitted-input hypothesis. Effects are complementary and conditional on
which models survived QC. Report null or opposing patterns as well as agreement.

`--no-plots` runs tables only. `--seed` reproduces resampling, with separate
streams by endpoint and analysis. The tests use constructed rasters and known
statistical patterns; they are not environmental findings. Production data
must be run separately to obtain results for the paper.
