# VAR–LiNGAM responses along an external aridity gradient

`confoundry.varlingam_aridity_stratification` joins saved cumulative effects to
an independently produced annual aridity climatology and produces publication
figures, tables, and spatial inference. It does not refit the causal models or
modify the effect database.

## Run the experiment

Use the environment containing Confoundry, or prefix the commands with
`PYTHONPATH=src` when running from an uninstalled source checkout.

```bash
python -m confoundry.varlingam_aridity_stratification \
  --effects-db data/iberian_drought_experiment/paper_var3_reproduction/varlingam_effects.duckdb \
  --effects-table pixel_varlingam_effects \
  --qc-csv data/iberian_drought_experiment/paper_var3_reproduction/pixel_qc.csv \
  --reference-raster /path/to/the/original/ndvi_reference_grid.tif \
  --download-aridity \
  --output-dir data/iberian_drought_experiment/aridity_stratification
```

The reference raster must have **the exact extent, transform, CRS and dimensions
used to define `row` and `col` in `confoundry.gather`**. Any time slice on that
unchanged grid is suitable; its pixel values are not used. A newly cropped
Iberian raster is unsuitable if cropping changed the original pixel indices.

The command downloads only the annual AI archive from **Global-AI v3**, pinned
to [Figshare release 5](https://doi.org/10.6084/m9.figshare.7504448.v5). It caches
the metadata and archive, verifies the publisher's MD5, and records input SHA256
hashes. A changed filename or ambiguous archive causes an explicit error listing
available files; select the annual AI archive using `--aridity-archive-name`.
It never silently falls back to the monthly product or a newer release.
The initial download requires network access. Subsequent runs use the cache.

Alternatively, supply an already downloaded **annual aridity** GeoTIFF:

```bash
python -m confoundry.varlingam_aridity_stratification \
  --effects-csv /path/to/varlingam_effects.csv \
  --qc-csv /path/to/pixel_qc.csv \
  --reference-raster /path/to/the/original/ndvi_reference_grid.tif \
  --aridity-raster /path/to/ai_v3_yr.tif \
  --aridity-scale 0.0001 \
  --aridity-description 'Global-AI v3; 1970–2000; doi:10.6084/m9.figshare.7504448.v5' \
  --output-dir /path/to/new/aridity_results
```

For a raster already expressed in physical P/PET units use
`--aridity-scale 1`. The multiplier applies to raw raster values: raster scale
metadata is deliberately not applied a second time. Nodata must be encoded in
the raster mask/nodata metadata; negative AI is also treated as invalid. Genuine
zero AI is valid. Specify `--aridity-band` only when an annual AI band has been
identified independently. The code cannot determine whether a user-supplied
file is actually an independent climatology.

The external AI is **P/ET0**, based on WorldClim climatology and FAO reference
evapotranspiration, not the ERA5 potential evaporation series in the causal
model. The 1970–2000 baseline, resolution, and raw-value scale are documented by
[Zomer, Xu and Trabucco (2022)](https://doi.org/10.1038/s41597-022-01493-1).
This provides a separate climatological classification, not an independent
causal validation dataset. Increasing AI means **wetter** conditions.

Both north-up geographic and projected grids are supported. The script reads
the source window intersecting the study extent, reprojects it, and averages
valid AI values over each reference footprint using GDAL overlap weights.
The minimum valid fraction is `0.5`, configurable with
`--min-aridity-coverage`; partial coastal coverage is exported. This is the
average of fine-grid ratios, not a ratio of aggregated precipitation and ET0.

## Inputs and population

- Effects come from `per_pixel_varlingam_analysis`, as DuckDB or CSV. The default
  table is `pixel_varlingam_effects`. A primary-only table also works, but the
  **QC CSV must still contain all fitted pixels** for coverage and bin definition.
- Required columns are `row`, `col`, `source`, `target`, `horizon`, and
  `scaled_cumulative_total_effect`. The target defaults to `ndvi_resid` and
  horizon to `12`. The value at horizon 12 already sums **months 0–12 inclusive**;
  the script does not sum the cumulative column again.
- The five sources are `precipitation_resid`, `temperature_resid`,
  `evaporation_resid`, `soil_moisture_7_to_28_cm_resid`, and
  `soil_moisture_28_to_100_cm_resid`.
- QC comes from `prepare_varlingam_paper_outputs`, using `primary_eligible` by
  default. Use `--qc-column analytical_sensitivity_eligible` for that sensitivity
  population, in a separate output directory.
- Main summaries use the **same complete-case population for all five sources**:
  QC pass, sufficient aridity coverage, and five finite effects. Missing rows
  are not zeros. Pruned zero effects are retained. Individual bootstrap
  significance is not an inclusion condition.
- If present, raw cumulative derivatives, source quantile ranges, and target
  quantile ranges are copied into the pixel audit table. They are not combined
  across variables because their units differ.
- A native ERA5 evaporation series can have negative values for evaporative
  losses. The default preserves the fitted sign. Use `--evaporation-multiplier -1`
  **only if verified appropriate** to express the signed response to greater
  evaporative demand. Absolute-response indices are unaffected. The exported
  raw derivative audit columns always retain the original signs.

## Prespecified summaries

The signed cumulative effect for each source is the primary interpretable
quantity. It measures accumulated NDVI-anomaly response to a one-time source
pulse under the fitted model. Its local scaling compares source and NDVI
quantile ranges; it is not a response to the same number of millimeters or
degrees everywhere. A large magnitude can reflect a large response, persistence,
local source variability, or a small local NDVI range.

For each pixel, the requested exploratory summaries are:

```text
W = |CE_precipitation| + |CE_shallow_soil_moisture| + |CE_deep_soil_moisture|
A = |CE_temperature| + |CE_potential_evaporation|
ratio = W/A                         (undefined when A = 0)
water_share = (W/3) / (W/3 + A/2)    (undefined when W = A = 0)
```

The primary response-regime test uses `water_share`. It is bounded between
zero and one and gives the groups equal weight per source despite containing
three and two variables. A value of `0.5` means equal average magnitude per
source. It is an operational index, not an estimated fraction of vegetation
control. The exact requested `W/A`, both sums, and both means are also exported.
No epsilon is added to denominators. Very small positive A can still give
large finite ratios; inspect the component magnitudes and undefined counts.

Absolute values are taken **after** summing the signed trajectory. Opposite
responses over time may cancel. These indices are not sums of absolute
horizon-specific responses. Total effects can share mediating paths, and
potential evaporation combines water and energy controls, so the two groups
are not a decomposition into independent mechanisms or explained variance.

Two stratifications are always produced:

1. **Aridity quintiles**, defined over all fitted pixels with valid AI before
   applying QC, with Q1 driest. Change the number using `--quantiles`.
   Ties are never split arbitrarily; bins can be unequal or fewer than requested.
2. **Climate classes**, following the chosen data descriptor: hyperarid below
   `0.03`, arid `[0.03, 0.20)`, semiarid `[0.20, 0.50)`, dry subhumid
   `[0.50, 0.65)`, and humid at or above `0.65`. The hyperarid cutoff is `0.03`
   here; classifications using `0.05` are a different convention.

Bin medians, interquartile ranges, 5th–95th spatial percentiles, positive/zero
fractions, sample sizes and uncertainty are exported. Medians of pixel ratios
are calculated directly; they are not ratios of group medians.

## Spatial inference

The default spatial blocks are 100 km squares in EPSG:3035 (European equal-area
projection), with a fixed origin at coordinate zero. Sensitivity tests also use
50 and 150 km blocks. Set `--block-km` and `--sensitivity-block-km` to change
these; an empty sensitivity argument disables additional sizes.

For trend inference each block contributes its median AI and median response
over the same valid pixels. Spearman correlation tests a **monotonic association
between block medians**, giving each block equal weight. It does not test every
possible nonlinear or class-specific difference. The individual-pixel Spearman
correlation is supplied only as a descriptive statistic.

Two-sided block-label permutations provide p-values, using a plus-one
correction and 4,999 permutations. Paired bootstrap sampling of blocks provides
percentile 95% intervals for the rank correlation. There are 999 bootstrap
replicates by default. Results with fewer than 10 contributing blocks or
constant block values have explicit status labels and no inferential result.
Bootstrap correlation intervals require at least 95% valid replicates.

The primary test is `water_share` at the specified primary block size.
The five signed effects receive Benjamini–Hochberg adjustment as a separate
family within each block size. The W/A test is exploratory. Report block-size
sensitivity; do not select the size yielding the smallest p-value. The module
does not claim multiplicity control across repeated horizons, QC populations,
alternative index choices, or block-size searches.

Bin intervals resample whole blocks **within each bin**, keeping their pixels
together, and recompute the pixel median. Thus bin estimates remain pixel
weighted, while trend inference uses equal block weights. Intervals are
pointwise and conditional on saved effect estimates and fixed aridity bins.
They do not incorporate graph-fitting/bootstrap uncertainty. Saved marginal
effect intervals cannot substitute for paired graph replicates when estimating
uncertainty in sums or ratios.

Block permutation requires exchangeability under the null, and neither
projection nor a nominal block size proves independence. Broad spatial
gradients and dependence beyond a block can remain. The sensitivity analyses
help assess this limitation. Aridity covaries with land cover, soils,
topography and irrigation; a trend is external consistency evidence for
geographically varying responses, not identification of an aridity mechanism
or proof that the graph is causal.

## Outputs

| File | Contents |
| --- | --- |
| `pixel_aridity_effects.csv` | All fitted pixels, AI, coordinates, QC/analysis flags, signed effects, indices, and available scaling audit columns |
| `aligned_aridity.tif` | Cropped analysis-grid AI and valid-coverage bands, with original row/column offsets |
| `bin_definitions.csv` | Exact boundaries and endpoint conventions |
| `bin_summaries.csv` | Per-bin medians, spatial spread, block intervals and sample sizes |
| `qc_coverage.csv` | All-fitted, QC-passing and analyzed counts along the gradient |
| `trend_tests.csv` | Block correlations, intervals, permutation p-values, adjusted signed-effect p-values and validity status |
| `block_summaries.csv` | Values used in each block-level trend test |
| `signed_effects_*.pdf`, `signed_effects_*.png` | Signed effects and coverage by each stratification |
| `response_regime_*.pdf`, `response_regime_*.png` | Component magnitudes, balanced water share and W/A |
| `block_aridity_relationship.pdf`, `block_aridity_relationship.png` | Continuous AI relationships at the primary block size |
| `aridity_coverage_map.pdf`, `aridity_coverage_map.png` | Georeferenced AI and analysis coverage |
| `trend_tests.tex`, `signed_effects_*.tex` | LaTeX table fragments; require `booktabs` |
| `report.md`, `references.bib` | Run-specific counts, methods, interpretation notes, figure captions and recommended dataset citations |
| `manifest.json` | Parameters, dataset provenance, input hashes, code hash, versions and counts |

PDF figures retain vector text and lines; dense map/scatter marks are rasterized.
PNG figures use 300 dpi. Spatial-percentile spread is exported separately from
bootstrap confidence intervals. Figures do not claim significance when
inferential output is unavailable. Output directories must be new or empty,
so previous experiments cannot be overwritten silently.

`--seed` makes resampling repeatable. `--no-plots` supports faster table-only
runs. No production conclusion should be inferred from the synthetic test
fixtures in the repository.
