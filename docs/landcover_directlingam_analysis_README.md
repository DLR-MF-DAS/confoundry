# DirectLiNGAM effects vs land cover

Compare per-pixel DirectLiNGAM analysis outputs, such as
`scaled_total_effect`, with ESA WorldCover land-cover classes.

The command joins rows from `confoundry.per_pixel_directlingam_analysis` to
dominant land-cover labels and writes:

- class-wise effect summaries;
- one-hot class-indicator correlations for each source/effect metric;
- heatmaps and boxplots for the strongest source/class associations.

Because land cover is categorical, the reported correlation is the Pearson
correlation between a numeric effect column and a binary indicator for one
land-cover class.

## Recommended run

First compute DirectLiNGAM effects:

```bash
python -m confoundry.per_pixel_directlingam_analysis \
  --config-path data/iberian_drought_experiment/experiment.yaml
```

Then compare the effects with land cover:

```bash
python -m confoundry.landcover_directlingam_analysis \
  --config-path data/iberian_drought_experiment/experiment.yaml \
  --metric scaled_total_effect \
  --class-set vegetation \
  --min-purity 0.80
```

To compare several effect columns:

```bash
python -m confoundry.landcover_directlingam_analysis \
  -c data/iberian_drought_experiment/experiment.yaml \
  --metric scaled_total_effect \
  --metric abs_scaled_total_effect \
  --metric scaled_total_effect_boot_sd
```

## Reuse Existing Land-Cover Labels

If `confoundry.landcover_graph_validation` has already been run, this command
automatically reuses:

```text
<experiment directory>/landcover_graph_validation/<name>_landcover_validation.duckdb::landcover_labels
```

You can also pass labels explicitly:

```bash
python -m confoundry.landcover_directlingam_analysis \
  -c data/iberian_drought_experiment/experiment.yaml \
  --labels-db data/iberian_drought_experiment/landcover_graph_validation/iberia_landcover_validation.duckdb
```

If no saved labels are found, the command samples ESA WorldCover using the same
helpers as `landcover_graph_validation.py`. Set `--graph-window-size` to match
the graph discovery neighborhood.

## Output

The default output directory is:

```text
<experiment directory>/landcover_directlingam_analysis/
```

Main files:

```text
effect_landcover_samples.csv
effect_landcover_class_summary.csv
effect_landcover_correlations.csv
<name>_landcover_directlingam_analysis.duckdb
summary.json

<metric>_landcover_indicator_correlations.png
<metric>_landcover_class_means.png
<metric>__<source>__landcover_boxplot.png
```

DuckDB tables:

```text
landcover_labels
effect_landcover_samples
effect_landcover_class_summary
effect_landcover_correlations
```

## Interpretation

`effect_landcover_correlations.csv` answers: for a source variable and effect
metric, is that effect larger or smaller in pixels belonging to a given
land-cover class than elsewhere?

Positive `indicator_correlation` means the metric tends to be higher in that
class. Negative values mean it tends to be lower. Use
`class_minus_other_mean` and the boxplots to check the magnitude and direction
in the original effect units.

The default `--min-class-samples 1` keeps rare land-cover classes visible for
exploratory analysis. Check `n_class`, `class_fraction`, and
`effect_landcover_class_summary.csv` before interpreting small classes.
