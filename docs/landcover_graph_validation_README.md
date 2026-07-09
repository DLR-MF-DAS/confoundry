# Graph-to-land-cover validation

## Installation

Place `landcover_graph_validation.py` in:

```text
src/confoundry/landcover_graph_validation.py
```

Install the additional classifier dependencies:

```bash
pip install -r landcover_validation_requirements.txt
```

## Recommended first run

Set `--graph-window-size` to the same value used for graph discovery.

```bash
python -m confoundry.landcover_graph_validation \
  --config-path data/iberian_drought_experiment/experiment.yaml \
  --graph-window-size 0 \
  --feature-set combined \
  --class-set vegetation \
  --min-purity 0.80 \
  --block-size-km 100 \
  --folds 5 \
  --classifier random_forest \
  --trees 500 \
  --workers -1
```

For graphs fitted from 3x3 pixel neighborhoods:

```bash
python -m confoundry.landcover_graph_validation \
  --config-path data/iberian_drought_experiment/experiment.yaml \
  --graph-window-size 1
```

For 5x5 neighborhoods:

```bash
python -m confoundry.landcover_graph_validation \
  --config-path data/iberian_drought_experiment/experiment.yaml \
  --graph-window-size 2
```

## Class sets

```text
vegetation:
    Tree cover
    Shrubland
    Grassland
    Cropland

terrestrial:
    vegetation classes
    Built-up
    Bare or sparse vegetation

all:
    all 11 ESA WorldCover classes
```

The default `vegetation` set avoids obtaining deceptively high scores merely
by separating water, built-up areas, and vegetated land.

## Graph feature sets

```text
consensus:
    flattened consensus adjacency matrix

raw:
    flattened raw DirectLiNGAM adjacency matrix

probability:
    flattened bootstrap edge-probability matrix

total_effect:
    flattened total-effect matrix computed from the consensus graph

combined:
    consensus weights + edge probabilities + total effects
```

`month_sin` and `month_cos` are excluded by default. Repeat
`--exclude-variable` to exclude additional nodes.

## Output

The default output directory is:

```text
<experiment directory>/landcover_graph_validation/
```

Main files:

```text
<name>_landcover_validation.duckdb
validation_samples.csv
class_summary.csv
cv_metrics.csv
cv_predictions.csv
summary.json

confusion_matrix_majority.png
confusion_matrix_graph.png
confusion_matrix_raw_summary.png
confusion_matrix_graph_plus_raw.png
cv_metrics.png
landcover_class_map.png

graph_classifier.joblib
graph_plus_raw_classifier.joblib
feature_importance_graph.csv
feature_importance_graph.png
feature_importance_graph_plus_raw.csv
feature_importance_graph_plus_raw.png
```

DuckDB tables:

```text
landcover_labels
validation_samples
class_summary
cv_metrics
cv_predictions
feature_importance
```

## Reuse downloaded labels

WorldCover files are reused automatically. To avoid resampling the land-cover
raster after a completed run:

```bash
python -m confoundry.landcover_graph_validation \
  --config-path data/iberian_drought_experiment/experiment.yaml \
  --graph-window-size 1 \
  --reuse-labels
```

Only use `--reuse-labels` when the reference raster, graph window size, and
land-cover sampling configuration have not changed.

## Important interpretation

The key comparison is:

```text
graph vs majority
graph vs raw_summary
graph_plus_raw vs raw_summary
```

A strong result is not merely high graph accuracy. The graph model should
outperform the majority baseline under spatially blocked cross-validation,
and ideally `graph_plus_raw` should outperform `raw_summary`. That indicates
that causal graph structure adds information beyond ordinary environmental
means and variances.

Successful classification demonstrates a systematic association between graph
structure and independently mapped land cover. It does not by itself prove
that every inferred edge is physically causal.
