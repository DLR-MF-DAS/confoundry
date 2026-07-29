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

For VAR graphs, the diagnostic residuals are structural innovations computed
from both contemporaneous and lagged matrices:

\[
e_t = x_t - B_0 x_t - \sum_{\tau=1}^{p} B_\tau x_{t-\tau}.
\]

These are model errors used to assess the LiNGAM assumptions. They are not the
residualized environmental variables produced by the earlier residualization
procedure.
