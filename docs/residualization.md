# Seasonal and trend residualization

`confoundry.residualize_timeseries` removes deterministic calendar seasonality
and an optional linear trend independently for every pixel and environmental
variable. The default model is

\[
x_t = \alpha + \sum_{m=2}^{12}\gamma_m I(\text{month}_t=m)
      + \beta (t-\bar t) + r_t.
\]

Calendar-month fixed effects allow an arbitrary repeating annual shape. This
is more suitable for asymmetric environmental cycles than the legacy model
containing only one sine/cosine pair. Centering time and expressing the trend
per year improves numerical conditioning. The residual `r_t`, rather than the
fitted deterministic component, is supplied to graph discovery.

The default fit requires at least 60 observations overall and at least three
observations in every calendar month. The residualization-model table records
the design rank, condition number, month counts, fit R², residual standard
deviation, time center, and complete coefficient JSON for every pixel-variable
fit.

## Run the recommended model

Write new outputs rather than overwriting the previous harmonic residuals:

```bash
python -m confoundry.residualize_timeseries \
  -c path/to/experiment.yaml \
  --seasonal-model monthly-fixed-effects \
  --trend \
  --min-fit-samples 60 \
  --min-month-samples 3 \
  --output-table experiment_residualized_monthly \
  --output-config path/to/experiment_residualized_monthly.yaml \
  --graph-db path/to/experiment_residualized_monthly_graphs.duckdb
```

The legacy model remains available only for reproducibility:

```bash
--seasonal-model annual-harmonic
```

## Visual validation

```bash
python -m confoundry.visualize_residualization_sample \
  -c path/to/experiment_residualized_monthly.yaml \
  --selection representative \
  --output-dir path/to/residualization_figure
```

Use repeatable `--label RAW=DISPLAY` options for variables without built-in
publication labels.

The figure contains four columns for every variable:

1. original observations and the fitted deterministic baseline;
2. residual anomalies supplied to causal discovery;
3. standardized monthly climatologies before and after fitting;
4. autocorrelations at lags 1–12 before and after fitting.

Monthly residual means are in-sample fit targets and therefore are not an
independent validation of seasonal removal. The autocorrelation profile is the
important separate check: calendar residualization can remove average monthly
seasonality while leaving temporal persistence, changing seasonal amplitude,
nonstationarity, or omitted-driver dependence. The command exports the plotted
profiles to a dedicated `_autocorrelation.csv` file.

Residualization adequacy does not establish the causal assumptions of
VAR-LiNGAM. After residualization, run the reduced-form innovation-whiteness,
structural-error, and stability diagnostics separately.
