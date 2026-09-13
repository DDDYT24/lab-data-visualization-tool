# LabViz V2.2 bounded statistics contract

This is the user-facing contract for the V2.2 local-first analysis slice. It describes what the
software calculates, what it does not calculate, and which failures must remain explicit. It is a
scientific-method contract, not a claim that the data or study design is valid.

## Supported methods

The chart-analysis API returns the selected method in `fit.fitMethod`, `fit.confidenceMethod`, and
`fit.intervalKind`. The chart preview and Matplotlib PNG/SVG/PDF export use the same values.

| Selection | Supported model | Meaning | Method details |
| --- | --- | --- | --- |
| Pointwise mean interval | Linear or polynomial | Interval for the fitted mean at each displayed X | Student-t or the existing residual-bootstrap pointwise method |
| Prediction interval | Linear or polynomial | Range for one future response at an X value | Student-t; includes estimated residual variation, so it is normally wider than the mean interval |
| Simultaneous mean band | Linear or polynomial | One band covering the displayed family of fitted mean values | Working–Hotelling critical value using the model design rank and residual degrees of freedom |
| Huber robust fit | Linear only | A fitted line with reduced influence from large residuals | Iteratively reweighted least squares, Huber tuning constant 1.345, maximum 50 iterations; confidence bands are intentionally unavailable |

Prediction and simultaneous selections currently use `confidenceMethod: student-t` in the request.
The response reports `working-hotelling` for a simultaneous band because the critical value is the
Working–Hotelling method. A robust fit reports `confidenceMethod: none` and `intervalKind: none`.

## Required evidence and assumptions

- Only finite, complete X/Y pairs enter a fit. The response reports `sampleSize` and
  `excludedCount`; it never silently turns excluded rows into observations.
- The selected model needs enough distinct X values and positive residual degrees of freedom.
  Polynomial order is bounded by the existing chart contract. Singular or insufficient designs
  return an unavailable fit rather than a fabricated band.
- Student-t intervals use the fitted residual variance and the model design matrix. Prediction
  intervals add the residual variance for one future response.
- Working–Hotelling is a simultaneous mean band for the displayed family of fitted values; it is
  not a prediction interval and does not correct arbitrary multiple comparisons.
- Huber IRLS downweights residuals according to the Huber rule. It does not repair confounding,
  selection bias, non-independent observations, a poor experimental design, or incorrect units.
  Non-convergence is a reported failure, not a successful robust result.
- A fit, R² value, interval, or residual diagnostic is descriptive evidence. It does not establish
  causality, normality, independence, or validity outside the observed X range.

## Stable failure codes

The API raises a machine-readable `ProcessingError` for invalid selections:

| Code | Meaning |
| --- | --- |
| `invalid-interval` | An advanced interval was requested without a supported fit and confidence band |
| `unsupported-interval-method` | Prediction/simultaneous intervals were paired with bootstrap |
| `unsupported-interval-model` | Prediction/simultaneous intervals were paired with an unsupported model |
| `unsupported-robust-model` | Huber robust fitting was paired with a non-linear model |
| `robust-interval-unsupported` | A confidence band was requested for Huber robust fitting |
| `robust-not-converged` | Huber IRLS could not converge or solve the selected points |
| `fit-not-suitable` | The selected design is singular, insufficient, or numerically unsuitable |

These codes are deliberately narrower than a general statistical validation service. User-defined
formulas, broad multiple-comparison correction, causal inference, and automatic method selection
are outside V2.2.

## Independent evidence

The synthetic answer key is tracked at
[`../api/samples/v22/statistics_answer_keys.json`](../api/samples/v22/statistics_answer_keys.json).
`tests/test_v22_statistics.py` compares deterministic coefficients, intervals, exclusions,
residual diagnostics, stable failures, and PNG/SVG/PDF signatures. The cases are synthetic and do
not represent laboratory measurements.
