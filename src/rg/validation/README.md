# Validation utilities

The validation scripts emit lightweight artifacts that document the synthetic phase structure used throughout the paper drafts.

## Phase boundary fit outputs

Running `python -m rg.validation.phase_boundary_fit ...` generates two files under `rg/results/sage_corrected/`:

* `phase_boundary_fit.csv` — tabulates each sampled `(lambda, eta_critical)` pair. Downstream notebooks can plot this scatter directly or compare it against the linear model.
* `phase_boundary_fit.json` — summarises the linear fit `eta_c = a * lambda + b`. The JSON stores the slope (`a`), intercept (`b`), coefficient of determination (`R2`), the inferred `I_hat = 1/a`, and the reference `gamma` supplied on the CLI alongside the simulation flags used to generate the data.

Interpretation tips:

* A larger slope tightens the boundary; `I_hat = 1/a` serves as the characteristic information scale used in the main text.
* When the intercept deviates substantially from `gamma`, the script warns that the placeholder dynamics might need retuning for that configuration.
* The flags section records algebra/noise/MI estimator choices so repeated sweeps can be compared without ambiguity.
