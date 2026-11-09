# Paid-probe causal predictability diagnostic — frozen before fitting

2026-09-21. This is a bounded supervised diagnostic, not RL training, an
entire closed-loop policy, a replacement manuscript result, or a reopening of
the failed INFORMATION_PROBE gate. Original data, source, gates, and PDFs stay
unchanged. No installation or host/server configuration changes are allowed.

## Question and evidence contract

| Claim under examination | Evidence | Scope / exclusion |
|---|---|---|
| Observable state contains usable signal for a profitable single probe | Frozen predictors choose before reading new outcomes; new paired suffix replays | Not proof of optimal control or of RL superiority |
| Probe energy gains do not simply purchase more rate failures | Report energy AND whole-episode outage deltas, including skipped roots | Point predictions do not guarantee reliability |
| Information, rather than only extra active hardware, matters | Replay selected learned probes while suppressing their first four slots of new CSI | Counterfactual diagnostic; not deployable ideal-CSI hardware |
| Results generalize beyond the already inspected branch data | Stream 114 development versus unused stream 115 validation | New stream is validation for this diagnostic, not a future RL test set |

Primary inspiration: Krale, Simão and Jansen, *Act-Then-Measure*, ICAPS 2023,
https://doi.org/10.1609/icaps.v33i1.27197. The official abstract separates
control and measurement actions and introduces measuring value. It does not
establish savings in this XL-MIMO simulator or justify copying its guarantees.
This experiment only asks whether expected incremental outcomes can be
approximated from existing causal controller observations.

## Data and fixed budget

- Fit only the 423 normal probe outcomes at 36 roots from 12 already viewed
  stream-114 trajectories. The blinded outcomes are NOT prediction targets.
- Features use saved causal snapshots, known current time/configuration, and
  the same causal Gram/RZF hold forecast as search. No future channel, future
  demand, physical outcome, scenario identifier, trajectory identifier, or
  module identifier is a feature. Spatial distances are physical features.
- Leave-one-external-trajectory-out predictions (12 groups) characterize
  development error. All scalers/target means are fit within each fold.
  This viewed source cannot become an independent test by cross-validation.
- Fit two final models once, then freeze their files before any validation:
  linear ridge and Gaussian-kernel ridge. No hyperparameter sweep, no new
  model choice based on validation, no checkpoint selection.
- New validation: stream 1150000000 + scenario*10000 + index, indices 0..7,
  three unchanged scenarios, 24 external trajectories, roots 32/64/96.
  A root is on the original cost-search trajectory, never selected by outcome.
- At all 72 roots evaluate every eligible normal probe plus no probe. Replay
  at most two additional blinded probes selected by the learned predictors.
  Each unique branch is simulated once and independently checked once.
  At most two single-thread workers. No automatic expansion of this budget.

## Model, labels, and decisions

The 23 feature names and exact formulae are frozen in
`probe_predictability.py:FEATURE_NAMES` / `features`. They encode remaining
time, CSI age, array readiness, current requirements, relative RA score,
gain/support/uncertainty/velocity, physical distance to the reference mask,
and causal hold-energy / outage / shortfall forecasts. Predictors receive
only a snapshot, not a trajectory or rollout object.

For each actual probe, y=(E_probe-E_base-0.33 J,
100*(O_probe-O_base)). The known 0.33 J first-four-slot overhead is added back
at inference. O is the same whole-140-slot user-slot rate-failure fraction.
Subtracting known cost is algebraic; it supplies no unknown future label.

Both models center targets, standardize inputs with training mean/std
(std floor 1e-6), and use ridge lambda=1. Linear ridge fits centered outputs
against standardized inputs. Kernel ridge uses
K(x,z)=exp(-||x-z||^2/(2*d)), d=23, on standardized inputs. No calibrated
probability, risk bound, or confidence score is asserted.

For each model: discard probes with predicted energy delta >=0 or predicted
outage delta >0; among the rest choose smallest predicted energy delta, with
module index only a deterministic tie-breaker. Otherwise select no probe.
Keep skipped roots in every average. Predictors and choices are saved BEFORE
future replays. Report both models; do not select a winner and hide the other.

Comparators: no probe, highest RA-score probe, oldest-CSI probe (same exact
rules as the prior diagnostic). All have identical candidate permissions,
past observations, physics, requirements, continuation controller, and cost
accounting. The two simple rules are not substitutes for the strong search:
all changes are measured against the same strong no-probe cost search.

## Reporting and decision boundaries

Report candidate prediction RMSE/MAE and skill versus the development-mean
predictor, grouped-out development results, all validation rule outcomes,
absolute energy/outage, selection frequency, per-trajectory deltas, and
normal-minus-blinded information differences. Candidate-weighted prediction
metrics are descriptive, not independent samples. No branch-level p-value.

For policy-delta uncertainty, average the three roots within each trajectory
then bootstrap the eight trajectories per scenario (10000 resamples, seed
1150900001). Show 95% percentile intervals, labelled descriptive and without
a claim of simultaneous coverage across multiple models/scenarios.
These are single-intervention averages, NOT a three-probe policy or savings
that may be added across roots.

A nominal model with mean energy delta <0 and mean outage delta <=0 counts
only as a positive exploratory mean signal. Also report whether both CI
upper bounds satisfy those conditions and whether absolute outage <=5%.
Regardless of result, `permit_rl_training=false`,
`permit_full_policy_screen=false`, `scientific_completion=false` for this
diagnostic. A positive result would require a separate, explicitly designed
closed-loop study with equally capable planning comparisons. A negative
result stops these two predictors without added training or fresh-seed hunting.

## Verification

Unit checks: feature/selection independence from hidden fields; reconstruction
from saved causal snapshots; no input mutation; training-only normalization;
closed-form residual equations; model serialization; skip/tie behavior; known
probe overhead. Rebuild all validation roots and decisions independently and
replay every recorded branch using the original manual-assimilation verifier.
Verify source, input, model, and output hashes. No raw result is overwritten.
