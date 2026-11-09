# Causal rate-risk correction, 2026-09-22

This protocol is frozen before generating its data. It preserves the original
mobile-UE, fixed-BS, single-cell geometry, instantaneous rate requirements,
equal-power RZF, physical power model, hardware clock, active-only CSI, and
RA spatial ranking. It does not add queues, transmit-power actions, sleep
levels, extra probing, or future information at deployment.

The interrupted stream-115 probe diagnostic remains INCOMPLETE and untouched;
its partial files are not used here. No old failed gate or published number
is replaced. Only the existing project venv is used, maximum two single-thread
workers; no installs or host/server changes.

## Claims and controls

| Question | Intervention / control | Required evidence |
|---|---|---|
| Is known current CSI preserved? | Override only h=0 Gram forecast on already observed modules with their measured Gram | Exact RZF parity and unchanged hardware energy |
| Are future rate failures better predicted? | Fixed logistic model using causal forecast/context, then held-out temperature scaling | Brier, log loss, reliability bins, age/horizon/scenario strata |
| Does prediction improvement save energy at comparable service? | Original cost search / current-CSI correction alone / corrected calibrated-risk search | Same count domain, independent parameter-selection trajectories, new paired evaluation |
| Does RL add value beyond the corrected estimator? | Subsequent matched original/corrected RL versus corresponding searches | Separate frozen training protocol only after the estimator screen passes |

## Meaning of the correction

Old forecasts rescale even a freshly measured active module's complex CSI to
the tracker's filtered gain. Under the existing ideal active-CSI PHY, that
needlessly changes the known present channel. The new forecast uses its
measured Gram at h=0. Forecasts h=1..7 are unchanged. This is a current-state
consistency correction, not a new physical channel or a claim of perfect
future CSI. In the four-slot wake model no previously unobserved module can
become active in the current slot, so every current available contribution
is known exactly. The original tracker and score are retained as controls.

The probabilistic model predicts each user-slot's rate-failure probability
conditional on the current causal summary and holding one candidate mask for
eight slots. Their average is an expected user-slot failure fraction, NOT the
probability that no user ever fails, and not a hard safety certificate.
Future demand changes are part of the outcome, not exposed as inputs.

## Fixed data partitions and budget

Trajectory seed = stream + scenario*10000 + index, unchanged three scenarios.
- Fit: 1160000000, 8 trajectories/scenario (24 total).
- Probability calibration: 1170000000, 4/scenario (12 total).
- Predictive and policy audit: 1180000000, 8/scenario (24 total).
- Search-parameter selection: 1190000000, 4/scenario (12 total).
- Streams 120/121 are RESERVED, not used until a separately frozen RL plan.

All predictive roots are t=0,8,...,128 (17/trajectory). All 16 ranking prefixes
are labelled over all 8 future slots and 3 users using the unchanged actual
channel/requirements. Roots follow the original cost search (margins
1/1.08/1.04, minimum counts 1/1/4), except fitting trajectories with index 3
or 7 use a uniformly random prefix at every other decision. This controlled
mixture broadens training hardware states; calibration/audit use only search.
Root features are saved before obtaining future labels. No outcome-dependent
root selection, sample exclusion, oversampling, seed hunting, or budget expansion.

## Fixed probability model

Features are the 14 formulas in `risk_correction.py:risk_features`: predicted
log rate margin, its square, horizon, margin-horizon interaction, current CSI
age weighted by estimated channel contribution, age-horizon interaction,
fraction of available modules currently unobserved, active fraction, user's
current demand and recent change, support velocity, tracker uncertainty,
filtered-versus-measured gain difference, and margin-age interaction.
No future CSI, UE true position/velocity, future demand, identifiers, or labels.

Fit one logistic regression, training-only standardization, unweighted binary
cross entropy plus 1e-4 mean squared slope penalty, deterministic zero start,
full-batch float64 L-BFGS up to 150 iterations. Only uncertain future rows are
fit: h>0 and a nonempty active set. Scale logits by a positive temperature fit
only on the disjoint probability-calibration split (at most 80 L-BFGS steps).
The sign of logits is preserved. Current-slot risk is set to its exact known
indicator; zero available antennas give failure probability one. The old
probability/uncertainty heuristic is NOT relabelled as calibrated probability.

Temperature scaling reference: Guo et al., ICML 2017,
https://proceedings.mlr.press/v70/guo17a.html. Its classification results are
not a proof of calibration under our dependent trajectories or policy shift.
Risk-aware incomplete-CSI planning inspiration: He et al., TWC 2024,
DOI 10.1109/TWC.2024.3377733. This is not a reproduction of their generative
channel posterior or RA-MCTS algorithm.

## Search comparisons and selection

Identical 16 prefixes, with minimum count either 1 or 4 for every method.
Old and current-corrected point searches use margins
(0.90,1,1.04,1.08,1.16,1.24), target predicted outage 0.05.
Calibrated search uses probability thresholds (.01,.025,.05,.075,.10,.15).
Thus each method has 12 candidates per scenario and identical selection data.
Choose minimum mean energy among candidates with mean actual outage<=.05;
if none is feasible, choose minimum mean outage then energy. Ties prefer
smaller minimum count then parameter. All candidates and failed results stay.

Feasible point-search actions minimize predicted total eight-slot energy,
otherwise minimize predicted shortfall. Feasible calibrated actions minimize
the SAME energy; otherwise choose lowest predicted failure fraction. This
fallback change is part of the risk-aware controller, explicitly disclosed.

Audit all three selected methods on the 24 new stream-118 trajectories.
Report paired differences against BOTH original and current-corrected search,
absolute energy/outage, and trajectory-level 95% bootstrap intervals (10000
paired resamples per scenario, seed 1180900001). Intervals are descriptive,
not simultaneous or a population safety guarantee. Lower energy with higher
outage is reported as a tradeoff, not strict superiority.

## Gate and verification

Also report a training-prevalence constant-risk predictor, with the same
known-current/zero-active overrides, to check improvement beyond a trivial
probabilistic reference. This bounded screen passes only if calibrated future
Brier is lower than both the old deterministic forecast and that constant
predictor in every scenario, pooled future ECE (10 fixed
equal-width bins) <=.03, and nominal closed-loop outage<=.05. Passing permits
design of a matched RL study, not an automatic claim of RL superiority. Failing
stops additional training on this candidate; no revised thresholds on audit.

Verify current-slot algebra against physical RZF; all hold energies/rates
against independent scalar hardware simulation; hidden-future poisoning of
the causal interface; no state mutation; source/data/model hashes; all final
selected episode replays and summary arithmetic. Test phase completed outputs
are separate from scientific_completion=false for the whole major revision.
