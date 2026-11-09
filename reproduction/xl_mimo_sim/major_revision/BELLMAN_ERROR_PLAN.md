# Frozen-model immediate-risk and deployment-continuation diagnosis

2026-09-21. The preceding Bellman candidate remains FAILED and stopped. This is
an author-only diagnosis, not more training, price tuning or a new control curve.
Detected axes: communications; ablation/statistical/reproducibility;
overclaimed-results; audit. R1 remains the target proof gap.

## Questions and evidence limits

| Question | Required comparison | This bounded audit | Cannot establish |
|---|---|---|---|
| Is immediate risk misestimated on visited states? | Same frozen predictor versus actual four-slot outcomes | All16 actions at learned/search roots; train-root reference | Calibrated probabilities, unique distribution-shift cause |
| Are action rankings distorted by immediate or continuation terms? | Relative learned-first minus search-first cost errors | Exact four-term decomposition, same root/future/learned continuation | Population-optimal Q or full-policy superiority |
| Would perfect immediate knowledge remove sampled mistakes? | Replace ONLY root immediate-risk prediction | Hindsight three-slot-risk replacement, unchanged predicted tail | Deployable oracle, achievable energy saving |

Offline extrapolation and distribution-shift errors are established risks, not a
newly invented diagnosis: Fujimoto, Meger and Precup, ICML2019,
https://proceedings.mlr.press/v97/fujimoto19a.html; Kumar et al., CQL2020,
https://arxiv.org/abs/2006.04779. These sources motivate checking the error; they
do not prove that it is the cause here or transfer a safety guarantee. This audit
does not implement BCQ/CQL or claim their novelty.

## Fixed sample and policies

- Existing111 validation stream, already viewed: first four trajectories in each
  of all three scenarios. Twelve shared external trajectories, not fresh tests.
- Frozen seeds11/22, depth16, same previously selected per-scenario prices. No
  new prices/depths/checkpoints. Strongest feasible pure search settings are read
  from the existing complete48-configuration validation table, not reselected
  using diagnostic results. Search receives unchanged information/actions.
- For each seed, separately roll in with its frozen learned policy and that
  scenario's frozen search. Save roots t=0,32,64; three roots per trajectory.
  Thus72 roots/seed,144 total. Search state histories are duplicated across seeds
  only for shadow predictor evaluation; duplicates are not independent trials.
- Reproduce every corresponding stored episode metric before interpreting roots.

## Root labels and decomposition

At each frozen root, cache causal features, all16 immediate/tail predictions and
the learned/search actions BEFORE reading any future labels. Physically execute
every candidate for four slots in isolated clones:2304 short branches total.
Record actual normalized energy E4 and user-slot outage fraction O4. Exact E4
and current-slot outage identities must hold; root arrays must not mutate.

Define three diagnostic first actions:
1. the already frozen learned greedy action;
2. the causal search action at this same root;
3. a NONDEPLOYABLE immediate-risk oracle: minimize the old predicted64-slot cost
   after replacing ONLY its predicted O4 by sampled actual O4. The predicted
   energy and all predicted continuation terms stay unchanged. No long-return
   labels enter this choice. It is hindsight, not a learned controller.

For each distinct first action, replay64 physical slots: hold first command4slots,
then repeatedly execute the SAME frozen depth16 learned policy at its selected
price for60slots. Deduplicate equal first actions. Maximum432 long branches,
27648 long-branch slots; retain all three labels even when actions coincide.
Use gamma=.98 per four-slot block, as in the candidate. All labels include
operating/switching/pilot energy and actual per-user service failures.

This continuation is the deployed receding policy. Its sampled return is not an
exact optimal finite-horizon value or conditional expected Q. The fitted network
used preceding-depth greedy targets, so disagreement also includes this finite
horizon/receding-policy mismatch, not just statistical prediction error.

For learned-first minus search-first, decompose the total scalar-cost prediction
error EXACTLY into:
- known immediate energy error;
- lambda-weighted immediate outage error;
- continuation energy error;
- lambda-weighted continuation outage error.
Report signs, absolute errors, sampled ranking reversals, and trajectory-cluster
summaries. The four SIGNED terms add, their absolute errors do not. Prices differ
by seed/scenario; do not compare raw weighted error magnitudes across these cases
as a common physical metric. Discounted outage cost is not a percentage point.

Training reference: read the existing840-root training files, restrict to the SAME
root times0/32/64 (72 roots/seed), and compute immediate-risk errors from unchanged
labels. Training roll-ins were mu/old mix_pi; diagnostic roll-ins are current
learned/search. This is not a one-factor causal distribution-shift experiment.

## Reporting and stop rules

Report all seeds, scenarios and roll-ins, including no-op comparisons. Immediate
risk errors use prediction minus sampled actual block fraction; translate to
percentage points only after multiplying by100. Use MSE/MAE of fractions, not
claiming independent Bernoulli calibration. Relative returns use the same future,
but roots on one trajectory overlap and are correlated. No p-value from roots.

Verify exact episode replay, learned-on-own-state64-slot suffix replay, identical
short/long first-block labels, prediction/source/checkpoint immutability,
error-sum identity, and duplicated search features/short labels across seeds.

The audit cannot reopen the failed gate, generate performance-promising figures,
alter current R2/R3 claims, or authorize unlimited training. A future method needs
a separate evidence-motivated protocol; negative or ambiguous diagnosis is kept.
Only existing project venv, at most two single-thread workers, no host/server
installation or global environment changes. All prior code/data/plans remain intact.
