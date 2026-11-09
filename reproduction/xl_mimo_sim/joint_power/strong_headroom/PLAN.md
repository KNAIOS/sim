# Strong-reference single-intervention headroom — frozen before collection

2026-09-22. Communications; baseline/ablation/reproducibility; unfair-comparison
and overclaim prevention; planning. This is a training diagnostic, not R1 evidence.

## Question and fixed design

The prior learned controller does not beat matched-domain PRS. Before fitting new
values, measure whether changing one action can improve that stronger reference,
and whether the improvement requires considering energy beyond one action block.

Use the existing validation-size physical configuration unchanged. Generate 12
new training-diagnostic traces: stream 5109220000, four per each of the three
unchanged speed/load scenarios. No matching stream was found in existing sources,
plans, or manifests. This stream is NOT reserved independent testing.

For each trace, run both Expanded PRS and Expanded budgetPRS, allowing 1–16
subarrays. Sample roots at slots 20, 60, and 100 on each reference trajectory:
72 correlated root states, 17 candidate entries per root, with duplicate masks
reused. Candidates are the same ranked prefixes plus the previous command.

For every distinct candidate, execute it for four slots, then resume the same
reference until slot 140. The budget reference must update its failure ledger
after the intervention and throughout its suffix; do not hold the original
reference's future budget, observations, or commands fixed. All policies retain
the identical analytical power allocator and hardware/observation model.

Save actual energy components and failed user-slot counts at horizons 4, 16,
and the complete suffix. At each root, retain candidates whose full-suffix
failure count is no greater than the reference's. Compare minimum actual energy
over horizons 4, 16, and full within that SAME hindsight risk-feasible set.
All three selectors use unavailable future information. They are diagnostic
oracles, not trained policies, causal baselines, or deployment results.

## Interpretation fixed in advance

Report all roots and scenarios. Average single-intervention differences over
roots, then trajectories; never sum savings from overlapping suffixes. A percent
uses the corresponding full reference episode energy as denominator: one
uniformly chosen sampled intervention time, not repeated-policy performance.
Bootstrap the four complete traces within each scenario, not individual roots.
Small-sample intervals are descriptive. This is not a global optimality bound.

For deciding whether another bounded value-learning probe is justified, use the
nominal budgetPRS reference: at least 2% mean full-episode energy headroom and
strict savings at at least 25% of sampled roots. To specifically motivate a
long-horizon energy head, require at least 1% full-episode energy difference
between the full-tail and four-slot energy selectors with identical hindsight
risk information. These are development screening thresholds, not significance,
novelty, or revision acceptance criteria. Failure stops that specific expansion.

## Claim–evidence matrix

| Claim | This diagnostic can provide | Still missing | Risk |
|---|---|---|---|
| Strong PRS leaves one-action savings | Exact paired simulator suffixes | Causal generalization | Major |
| Long-term energy choice matters | Same risk filter, different energy horizons | Learned realization | Major |
| RL beats PRS | Nothing directly | Frozen causal policy on new traces | Critical |
| Standard-channel applicability | Nothing | New-method standardized-channel run | Major |

## Checks and boundaries

Test reference suffix identity, no parent-state mutation, duplicate masks,
causal feature invariance to future trace changes, budget accounting, and selector
feasibility. Independently recompute antenna-domain rates and energy for the
reference and selected branches. Preserve all old source/data/weight hashes.
Use two local workers and existing venv only; no installs, SSH, environment
changes, formal-manuscript overwrite, or outcome-dependent stream replacement.
