# Bounded reliability-selection and training-budget audit

Frozen before execution, 2026-09-20. Original goal remains a defensible major revision;
this is not permission to replace the required energy comparison with latency alone.

## Evidence behind this action

The previous rate-context candidate trained for 100 episodes and selected a checkpoint
by mean normalized energy plus four times outage, averaged over three scenarios.
That scalar objective contains neither a feasibility gate nor a per-scenario constraint.
The independent 40-trajectory evaluation subsequently had 8.50% outage. This is a
selection/optimization mismatch with the stated 5% objective, not proof that fixing
selection will make the learned policy outperform PRS.

Finite-penalty objectives are not automatically equivalent to constrained policy
optimization. The distinction is described, for example, by Achiam et al.,
Constrained Policy Optimization, ICML 2017:
https://proceedings.mlr.press/v70/achiam17a.html
This audit does not implement CPO or claim its theoretical guarantees.

## Fixed scope

- Keep the 178-dimensional causal state, 64-unit Double-DQN architecture, 13 count
  actions, ranking, physical channel, demand process and all power constants unchanged.
- Two fixed prices: 4 (previous candidate price) and 8 (one prespecified stronger
  reliability penalty); two training seeds 11 and 22; 300 episodes each, 1200 total.
- Same training scenario/trajectory streams and exploration schedule across both arms.
  Epsilon is `0.04 + 0.76 exp(-4 episode / 99)`; longer training therefore mostly
  adds low-exploration experience rather than stretching the original decay.
- Record all checkpoints at 50, 100, 150, 200, 250 and 300 episodes, including failures.
- Twelve validation trajectories per scenario, not the previous four. Select by
  `(max_scenario(mean_outage)-0.05)_+`, then mean energy; mark infeasible selections.
- Save and restore the policy exploration RNG around greedy validation: the number
  of validation calls must not change later training exploration. This is a new
  protocol, not a claim of exact reproduction of the old 100-episode run.
- Validation is selection data, not evidence of certified reliability. Confidence
  intervals and independent testing remain necessary.

Streams: training 971000000 + 10000*seed + episode; validation 972000000 +
10000*scenario + trajectory; development 973000000 + 10000*scenario + trajectory.
These are distinct from previously viewed confirmation and standardized data.

## Fair comparison and stopping gates

If selected models are validation-infeasible, report that result without more price
search. Otherwise use a development screen with 16 shared trajectories per scenario.
Give PRS and switch-aware PRS their own margin selection on the same validation set;
use the same 13 margins as the prior confirmation (1.00 to 1.10 in .01 steps,
1.15, 1.20), retain all outcomes. True demand is never changed by tuning a margin.
Report energy together with outage, including full-array feasibility.

A lower-energy policy that fails 5% is not a win. Only a feasible, promising screen
justifies a separate frozen-method confirmation with new trajectories and uncertainty
intervals. This experiment does not establish novelty or justify swapping the final
paper's model, timing or standard-transfer results prematurely.

At most two single-thread workers run simultaneously, inside the existing project
venv. No package installation, server connection, system environment or driver change.
All outputs go to a new results directory, preserving prior results and manuscript.
