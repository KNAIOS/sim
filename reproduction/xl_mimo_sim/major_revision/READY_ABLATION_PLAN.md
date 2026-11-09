# Readiness-conditional input ablation

Frozen before training, 2026-09-20. This follows the constructed alias in
READINESS_STATE_ALIAS.md, not a post-hoc attempt to hide the negative PRS comparison.

The existing 178-dimensional, price-4, 300-episode, two-seed controller is the
reference. Add 39 immediate-after-command rate ratios computed from causal CSI
and actual RF availability. All existing 178 features remain in their original
order. Input becomes 217; hidden widths and other DQN hyperparameters stay fixed.
The input layer has 2496 additional weights; an improvement cannot be attributed
solely to information content without an equal-capacity control. That control
is warranted only if the initial performance screen is promising.

Keep the identical training/validation/development streams, two seeds 11/22,
300 episodes per seed, price 4, exploration schedule, checkpoint cadence and
feasibility-first selection used in CONSTRAINT_TRAINING_PLAN.md. The already
viewed 16-trajectory-per-scenario development set remains development data.
Both previously tuned search baselines and every original training seed remain
in the comparison. There is no new price or search-margin sweep.

The physical episode adapter is tested against the frozen engine over delays
0, 2, 4, 6, including intermediate and terminal replay transitions. With an
identical forced policy, adding the new features must not alter physical results.
No future channel, extra pilots, uncharged switching or modified power constants
are introduced. Original hashed engines and result folders remain intact.

Reporting: energy, outage, paired seed/trajectory intervals, added per-decision
RZF work. Reliability-violating lower energy is not a success. Higher energy
at lower outage is a tradeoff. Only a promising feasible energy improvement
justifies equal-capacity control, independent confirmation, and replacement of
the manuscript method and all method-dependent experiments. Otherwise retain
the negative result and do not describe the feature as experimentally validated
for energy saving.

Two single-thread workers at most; existing local project venv; no installation,
global environment edits or remote-server modification.
