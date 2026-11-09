# Long-term action-value diagnosis, frozen before collection

2026-09-22. The preceding goal turn made progress: phase3 implementation,
four training runs and independent development/physics audit. Its gate failed;
R1 remains open. Stop expanding that candidate, not the full major revision.
No currently live job is assumed. No new research scope permission is needed.

## Question and immutable controls

Does the already-authorized joint subarray/power task contain predictable net
energy improvements over PRS, after accounting for the hardware/observation
state created by a different first action? Preserve the existing simulator,
all17 actions, power allocator, CSI permissions, demands, energy coefficients
and original paper. Do not weaken PRS or fabricate experimental data.

From states actually visited by causal PRS, execute each of the17 candidate
commands for4 slots, then use the SAME causal PRS policy until episode end.
Every branch has its own hardware, tracker and CSI cache. Share only exogenous
channel/demand trajectories. Labels include the first4, first16 and entire
remaining episode. Collect energyJ and integer count of failed user-slots.
The baseline action is included and its complete continuation must reproduce
the original PRS trajectory suffix. Duplicate masks may share identical labels.

Current features are computed BEFORE any branch runs. They are the frozen
343-dimensional PDS features and a reference-action index. True future is
used to form offline outcome labels, NEVER as a deployment input.

## Fixed data and bounded diagnostic

Train stream3609220000, 8 trajectories/scenario; probe-validation3709220000,
4/scenario. Same three existing speed/load scenarios. Sample eight states at
slots0,16,...,112 per trajectory. Thus24 training +12 validation trajectories,
288 states and4896 action labels. Overlapping state suffixes and candidate
branches are NOT independent samples. Training and validation split by entire
trajectory. No previous development trace is included.

Primary diagnostic: hindsight action minimizing remaining energy subject to
no larger total failed-user-slot count than PRS continuation. Secondary: also
require no increase at both4 and16 slots. These are per-state single-intervention
diagnostics, NOT implementable controllers, global optimal bounds or additive
episode savings. Report all scenarios and horizons. Compare4-slot and16-slot
hindsight choices on the FULL remaining suffix to expose delayed costs.

No larger RL retraining just because this hindsight selector wins. Only if
nominal validation has >=25% states with >=0.2J no-extra-failure improvement,
and mean hindsight saving >=0.2J/intervention, proceed to the learning probe.
This is a research triage rule, not a significance or manuscript acceptance test.

## Prespecified learning probe, if the diagnostic gate passes

Fit Monte Carlo action-value differences relative to PRS using training
outcome labels only. Two64-SiLU layers, two outputs for normalized energy and
failed-slot differences, prediction f(x_action)-f(x_reference) so the reference
has exact zero advantage. Normalize inputs using training only; labels:
deltaEnergy/full_slot_energy and deltaFailedUsers/users. SmoothL1 loss,
AdamW0.001/weight decay0.0001, clip5, full-state batches16,80 epochs, seeds11/22.
Save every20 epochs including initial. A linear paired-feature ridge model
(alpha1) and unchanged PRS are simple comparison probes.

Use costs deltaE_normalized + lambda*deltaFailedEquivalent with lambda{4,8,16}.
Choose the minimum predicted cost, allowing the exact-zero baseline action.
Select epoch/price on probe-validation by largest per-scenario mean extra
failed-user-slot violation above0, then mean energy difference, then validation
Huber. All candidates and scores retained. Report lack of improvement if the
initial/reference policy is selected. This is one approximate policy-improvement
iteration, not Double DQN, convergence proof or distribution-free safety.

If a fitted probe is selected, check its genuinely causal closed-loop behavior
on12 fresh development trajectories stream3809220000 (4/scenario), alongside
PRS+power, raw joint full/coarse, myopic, frozen phase3 branchRL(two seeds),
and full-array power. Do not use diagnostic hindsight labels at deployment.
Keep the original5% per-seed scenario outage target and nominal>=2% energy
gain versusPRS, with neither seed regressing. All baselines share capability.
No new standardized-channel performance expansion if this gate fails.

## Literature and boundaries

This uses base-policy rollout/approximate policy improvement as described in
[Bertsekas's original lecture notes](https://web.mit.edu/dimitrib/www/RLTopics_2021_Lect2.pdf).
It is not a newly invented RL principle. The guaranteed policy-improvement
theory requires appropriate exact expectation/value assumptions; a single
future path, finite dataset, neural approximation and partial observation do
not satisfy them automatically. Restricting deviation from a baseline also
has substantial prior work, e.g. the authors' [SPIBB project](https://www.microsoft.com/en-us/research/project/spibb/).
We do not implement SPIBB or inherit its tabular safety guarantees.

| Claim | Current evidence | Required evidence / risk |
|---|---|---|
| PRS has avoidable delayed energy cost | Phase3 gap alone is insufficient | Paired suffix diagnosis, not oracle paper results |
| Improvements can be learned causally | Not established | Whole-trajectory holdout plus closed-loop check |
| R1 resolved | No | Reliable advantage over retained strong baselines |
| Full revision ready | No | Valid method + new R2/R3 + four-page marked/clean/rebuttal |

Use project-local existingvenv only, maximum2 worker processes, one library
thread each, no installs/SSH/host settings. New output folder refuses overwrite;
freeze inputs, store source/data hashes and preserve every previous failure.
