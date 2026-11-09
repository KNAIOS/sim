# Matched-size visitation coverage experiment

Frozen before new collection/training, 2026-09-22. The preceding goal turn made
progress: completed rollout/risk trials, independently verified results and
identified a concrete visitation gap. There is no live job from that turn.
R1 remains open; the complete major-revision goal and joint-control permission
remain unchanged. No installations, SSH or host environment changes.

## Question and fixed controls

Does adding states visited by the previous learned policy improve causal
closed-loop energy control, compared with adding the SAME NUMBER of PRS states?
This is a one-round data-distribution repair, not a new RL principle, a claim
that more data guarantees a win, or a license to weaken the search baselines.

Keep all previous physical parameters, 17 commands, causal CSI, readiness,
343 features, common power allocator and full-tail PRS rollout labels unchanged.
Keep all failed candidates and their frozen artifacts unchanged.

New training stream4109220000:8 exogenous trajectories/scenario,24 total.
New prediction-validation stream4209220000:4/scenario,12 total. Three existing
speed/load scenarios,140 slots, roots at0,16,...112. For EVERY trajectory run
two visitation policies: original PRS and fixed previous absolute-count critic.
For even trajectory indices use its seed11 checkpoint/price4; odd indices use
seed22/price2. Those choices were already frozen before these new trajectories.
Do not change the visitors after seeing collection or validation results.

At every sampled root execute all17 first commands for4slots then causal PRS
to the end; record4/16/full outcome labels, reference candidate, masks and
current features. Collect before-action active count explicitly. Labels share
only exogenous channels, not hardware/tracker/cache. No future is an input.
Thus72 behavior trajectories,576 root states and9792 candidate labels, arising
from36 independent exogenous trajectories, not72 independent trajectories.

Train two arms with equal384 root states:
- PRS-added:192 old training PRS roots +192 new PRS roots.
- Visitation-added:the SAME192 old roots +192 new learned-visitor roots.
Both get the same new192-state prediction-validation mixture (96PRS+96learned).
No old or new evaluation/test trajectory enters either training set.

Both use identical AbsoluteValue, two64-SiLU layers, energy SmoothL1 and
bounded absolute failed-count loss, AdamW.001/wd.0001, clipping5, batches16,
80epochs, seeds11/22. Normalize inputs from each training arm only. Save epochs
0/20/40/60/80, select each model by common prediction-validation loss only.
This is equal architecture/training budget and data quantity, not identical
input normalization statistics. Include the previous fixed critic separately.

## Control screen and stopping rule

Reuse the already-examined control-validation stream3909220000,4/scenario.
It is explicitly research/parameter-selection data, NOT an independent test.
For each arm/seed choose one price from{0,.25,.5,1,2,4,8,16,32}, using the
unchanged key:worst scenario violation above5%, mean energy, worst outage.
Preserve all scores. Retain the previous original/tunedPRS, full/coarse search,
full-array power and old absolute critic on these IDENTICAL trajectories;
their frozen results may be reused after input/output integrity checks.

Only advance Visitation-added to fresh development stream4009220000 if both
seeds satisfy5% in every scenario, mean nominal energy is >=1% below the stronger
original/tunedPRS, neither seed regresses versusPRS, and selected checkpoints
are noninitial. Report paired Visitation-added minus PRS-added and previous
critic differences regardless of this gate. A possible RL gain without a
coverage-arm advantage would not establish the coverage mechanism.

If passed, evaluate fixed selected prices/checkpoints and all retained baselines
on4009220000,4/scenario. Keep >=2% mean nominal saving, each-seed<=5% and no seed
regression as the development gate. Whole trajectory/seed uncertainty must be
reported; neither validation selection nor a small development screen proves
general superiority. If failed, stop this candidate's confirmation, diagnose
using existing train/validation outcomes and preserve the failed result.

## Verification and claims

Check fixed visitor actions reproduce the old implementation, zero-initial
critic equalsPRS, no root mutation/future feature leakage, reference rollout
and duplicate-mask identity, sample-count equality and disjoint trajectory IDs.
Independently reconstruct antenna-domain rates, hardware and energy for every
selected learned control episode. Freeze source and data hashes; use project
venv only, at most2 collection workers,1library thread each. Never overwrite.

Claim | Required evidence | Current state
---|---|---
Training covers low-activation states | actual before-action counts | to measure
Coverage changes control, beyond sample quantity | paired equal-size arms | unproven
R1 learned advantage | feasible win over retained strongPRS, fresh confirmation | open
R2/R3 for a new method | sensitivity and standardized-channel control | not inherited
Complete paper | evidence+four-page clean/markedPDF+consistent rebuttal | incomplete
