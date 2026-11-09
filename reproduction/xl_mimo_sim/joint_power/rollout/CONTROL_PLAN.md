# Absolute-count Monte Carlo critic and actual-constraint control screen

2026-09-22, after the original rollout diagnostic/probe, before new training
or access to the control/development streams below. Preserve all original
choices, including both epoch0 failures. This is a new explicit research stage,
not a retroactive success under the previous zero-extra-failure criterion.

The original difference critic repeatedly predicts unrealized negative failure
differences. Huber regression of signed return differences is not a calibrated
conditional-mean failure model, especially with rare large losses. This evidence
motivates testing a physically bounded ABSOLUTE count head, not assuming success.

Reuse the frozen24 train/12 probe-validation trajectories from3609220000 and
3709220000. No development traces enter training. Same two64-SiLU-layer network,
seeds11/22,80epochs, AdamW.001/decay.0001, clip5, batches16 complete states.
An energy head predicts the reference-anchored energy difference divided by
full-slot energy. A risk head predicts q∈(0,1), yielding expected failed-user-slot
count Nq, N=users*remaining_slots. Train with energy SmoothL1 plus
`-F log q -(N-F)log(1-q)` averaged over state/action, where F is observed count.
The latter estimates a conditional mean under population risk; it is not a
claim that correlated user-slots are independent Bernoulli observations.
No calibration, confidence or safe-policy guarantee is inferred from the link.

Save epochs0/20/40/60/80; choose checkpoint by probe-validation prediction loss,
not its action outcome. The paired signed-critic ablation uses its checkpoint
with the lowest original probe-validation Huber (all epochs, including0).
This isolates output/target representation more closely than comparison with
old multi-stepDQN, but distinct loss functions and initialization are disclosed.

For deployment, both networks choose minimum
`deltaEnergy/fullSlot + lambda*deltaExpectedFailedUsers/users`, and exactly
retain PRS on zero/tied advantage. The absolute-count method computes the risk
difference only AFTER subtracting two nonnegative bounded predicted counts.
This does not prevent all prediction error or guarantee dominance over PRS.

New causal closed-loop control-validation stream3909220000,4 trajectories in
each existing scenario. Prices{4,8,16,32}, one fixed price per model/seed selected
by worst-scenario outage violation above5%, then mean energy, then worst outage.
No extra-failure-relative-to-PRS condition is imposed: it was a deliberately
strict diagnostic probe, not the manuscript's original5% operational constraint.

Tune each raw full/coarse and originalPRS reliability threshold over
{0,.025,.05,.1} on the SAME control-validation trajectories by the same rule.
Physical constraints, information, power allocation and energy accounting are
unchanged. Keep untunedPRS(.05) as a separate reference too. Include full-power.
This does not add observation noise or lower a baseline after seeing outcomes.

Only if both absolute-critic seeds are feasible and their nominal mean energy
is at least1% below the stronger of tuned and originalPRS on control validation,
with neither seed regressing, run fresh development stream4009220000,4/scenario.
Otherwise stop this candidate before further confirmation; retain all results.
If run, development compares both absolute and signed critics(two seeds each),
tuned/rawPRS, full/coarse joint search and full-array power, and retains the
original>=2% nominal saving, each-seed<=5% outage, no seed regression gate.

New outputs in`results/joint_power_20260922_rollout_absolute`, never overwrite.
No system changes/installs/SSH. This is bounded diagnosis/training/control
screening, not scientific completion or an excuse to omit newR2/R3 validation.
