# Physical failure-floor correction: frozen paired check

2026-09-22, recorded before evaluation. This is a small repair experiment,
not another parameter search and not an independent final test.

## Hypothesis

The phase-six risk estimator sometimes predicts fewer remaining failed
user-slots than already determined by the current observable channel and
candidate command. Repair this inconsistency by C_plus=max(C_hat,U), where
U is the exact current-slot failed count plus forced all-user failures in
later slots of the four-slot block with no active subarray. Positive demands,
positive wake delay and perfect current active-array CSI are assumed, as in
the existing simulator. U is a lower bound, NOT a safety upper bound.

Keep both frozen training arms, seeds 11/22 and selected epoch20, selected
phase-six alpha for EACH controller, 17 candidates, ranking, power allocator,
channel/traffic/power/energy settings, tie breaking and fallback unchanged.
Apply the same max correction to Value, Myopic and Raw budget count estimates.
Original PRS and PRS budget remain intact: they use a different analytic rate
threshold rule, not the learned absolute-count head. This repair introduces
no information unavailable to PRS, which also observes current active CSI.
It does not justify omitting any stronger original method.

## Scope and stopping rule

Run nine fixed controllers (eight learned, one Raw budget), twelve shared
previously examined control-validation traces, hence 108 new episodes. No
alpha or checkpoint selection, no new training, no mixed-in training traces,
no weakened or noisy baseline, no final-test access, no host modifications.
Append all 276 frozen previous episode rows as explicitly named references.
Record every decision, including uncorrected action on the SAME visited state,
selected floor, raw and projected count and number of corrected candidates.

For all episodes independently recompute physical rates and energy; verify
projected count>=floor and floor<=realized block failures, and integer ledger.
Compare energy AND outage against the uncorrected controller, matched Myopic,
original PRS and budget PRS. Report per-scenario, per-seed values, trajectory
paired differences and descriptive resampling intervals. No claims of
independent significance from reused validation and four traces/scenario.

Keep phase-six screening criterion: both seeds feasible (<=5% scenario mean),
nominal mean energy at least 1% below the strongest feasible non-learning
reference, and neither seed worse than that reference. Include corrected Raw
budget among potential references; original strong baselines remain. Passing
would only motivate independent confirmation, not prove superiority.

If it fails, do not tune this projection: conclude whether physical consistency
was repaired and whether that explains the energy gap. Do not expand parameter
grids or relabel the failed run as a win. Retain all artifacts and stop this
candidate's performance expansion. The overall major-revision claim remains
open until actual adequate evidence or a justified narrower manuscript exists.

## Claim-evidence matrix

| Claim | Required evidence | This check | Remaining limit |
|---|---|---|---|
| Physical count consistency | causal bound and all-action/unit checks | tests plus every executed decision | perfect-current-CSI assumption |
| Better control due to correction | frozen paired before/after energy and outage | 108 new episodes | repeated validation, small sample |
| Learned long-term energy is useful | matched Myopic and strong non-learning baselines | all retained | no guaranteed advantage |
| Joint energy-saving benefit | same variable-power solver for all methods | existing joint-control environment | power is analytical, not a learned continuous action |
| Reviewer R1 resolved | independent reliable gain over credible search | screen only | cannot claim resolution from this screen |
