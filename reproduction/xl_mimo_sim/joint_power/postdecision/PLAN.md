# Post-decision hardware and counterfactual learning: bounded protocol

2026-09-22, before this branch's diagnosis/training/evaluation.
The previous goal turn was progress (shared calibration, two-seed value
learning, failed development screen and independently verified wake audit).
The full objective remains major revision, not merely a passing software test.
Route: communications / baseline, ablation, statistical, reproducibility /
unfair-comparison, overclaimed-results / planning.

## A. Training-trajectory diagnostic

Use the frozen phase2b selected policies and histogram, no weight updates.
Fresh training-diagnostic stream3109220000+scenario*10000+i, four trajectories
per existing scenario, both seeds11/22. Detect when the policy would close a
module that has just completed waking but has not yet served a data slot.
At each such decision, compare (a) its proposed command and (b) the existing
hold-command action, then continue the SAME frozen policy causally for up to
16 slots. Report paired energy, outage and discounted cost at4/8/16 slots,
keeping near-terminal truncated cases separate. Branches share an exogenous
trajectory but never mutable state. Future CSI is a simulated outcome label,
never an input to either branch's deployed decision rule.

Both branches have already received the common decision-time observation;
this comparison cannot reclaim past wake energy or eliminate the prior CSI
benefit. It tests a conditional subsequent hold decision, not "all wake-ups
were wasted". Never implement mandatory hold just because some branches win.

## B. Representation and Bellman update correction

Add112 explicit deterministic action features (16 modules x7): commanded mask,
active immediately after command, countdown immediately after command/4,
active at the next decision, next countdown/4, new wake flag, off flag.
Derive them only from current hardware and candidate command, not true future
channel. Existing causal features, calibrated energy/risk and common context
are retained. All policies still have exactly17 count-prefix/hold actions.
This is hardware state encoding, not an imposed minimum-on-time or action veto.

The previous n-step controller combines replay from multiple behavior prices
with three-step returns. Such returns include actions taken under a different
price and need not evaluate the target price's policy. The new candidate uses
ONE-step Double DQN with separately stored energy/outage; reconstructing the
immediate cost at another price is exact, with policy dependence confined to
the bootstrap. This removes that particular multi-step off-policy mismatch;
it does not guarantee convergence in this partially observed function class.

Two matched learning variants: PDS-one-step (one actual transition/decision),
and PDS-branch (all17 simulated action transitions per TRAINING decision).
All-action branching preserves each action's own hardware, observation age,
tracker update and resulting next candidate states. It does not insert a
common next state for different actions. No branching during deployment,
validation or development. Branching has higher offline simulation expense;
do not claim equal training compute or inherent novelty from Dyna/DDQN alone.

Both variants have the same price-conditioned value/advantage architecture as
phase2b (two64-unit SiLU layers per branch), expanded action input343, common
context216, discount.98, prices{1,2,4,8}, AdamW.0004/decay1e-5, Huber, clip5,
batch32, replay20000, target copy100, four updates/real decision after64 replay
records. Initial advantages zero and state value.6; the initial action ranking
is the shared immediate energy+risk cost. Exploration is unchanged from phase2b.

Train both variants for160 episodes, seeds11/22; identical root-training streams
3209220000+seed*100000+episode and scenario/price randomization. Actual visited
states differ by policy and are not interchangeable. Control validation:
3309220000+scenario*10000+i, four trajectories/scenario; checkpoints every40
episodes plus initial. Save ALL checkpoint weights and scores, not just best.
Pick checkpoint/price by worst scenario mean outage violation above5%, then
mean energy, then worst outage. This is development selection, not a guarantee.

Development stream3409220000, four trajectories/scenario; do not access until
all checkpoint/parameter choices are saved. Compare full-fixed, full-power,
PRS-power, uncalibrated joint eight-slot full/coarse, shared calibrated
four-slot full/coarse, myopic calibrated cost, frozen phase2b RL(two seeds),
PDS-one-step(two seeds) and PDS-branch(two seeds). Calibrated search uses all
histogram threshold breakpoints in[0,.1], chosen on the new control-validation
split. Retain both seeds, all scenarios and poor results.

## C. Progress/claim gate

Primary branch candidate: each seed <=5% mean outage in all scenarios; nominal
energy >=2% lower than PRS-power with neither seed regressing; noninitial
selected checkpoints. To credit branching, compare PDS-one-step at its own
validation-selected point and report reliability as well as energy. No claim
about PDS alone follows from comparing this changed trainer to old phase2b.
Any gain needs larger independent confirmation and eight-slot calibrated or
robust search, not only the weak four-slot comparator. Old R2/R3 cannot be
relabeled as new-method evidence. Failure stops expansion of this candidate,
not the full author-approved research objective; do not rewrite thresholds.

| Claim | Required evidence | Existing evidence | Missing / review risk |
|---|---|---|---|
| Immediate re-close wastes future energy | Paired causal-policy branch outcomes | Counts alone | A diagnostic; not a blanket claim |
| PDS encoding matches hardware | Every action/timer tested against simulator | None yet for new encoder | Unit and trajectory consistency |
| Branching improves learned energy | Matched one-step ablation, shared physical capabilities | Old variants failed | New two-seed screen; critical |
| R1 resolved | Reliable confirmed gain over strong search | Not established | Independent confirmation, not software QA |
| Manuscript complete | Methods/results/R2/R3/rebuttal and four-page marked/clean PDFs | Old four-page draft only | Full scientific work remains |

All work isolated in project-local code/results and existing venv. One BLAS/
Torch thread per worker, maximum two training processes; no installs, host
settings, SSH or system changes. Freeze protocols/source/weights; refuse
result-folder overwrite. Preserve old paper and all prior failures.
