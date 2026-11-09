# One bounded policy-consistent value reconstruction

2026-09-21. Frozen before collection/training/new validation. Detected axes:
communications; baseline/ablation/statistical/reproducibility;
unfair-comparison/weak-ablation/overclaimed-results; planning.

## Mechanism and claim–evidence contract

The previous cost labels evaluate a first action followed by reference search mu,
whereas deployment repeatedly uses the learned policy pi. The old model remains
failed. This is a NEW single approximate-policy-improvement candidate, not an
extension of its old promotion gate. A teacher can recover from states in ways
that the learner cannot; its continuation return is not the learner's return.
The long-horizon audit does not establish this as the unique failure cause.

| Question | Evidence required | Fixed experiment | Limit |
|---|---|---|---|
| Does continuation mismatch change targets? | Same state/first action/future, different subsequent policy | Paired C16 labels under mu and frozen predecessor pi | Sampled finite returns, not exact expected Q |
| Does own-state coverage help? | Same architecture/updates/continuation, change roll-in states | mu-only versus equal mu/pi mixture | Different unique state count is intentional; no extra exogenous trajectories |
| Does policy-consistent evaluation help energy control? | Factorial target ablation and fair closed-loop comparisons | Four variants, two seeds, fresh fixed validation | Validation screening, not final significance or safety guarantee |

## Literature and novelty boundary

[Ross and Bagnell, 2014](https://arxiv.org/abs/1406.5979), Algorithms1/2 and
Sections2.3/2.5/3, explicitly distinguish expert-continuation cost-sensitive
learning and current-policy return evaluation; they also discuss optimistic
expert returns when the learner cannot replicate recovery. This is a motivation,
NOT a new generic RL algorithm or a theorem we have established for this POMDP,
neural approximator, one iteration or truncated return. No-regret guarantees
are not imported. [DAgger](https://proceedings.mlr.press/v15/ross11a.html)
similarly motivates policy-induced state coverage, not our performance claim.

The earlier235-input action-classification DAgger experiment is not restarted.
This experiment learns two actual physical costs with causal1500-input Gram
features, using counterfactual returns rather than copied expert action labels.

## Frozen collection

- Existing validation physical configuration, three speed/load scenarios,
  16prefix actions,4-slot decisions,4-slot recovery,T140,originalpower/RZF/CSI.
- mu configurations remain those used in the original value labels; pi is the
  frozen free epoch100 model for seed11 or22 at lambda2. No retuning mu/lambda.
- New TRAIN stream1080000000+scenario*10000+i, i=0..3:12exogenous trajectories
  shared by the two seeds. Each seed rolls in separately with mu and pi.
  Roots0,16,...112:96roots per roll-in,192per seed.
- At every root, all16first actions are held4slots; then either mu or frozen pi
  controls the next12slots. Both continuations use branch-owned causal inputs.
  Label shapes[192,2continuations,16actions,2horizons,2costs]. First-block C4
  must match across continuation choices; C16 discount remains gamma.98/block.
- Per seed6144branches /98304branch slots; total12288branches /196608slots.
  Shared mu/mu labels are deliberately repeated and checked equal across seeds;
  repeated labels are not independent evidence. No validation labels enter fit.
- Reference (mu action) and policy actions, inputs, all label costs and roll-in
  episode metrics retained. Own-policy suffix and mu suffix replay checks<1e-10,
  exact old branch_return parity for mu, unchanged snapshot, finite labels,
  source/data/weight hashes. Maximum two single-thread existing-venv workers.

## Exactly one training update, factorial controls

All four variants start from each SAME frozen free epoch100 checkpoint, create a
new AdamW optimizer(lr4e-4,wd1e-5),clip5,batch64. Same free1500-input32324-parameter
network, paired absolute+reference-relative squared losses and original training
scales. No new architecture, risk coefficients or future inputs.

| Variant | Training roll-in roots | Continuation target | Role |
|---|---|---|---|
| mu_mu | mu only | mu | additional-fit control |
| mix_mu | equal mu/pi | mu | state-coverage change only |
| mu_pi | mu only | frozen pi | continuation change only |
| mix_pi | equal mu/pi | frozen pi | proposed one-step policy improvement |

Only new108training labels used in this update: old mu labels are not incorrectly
mixed into a pi-value target. Old data influence initialization/loss scales only.
Each variant gets100epochs,192root presentations/epoch,300updates. mu-only uses
two independently permuted passes of96roots each epoch; mixture uses one192root
pass. Eight runs total; fixed epoch100 saved, no endpoint selection, second policy
iteration, early-stop search or additional fitting. Record training diagnostics
but do not use training accuracy as deployment evidence.

## Frozen fresh validation and strong controls

- Stream1090000000+scenario*10000+i,i=0..7:24shared external trajectories;
  not generated/opened until all eight fits complete.
- Evaluate four variants and unchanged predecessor, two seeds:240episodes.
- Four searches(prs,cost,current-prs,current-cost),12settings each(min1/4,
  margins.90/1/1.04/1.08/1.16/1.24):1152episodes. Plus mu/full:48episodes.
  All keep the same observation/action access; current versions use exact current
  active CSI and unchanged causal future forecast. OriginalPRS usesh1..8, current
  variants physicalt..t+7; preserve this disclosed difference.
- Strongest pure feasible search selected per scenario; also finite49-point
  episode-mixture frontier at each candidate's mean outage and5%. Mixture means
  are validation-fitted expectations, NOT independently executed stochastic tests.
  Baselines get more validation tuning than fixed lambda2 learned variants;
  this is a conservative screen, not equal-tuning-budget universal ranking.
- Total1440episodes on24exogenous traces. Report all variants,seeds,scenarios,
  energyJ,outage user-slot fraction,energy decomposition and paired differences.
  No root/seed duplication of baseline independent sample count; no significance
  claim. Fresh validation becomes viewed after use, not called untouched test.

## Promotion and stopping (primary scenario fixed to nominal)

Only mix_pi is a proposed candidate. Both seeds must pass NOMINAL scenario:
outage<=5%; energy<=98% of strongest feasible pure search; outage no more than
that search+0.5percentagepoints; energy<=98% of the finite search-mixture frontier
at the candidate's own outage. For each same-seed ablation and predecessor that
meets5%, mix_pi must have lower energy and outage<=control+0.5percentagepoints.
If a control violates5%, mix_pi feasibility suffices for that comparison.
No feasible pure/frontier search means no automatic promotion.

The low/high scenarios are reported boundaries, not alternate success searches.
No price sweep, extra iterations or passing ablation renamed as the main method.
Failure stops this candidate. Passing permits only separately frozen independent
confirmation; new sensitivity/standard-channel evidence and manuscript/figures/
yellow revisions/rebuttal still required. R1 is not closed by this screening.
No host/server installs, global variables, driver changes or submission actions.
