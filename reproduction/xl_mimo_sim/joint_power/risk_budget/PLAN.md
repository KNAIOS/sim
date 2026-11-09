# Direct energy selection under an estimated remaining failure budget

Frozen before control evaluation,2026-09-22. Prior turn made progress: matched
visitation-data experiment and audit completed, but its advantage gate failed.
All prior jobs are terminal. No installs, SSH or host changes. Keep the entire
major-revision goal active, all prior failures and the four-page manuscript.

## Hypothesis and exact rule

Test whether separating energy minimization from estimated reliability filtering
addresses a limitation of the common scalar penalty. Reuse BOTH frozen
aggregation critics (PRS-added, Visitation-added),seeds11/22,epoch20. No new
training or retrospective checkpoint selection. Original17commands, causalCSI,
power solver, energy coefficients, loads and channel model stay unchanged.

At decision t, let F_past be the integer number of failed user-slots ALREADY
observed. Total allowed count is delta*K*T=0.05*3*140=21. Available residual
budget is b_t=max(0,delta*K*T-F_past). It is a reliability ledger, not an
energy-saving target or fictitious energy debt. For each candidate a, the
frozen absolute critic predicts remaining failed count C_hat(s,a) under
one intervention then PRS, and relative remaining energy E_hat(s,a).

Keep candidates with C_hat<=alpha*b_t and choose minimum E_hat among them.
If none is feasible, choose minimum predicted count, then energy. Exact ties
prefer the referencePRS command. Update F_past only AFTER executing the4-slot
block; do not use hidden scenario labels, future channels, or future demand.
alpha is a disclosed empirical safety factor, NOT a calibrated coverage bound.
Grid alpha={0,.1,.25,.5,1}, one fixed choice per method/seed across all scenarios.

This is NOT the exact solution of a budgetedMDP: the value functions do not
depend on budget and evaluate a PRS tail rather than the new repeated policy.
An expected-count estimate is NOT a pathwise bound. Replanning and budget
accounting cannot by themselves prove5% reliability. Test actual outcomes.
Existing budgetedRL literature includes Carrara et al.,NeurIPS2019:
https://proceedings.neurips.cc/paper/2019/hash/4fe5149039b52765bde64beb9f674940-Abstract.html
We do not implement their budgetedBellman operator or inherit its guarantees.

## Fair comparisons and frozen screen

For EACH of the two critics and seeds, compare:
- Value rule: learned complete-tail energy on the learned-count feasible set.
- Myopic rule: common analytical first4-slot energy on the SAME learned-count
  feasible set. This isolates the energy-tail ranking, not the whole learner.

Two non-RL ledger controls get the SAME alpha grid and observed-failure ledger:
- Raw budget: eight-slot physical forecast outage timesK*(T-t),minimum
  physical eight-slot energy. This extrapolated risk is a heuristic, disclosed.
- PRS budget: originalPRS with threshold min(1,alpha*b_t/(K*(T-t))).
Neither gets added noise, extra switching, different channels or power limits.
Also retain all previous original/tunedPRS, full/coarse, full-power and fixed
price learning controls on the same traces, reusing frozen checked results.

Control-validation stays3909220000,4/scenario; it is repeatedly examined research
data, NOT independent testing. Eight learned controllers and two non-RL controls,
each5factors over12traces =>600new candidate episodes. Select by unchanged
worst-scenario5% violation,meanenergy,worstoutage key, then smaller factor.
Preserve all600scores and all120new selected episodes plus156old references.

For each Value family separately, core gate: both seeds<=5% in every scenario;
nominal meanenergy>=1% lower than the strongest feasible retained NONLEARNING
baseline; neither seed regresses versus that reference; noninitial checkpoints.
Baseline feasibility is checked across all3scenarios, not merely nominal.
Report each matched Myopic ablation even if better. Lack of Value superiority
means no evidence for learned-energy-tail contribution, even if core gate passes.

Only if a Value family passes, freeze its choice and run fresh development
4009220000,4/scenario alongside retained strong baselines and matched Myopic;
require>=2% nominal saving, everyseed/scenario<=5%, no seed regression. This is
still development evidence, not automatic manuscript readiness. If neither
passes, stop confirmation of this rule; do not keep changing alpha or scenarios.

## Verification and claim-evidence obligations

Unit tests: candidate feasibility before energy ranking, minimum-risk fallback,
reference tie, causal feature/command independence from future data, ledger
updates match actual integer failures and cannot restore spent budget.
Independent antenna-domain rate/hardware/energy reconstruction for all120new
selected episodes; original artifacts/weights/results preserved by hashes.
Record per-decision budget, feasible-set size, prediction and past failures;
small-sample uncertainty uses whole trajectories and training seeds.

Claim | Evidence required | Before this stage
---|---|---
Filtering avoids unearned risk credits | behavior and energy-risk outcomes | hypothesis
Learned energy tail adds benefit | matched Myopic ranking comparison | unproven
RL has reliable energy advantage,R1 | core gate then fresh confirmation | open
R2/R3 for updated algorithm | sensitivity/calibration+standard channel | not inherited
Whole revision finished | scientific evidence+four-page clean/marked+rebuttal | incomplete
