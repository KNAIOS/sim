# Frozen-controller heuristic sensitivity for R2

2026-09-22, frozen before generating the new stream. Axes: communications;
baseline/robustness/statistical/reproducibility; unfair-comparison and
overclaimed-results; reviewer-response. Prior turn changed the manuscript
working state and reference evidence; no previous job remains live.

Question: how do ranking uncertainty and forecast discount change the energy /
outage tradeoff of the implemented 343-input candidate? This tests sensitivity,
not calibration, optimized parameter selection, novelty or a new R1 victory.

Keep Visitation-added epoch20 seeds11/22, risk factor.1, Energy admission,
physical failure lower bound, power solver and all physical settings fixed.
Five one-factor-at-a-time (beta,gamma) points, selected before outcomes:
(.65,.93) default; (0,.93); (1.3,.93); (.65,.8); (.65,1).
Do not retrain, choose a new default or discard settings after seeing results.

New exogenous stream5709220000, eight traces per original speed/load condition,
24 independent traces. All settings and methods share these traces. Keep
Value4 and Value1 (each two frozen seeds), original PRS4 and budgetPRS4,
Expanded PRS1 and budgetPRS1. Full-array shared power control is invariant to
ranking parameters; run once per trace and disclose reuse as a reference.
Thus 40 parameter/method evaluations plus one full-array episode per trace:
984 executed episodes,137760 physical slots. Repeated seeds/settings are not
984 independent samples.

Control parameters change both RL candidate ranking and all PRS ranking / rate
forecasts identically. Count restrictions apply to commanded masks, not minimum
available antennas during recovery. The Value4 ablation changes action admission,
not what domain the network was trained on. Original4 and stronger1 references
are both retained, including adverse outcomes. No full-array speed advantage.

Save every episode and slot, model hashes, settings and source manifests.
Independently reconstruct antenna-domain rates, hardware and all energy terms
for each episode. Require unchanged raw channel/demand arrays; command-domain
and cumulative-failure checks; baseline command equivalence on default settings.
Report energy J, rate-outage fraction and energy components. For learned policies,
cross seed/trajectory bootstrap 4000 paired replicates against the default and
against same-range budgetPRS. Report all conditions and CIs, not just winners.
There are only two training seeds; intervals have this limitation and are not
simultaneous multiple-comparison certificates.

Evidence | Required | Supplied here | Remaining
---|---|---|---
R2 heuristic sensitivity | fixed-policy parameter perturbation with matching controls | planned five-point paired sweep | not probability calibration or joint tuning
R1 RL advantage | credible same-domain energy/service advantage | no prespecified superiority claim | earlier unsuccessful confirmations remain
R3 channel validation | current policy on standardized frequency-selective channels | not this experiment | separate correct wideband adapter and protocol
Final paper | consistent four-page clean/marked manuscript and reply | results can supply R2 paragraph/table | not declared complete

Use existing project venv, at most two local single-thread workers. No dependency
installation, SSH, system settings, old artifact overwrite or publication.
