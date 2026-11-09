# Baseline-anchored decision admission, bounded paired ablation

2026-09-22, before this controller is evaluated. Previous turn made progress:
floor correction was verified but failed locked development. All old jobs are
terminal and old outputs remain unchanged. This is a different explicit decision
rule, not retrospective retuning of the failed physical-floor projection.

## Evidence and hypothesis

On the already inspected development records of Floor Value, low/nominal/high
each contain280 decisions. Estimated feasible sets are empty10/70/122 times;
4/55/75 of these also have strictly positive estimated energy advantage relative
toPRS. This is NOT proof these actions truly waste energy: extra energy could
buy reliability. It identifies a measurable source of decisions for which the
energy-saving rationale is absent and risk is only an uncalibrated prediction.

Retain the frozen candidate proposer and count-floor correction. Test two rules:
1. Empty fallback: if no predicted-feasible candidate exists, execute originalPRS;
   otherwise execute the unchanged proposer.
2. Energy admission: additionally require proposed relative energy < -1e-8;
   otherwise execute originalPRS. This tolerance is numerical, not a tuned margin.
The same two rules apply to Value, Myopic and non-learning Raw budget proposers.
Myopic keeps the same learned-risk head, ranking by first-block physical energy;
Raw budget uses physical forecasts only. They isolate the long-term-energy
ranking and the contribution of fallback, not all neural representation at once.

An exact reference-policy Q can support policy improvement for an appropriate
fixed objective; estimated energy admission with a risk budget does NOT inherit
that result. The present head is not conditioned on budget and predictions can
be wrong. ExecutingPRS from a state reached by learning is NOT equivalent to
starting a whole episode withPRS, and cannot guarantee safety or lower energy.

Prior work on conservative baseline use includes Laroche etal.,ICML2019,
https://proceedings.mlr.press/v97/laroche19a.html. We do not implementSPIBB's
uncertainty/count constraints or assert its performance guarantee. BudgetedRL
https://proceedings.neurips.cc/paper/2019/hash/4fe5149039b52765bde64beb9f674940-Abstract.html
also does not imply that an empirical budget ledger solves a budgetedBellman
problem. Neither baseline fallback nor adding budget is claimed novel by itself.

## Fixed experiment

No retraining or parameter search: Visitation-added seeds11/22,epoch20;
alpha0.1 for Value/Myopic, Raw budget alpha0, originalPRS fallback threshold.05.
All17 commands, energy/power models and information access are unchanged.
Use both previously inspected streams3909220000 and4009220000, four trajectories
per scenario. Two learned proposers×two rules×two seeds plus twoRaw rules:
10 configurations×12traces×2streams=240 NEW episodes. These are now both research
evaluation sets, NOT a second independent confirmation or final test.
Preserve every old strong baseline, both previous failures and all new results.

Primary candidate is Energy admission Value. Empty fallback Value is an ablation,
not a posthoc replacement if it performs better. Screening condition on EACH
stream: every seed's scenario mean outage <=5%, mean nominal energy at least2%
below the strongest feasible non-learning reference and neither seed regresses.
Use all previously retained non-learning methods plus both new guardedRaws as
references. Report Myopic comparison separately; passing the search screen does
not establish a long-term-value contribution. No further grid on failure.

If the primary passes BOTH streams, it may go to new locked independent
confirmation with unchanged settings; otherwise stop this candidate's expansion.
Do not pick the favourable stream or choose a different primary after seeing
outcomes. Do not install packages, accessSSH or edit host environments.

## Evidence to save

Per decision save proposal and executed commands, admission reason, predicted
relative energy, estimated feasibility, actual budget ledger and executed
physical failure floor. Independently recompute all rates/energy from actual
antenna-domain records. Save per-episode/component/seed/scenario means, pathwise
target exceedances, paired effects and descriptive whole-trajectory intervals.

| Claim | Required evidence | This study | Remaining limitation |
|---|---|---|---|
| Empty fallback matters | fixed proposer before/after | paired complete episodes | repeated research data |
| Energy admission matters | primary versus empty-only | matched settings and two seeds | estimate, not proven advantage |
| Long-term learned value matters | Value versus Myopic with same admission | all metrics reported | shared learned-risk head |
| Beats credible search | strong original and newRaw controls | both full research streams | requires fresh confirmation |
| Revision complete | robust advantage, R2/R3 and four-page source/PDF/rebuttal | not claimed by screen | not yet achieved |
