# Paired performance-difference identity audit, after failed control screen

2026-09-22. Both Value families failed the frozen screen. Do not extend the
factor grid, rerun training, consume development data, or rename failure.
Instead diagnose the existing selected Visitation-addedValue policy (2seeds,
12already-examined trajectories) at ALL35actual decision states per episode.

At each causally visited state, compute current command and predicted relative
energy/absolute failed counts BEFORE looking at future labels. Then simulate:
B=PRS from that root until the end; Q=the actual selected4-slot command then
PRS until the end. Both branches have separate hardware/tracker/cache.
Use the same exogenous future solely as diagnostic labels. After labeling,
continue the real visited trajectory with the unchanged budget policy.

For each trajectory verify the exact sample-path telescoping identities:
sum_i[Q_energy(s_i,a_i)-V_B_energy(s_i)] = E_actual_policy - E_B(initial),
sum_i[Q_failed(s_i,a_i)-V_B_failed(s_i)] = F_actual_policy - F_B(initial).
At the final block continuation is zero. Because these are the complete actual
visited states and the same deterministic trace, this sum IS an accounting
identity, unlike summing alternative interventions at states of an unchanged
baseline. Explain that distinction; it does not prove causal predictability.

Record predicted minus realized selected advantages, current budget, hardware
switch counts and the paired branches. Compare absolute chosen-risk prediction
to the true chosen-first-action/PRS-tail label, not to a different learned tail.
No population calibration or safety guarantee from these noisy realized labels.

This is24policy trajectories,840related roots,at most1680suffix simulations
(equal commands may reuse exact labels),NOT840independent episodes. Allstates
are retained, including equal commands and non-beneficial deviations. Match
all resulting real episode metrics to the frozen selected control records.
Use2workers max,projectvenv,one library thread each; no host changes.

Purpose:locate whether remaining performance loss reflects systematically
misvalued selected actions, rather than infer it from scalar training loss.
Only use this evidence to design a NEW explicitly frozen model/target revision;
do not turn the future-informed labels into deployable performance claims.
