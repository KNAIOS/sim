# Matched 4–16 command-domain experiment

2026-09-22, before stream 5209220000 is generated. Communications;
baseline/ablation/statistical/reproducibility; unfair-comparison; planning.

The user asks to retain the original PRS's 4–16 range. Keep that baseline and
test whether the current learning advantage survives an identical command range.
This is a fixed-controller action-domain ablation, NOT retraining or a newly
justified physical requirement. Four requested subarrays need not mean four
available subarrays during recovery. The physical simulator is unchanged.

Use 12 new development traces, four in each unchanged scenario, stream
5209220000. No matching source/manifest/plan was found. Freeze Visitation-added
epoch20, seeds11/22, alpha0.1, Energy admission, and every predictor/physical
parameter. Before proposal selection, discard all candidate commands with fewer
than four requested subarrays. Keep the previous command only if it meets the
same requirement. Evaluate predictions on the original feature tensor so the
network itself is unchanged. Reference/fallback remains original 4–16 PRS.

Configurations (11 x 12 =132 episodes):

- Energy Value, minimum4, seeds11/22: primary;
- Energy Myopic, minimum4, seeds11/22: energy-head ablation;
- originalPRS and budgetPRS, both minimum4, original parameters;
- Raw full, minimum4, threshold0.05;
- Energy Raw budget, minimum4, alpha0, matching existing non-learning admission;
- Full array with the shared analytical power allocation;
- Energy Value, minimum1, seeds11/22: attribution-only, not a matched comparator.

All methods share trace, forecasts, spatial ranking, actual power allocation,
hardware delay, energy accounting, and observation permissions. No artificial
search noise, coarse-stride change, parameter retuning, trace selection, or
removal of existing 1–16 search results.

Report all scenario/seed energies, outages, components, active counts, command
counts, and actual deviations from reference. Paired bootstrap resamples whole
traces and training seeds; four traces/scenario makes intervals descriptive.
Never treat two policies on the same four traces as eight independent traces.

Development screen unchanged: primary nominal energy at least2% below the
strongest feasible matched non-learning method, neither seed regresses, and
each seed's mean outage <=5% in every scenario. Passing does not establish
general superiority over 1–16 search or justify a system minimum of four.
Failure shows this fixed policy does not meet the criterion, not that a newly
trained minimum4 policy is impossible. Do not retrain in response without a
separate, bounded, hypothesis-driven protocol.

| Claim | Required evidence | This run | Missing / risk |
|---|---|---|---|
| Original saving is attributable to learning | Matched action domain | Fixed-policy ablation | No general superiority / critical |
| Long-term energy head helps | Same risk head, Myopic energy | Matched Myopic | Statistical uncertainty / major |
| Four subarrays are physically necessary | Architecture/service requirement | Not supplied | Must not invent / critical |
| Revision complete | R1/R2/R3, full four-page artifacts | Not supplied | Still open |

Test minimum1 equivalence to frozen implementation, masked reference validity,
minimum4 command invariance, fallback, and raw-search restriction. Independently
recompute every slot's antenna-domain rate and energy. Preserve old hashes and
the original paper; existing venv only, two local workers, no installs/SSH or
host changes. Save actual episodes and ledgers regardless of result.
