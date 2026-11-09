# Matched frozen-policy power headroom audit (2026-09-22)

Detected axes: communications; baseline/statistical/reproducibility;
unfair-comparison/overclaimed-results; audit.

Previous goal turn: progress (separate official-template format repair and QA).
This turn addresses a remaining scientific question, not another format change.

Question: does the SAME fixed-beam power adjustment narrow the energy gap between
the original manuscript's five frozen RA-DQN policies and predictive search?
The earlier headroom audit included search and full-array schedules, not these
five original learned policies. Search-equivalent recent candidates are not
called new RL benefits here.

No author approval is inferred for joint-control integration. This is an
evaluation-only, fixed-schedule diagnostic; original simulator, manuscript,
weights, physical parameters and decision rules are untouched. No training,
retuning, test selection, new environments, packages, or remote processes.

## Frozen protocol before execution

- All24 already-viewed stream121 trajectories,8 per scenario (.5m/s,.85load),
  (2m/s,1load),(3m/s,1.15load),140slots,4-slot decision and recovery times.
- Original checkpoints RA-FVV_11/22/33/44/55 from revision_20260905. No selection
  across seeds. Exact manuscript predictive-rate search (counts4..16) is the
  primary comparator.144 new policy replays total; no newly claimed test set.
- Also show existing calibrated-search and full-array power-accounted records
  on exactly the same trajectories (48 archived episodes). Their provenance
  hashes and previous independent verification must pass. Calibrated search
  has a different expanded count domain/risk model; label it a stronger
  secondary comparator, not a matched-action-space ablation.
- At each slot the new allocator reads current active-branch CSI only. Same
  solver/targets/limits as POWER_HEADROOM_PLAN.md for every policy. It cannot
  feed into the policy or change its subsequent commands/observations. Original
  successful users remain successful and failed users keep at least their old
  rate, with exact failure-indicator identity. Excess achievable rate may fall.
- Retain all actions, powers, per-user rates, failed slots and numerical
  fallbacks. Save fixed-schedule original/adjusted energy and rate-outage means.
  New allocator computation energy, imperfect CSI and variable PA efficiency
  are not modeled, exactly as in the previous conditional diagnostic.
- Verify all144 original energy/outage/count trajectories against the original
  simulate_episode implementation. Independently reconstruct saved transmit
  amplitudes with original channel/RZF equations without calling the allocation
  solver; check all20160 slots and unchanged source/checkpoint hashes.
- Summarize all3 scenarios and all5 learning seeds. Energy-gap interaction is
  (RL-search)_adjusted - (RL-search)_original; negative means the gap narrowed,
  NOT that the learned policy is superior. Report outage differences alongside.
- For descriptive intervals, resample5 learning seeds and8 paired trajectories
  independently (crossed bootstrap,10000 draws,seed20260923). No significance or
  deployment claim from this reused development set. Include per-seed means.
- Stop after the fixed audit, independent checks and report regardless of
  direction. No new training, hidden parameter scan, or manuscript substitution.

| Claim | Required evidence | This audit | Remaining review risk |
|---|---|---|---|
| Shared power correction changes the original energy gap | Same trajectories, all seeds, identical allocator | Measured fixed-policy paired gaps | Conditional, already-viewed data only |
| Learned joint activation/power control is more energy-efficient | Fairly optimized controls at matched reliability | Not tested | Critical: requires approved scope and fresh training/testing |
| R1 resolved | Independent reliable energy advantage over strong search | Not tested | R1 remains open |
