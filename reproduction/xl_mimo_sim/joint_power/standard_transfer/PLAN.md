# Frozen current-candidate UMi transfer protocol

2026-09-22, locked before new channel generation. Axes: communications;
baseline/robustness/statistical/reproducibility; unfair-comparison and
overclaimed-results; planning/reviewer-response.

Question: how does the implemented 343-input energy/risk candidate transfer from
the controlled geometric simulator to a standardized frequency-selective fading
generator? This is R3 evidence and not a new search for an R1-winning setting.

Use the unchanged existing Sionna-no-rt UMi generator (TR38.901 v16.1, pinned
existing venv). Twenty independent drops each of LOS and NLOS, new channel
stream5809220000 + state*10000 + index. New demand stream5909220000 similarly.
Retain both offered-load factors .5 and1 on every drop. Forty channel/demand
pairs, eighty load cases, not eighty independent channel drops. Seeded topology
is fixed during each 140-slot episode; 2m/s Doppler,99 frequency bins,14GHz40MHz,
16 contiguous eight-element columns of the compact panel. No normalization.
The compact 0.1606m aperture replaces the geometric 12m aperture in features.
Do not call this near-field moving-VR calibration or measured validation.

Keep Visitation-added epoch20 seeds11/22, alpha.1, default ranking (.65,.93),
minimum command4, original centre-frequency features, reference, admission and
model weights. Only the current failure lower bound is evaluated with the same
wideband PHY as execution; otherwise a centre-bin failure is not a valid lower
bound. No retraining, load/seed selection, new default or stopping on favorable
outcomes. A different final policy requires its own transfer evidence.

Nine executions per load case:
- Value4 seeds11/22 (primary).
- original PRS4 and budgetPRS4.
- expanded PRS1 and budgetPRS1.
- Current-WB budgetPRS4 and Current-WB budgetPRS1.
- Full-array shared-power reference.
Thus720 episodes and100800 physical slots. Domain1 references are clearly
expanded comparisons; they do not establish a hardware minimum4.

Current-WB budgetPRS is an explicitly named additional reference, not a silent
replacement of original PRS. It uses the same causal spatial ranking and count
domain, but evaluates an eight-slot window containing the actual current slot
and the first seven original future forecasts. The actual current rate uses
observed active per-bin CSI and the shared allocator; newly requested sleeping
modules remain unavailable. The seven future centre-bin rates are forecasts,
not actual future channels. Choose the smallest count meeting the residual
outage threshold; if none, minimize normalized shortfall, as in PRS.
This exposes the same current wideband information to both learned and search
selection while retaining original search results. It is not an optimal
wideband MPC or a reproduced published algorithm. No runtime advantage claim.

Every execution uses per-bin RZF, user-total flat-PSD power allocation and the
same energy model, perfect current active CSI, recovery delay and pilot cost.
No current sleeping CSI, future channel or future demand is used for decisions.
Save every slot, decision, summary and pairing; include power-solver fallback
counts and all failures. Independently replay antenna-domain rates and hardware
energy per slot, and check lower bounds against realized block failures.
Save all seeds/channel hashes, one complete example per state, pinned versions,
model hashes and source manifests. Existing files and host environment intact.

Report energy J and rate-outage fraction by state/load/method, all four cases.
Use4000 paired crossed seed/drop bootstrap replicates for Value4 against every
reference, separately within each state/load. Only two model seeds; intervals
are descriptive, not multiplicity-adjusted or per-episode safety guarantees.
Do not filter overloaded NLOS or unfavorable seed outcomes. No parameter gate
or superiority stopping rule is applied to this transfer study.

Claim | Required evidence | This experiment | Remaining boundary
---|---|---|---
R3 standardized transfer | current policy, common physical channel, strong retained references | planned720 fixed evaluations | not XL-VR/field calibration
R2 heuristic sensitivity | current-policy five-point paired sweep | completed previous phase | no probabilistic calibration
R1 learned energy advantage | matched constraints and reliable incremental benefit | report without prespecified victory | earlier failed confirmations remain
Final paper | coherent four-page clean/marked source and response | transfer data for integration | not automatically complete

Use only existing package venvs, at most two single-thread local workers. No
installation, SSH, global environment edits, external submission or old-data
overwrite. Cached fixture tests must pass before generating the new streams.
