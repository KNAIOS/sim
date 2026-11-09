# Frozen finite-return tail diagnostic

2026-09-21. Author-only diagnosis under the authorized core reconstruction.
Detected axes: communications; ablation/statistical/reproducibility;
weak-ablation/overclaimed-results; planning.

## Claim–evidence contract

| Question | Existing evidence | Bounded check | Interpretation limit |
|---|---|---|---|
| Does a C16 target omit consequences of a root switch? | Actual free-return closed-loop screen failed | Replay the same root action followed by the same causal reference to C64 | A sampled tail, not conditional expected Q or learned-policy performance |
| Is omitted tail larger than existing prediction error? | Errors already exist at reference-policy states | Separately report prediction error, omitted tail and their signed MSE cross term | Cannot identify a unique cause of closed-loop failure |
| Is a longer target automatically beneficial? | No evidence | Report all C4/C16/C32/C64 views and both fixed models | No training, checkpoint, horizon or price selection |

## Frozen scope before execution

- Unchanged validation configuration, physical model, power, demand, recovery,
  ranking and reference policy. Do not edit old manifested code or data.
- Viewed validation stream 1020000000 + scenario*10000 + trajectory,
  scenarios 0/1/2, trajectory indices 0/1/2/3; root times 0/32/64.
  Twelve exogenous trajectories and 36 roots, not 36 independent trials.
- Sixteen prefix root actions each held four slots. Thereafter the frozen
  per-scenario cost-reference mu chooses causally every four slots, using its
  own branch observations/hardware. Each branch runs 64 slots, within T=140.
  Total: 576 branches / 36864 branch physical slots plus reference replay.
- Existing `branch_return(..., slots=64)` generates sixteen 4-slot blocks.
  From these SAME blocks compute discounted C4/C16/C32/C64 returns with
  gamma=0.98 per four-slot decision. Energy is normalized by four full-array
  slot energies; outage is the mean user-slot outage per block. Return sums
  are NOT episode joules or outage percentages.
- Before future labels are evaluated, freeze both free model epoch-100
  C16 choices (seeds 11/22, price=2) using archived causal 1500-D inputs.
  No projected scores or new fitted parameters. Check weights reproduce
  archived batch predictions exactly; reference actions and replayed causal
  features must equal archived entries exactly.
- Check C4/C16 returns against archived labels for every action (<1e-10),
  reference branch against actual reference suffix (<1e-10), unchanged root
  features after counterfactual replay, complete unique rows, finite data,
  source/data/checkpoint hashes unchanged.
- Store every action's block costs, every horizon, both models' original
  predictions/choices, all scenarios and adverse results. Count directional
  cost reversals with a 1e-6 zero tolerance. Descriptive summaries only;
  no p-values or per-root independence assumption.
- For selected actions, e=predicted_delta_C16-true_delta_C16,
  z=true_delta_C64-true_delta_C16. Then the C64 error is e-z and its MSE
  equals mean(e^2)+mean(z^2)-2*mean(e*z). Report all terms and identity error.
  Also report energy and outage components separately, not only price-2 cost.
- One single-thread worker using existing package venv; no installations or
  host/server environment changes. Unit tests before execution; stop if a
  consistency check fails. No automatic C64 training or confirmation gate.

R1 remains open regardless of this diagnostic. Promotion requires a separately
frozen bounded candidate, fair closed-loop comparison and untouched confirmation.
R2/R3 currently apply only to the original manuscript model.
