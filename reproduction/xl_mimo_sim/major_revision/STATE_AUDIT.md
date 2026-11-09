# Controller-state audit and bounded next candidate

## Confirmed limitation, not a performance explanation by itself

The original 72-dimensional state normalizes the spatial score and loses a
common absolute channel-gain change. The 139-dimensional pilot restores gains
relative to thermal noise, observation age and changes in demand. However,
it still does not retain inter-user channel correlation.

`test_rate_context.py` constructs two complex channel matrices with identical
element powers. One has aligned user rows, the other orthogonal user rows.
The 139-dimensional states are exactly identical, whereas candidate RZF rates
differ materially. An observation mapping with this alias cannot condition
its decision on that distinction. This is a constructive state-information
counterexample, not proof that it causes all losses on the geometric dataset.

## Candidate-rate context (implemented and development-tested)

Subsequent update: the independent confirmation failed the 5% target, and the
longer-budget/feasibility-first diagnostic still used more energy than search.
See CANDIDATE_CONFIRMATION_REPORT.md and CONSTRAINT_TRAINING_REPORT.md.
READINESS_STATE_ALIAS.md now gives an additional constructive alias in the
178-dimensional representation: fully-ready prefix rates do not determine the
actual available-mask rates after a command. The correcting feature routine is
implemented and unit-tested. Its subsequent 600-episode matched ablation is now
complete: see READY_ABLATION_REPORT.md. It does not establish an energy advantage
over either search baseline, nor a stable improvement over the 178-dimensional
learner. Resolving a constructive alias is not itself a performance result.

Update: the bounded two-seed ablation has finished; see RATE_ABLATION_REPORT.md.
It lowered high-load outage at higher energy. The small apparent energy benefit
over tuned search has a paired interval crossing zero and is not established.

For every ranked count m in 4,...,16, use the causal last-observed complex
channel estimate and compute the three-user RZF rates for that prefix.
Append clipped ratios estimated_rate / current_requirement (39 numbers).
The enriched state would grow from 139 to 178 dimensions. This communicates
both absolute link quality and multiuser separation without feeding future CSI.

These estimates assume a fully ready candidate; they are NOT certified
future rates, nor does the implementation itself resolve wake delay. Existing
hardware countdowns remain needed. Sleeping-branch phases are stale and no
new measurements are available until activation. A proper ablation must
quantify whether the additional context actually improves decisions.

This candidate requires 13 extra RZF evaluations per decision and increases
network parameter count. Do not reuse the original 0.71-ms timing claim.
Do not claim novelty merely from adding rate features or call an estimate a
calibrated reliability probability. If retained, compare old/rate-context
states at the same prices, episodes, seeds, validation budget and test traces.

## Why fixed-price training comes first

The current adaptive-dual pilot may sit at an unnecessarily conservative point.
The completed fixed-price sweep isolates that issue without changing state,
network, physical model, action clock or power constants. Its outcome determines
whether extra state is warranted; no large new training batch should be launched
solely because a feature counterexample exists.
