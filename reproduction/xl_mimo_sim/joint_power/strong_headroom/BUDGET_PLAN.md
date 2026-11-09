# Post-diagnostic analysis at the paper's failure target

The prespecified non-increasing-failure headroom probe failed its nominal 2%
development screen. Preserve that result. This additional analysis uses only
the already generated labels and is explicitly post hoc, not a new test.

Its question differs: does using a remaining failure allowance create energy
opportunity? For each saved root let the whole-episode allowance be 0.05*K*T=21
failed user-slots. Select only actions whose saved full suffix failures plus
the already observed failures do not exceed 21. Compare four-slot and full-tail
energy selection using that same hindsight feasibility set. If it is empty,
retain the reference and record infeasibility, including already spent budgets.
Do not discard these roots, replace traces, or count fallback as a safety proof.

Report single-intervention episode energy and failure rates, reference values,
empty sets, energy components, and paired whole-trace bootstrap intervals. Both
selectors know future outcomes. This is not deployable learning, not repeated
interventions, not mean-risk optimal control, and not performance at identical
actual outage. The risk condition is per episode when a feasible action exists;
the paper's mean constraint is different. Do not combine overlapping gains.

Only a nominal average saving of at least 2%, with resulting average outage at
most 5% in all scenarios, would justify considering a new budget-conditioned
causal learning probe. A long-term energy head additionally requires at least
1% full-episode gap over four-slot energy selection with the same full-tail risk
information. This does not override the original conservative-probe failure.
No new simulations, training, host modifications, or original-manuscript edits.
