# Terminal failed analysis attempt — no results generated

The initial post-hoc analysis exited with code 1 before reading episode data:
the reused CSV reader tried to parse the reference name `Expanded PRS` as a float.
Only the pre-analysis manifest was written. It records the earlier, now replaced
analysis script hash and must not be interpreted as a completed, verified run.

Correction: use an explicit parser for scenario, trajectory, reference name,
energy, and outage; add a regression test. Analysis semantics, saved labels,
thresholds, streams, and all completed simulations are unchanged. Preserve this
failed-attempt record and manifest; corrected results go to sibling
`target_budget_v2/`. No simulation is restarted or old result overwritten.
