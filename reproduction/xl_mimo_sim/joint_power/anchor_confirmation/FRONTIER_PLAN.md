# Post-confirmation equal-risk baseline-envelope diagnostic

2026-09-22, after observing frozen confirmation. This is explicitly posthoc
analysis of saved outcomes, NOT a new independent experiment or tuned deployed
baseline. No policy/weight/threshold changes, no extra trace selection.

For each of the three scenarios, retain ALL seven non-learning controls and
construct the minimum empirical expected energy attainable by choosing one
controller for an entire episode at random, independently of the trajectory.
Permit any two-controller mixture with empirical mean outage <= the evaluated
learning mean. In a finite set with one risk constraint an extreme optimum
uses at most two pure controllers; enumerate all pure/pair candidates.

Report pooled two-seed learner and each seed separately. Compute trajectory-paired
energy differences against each selected mixture expectation. Bootstrap eight
shared exogenous traces and two training seeds and REFIT the finite envelope
within each resample. This is descriptive plug-in uncertainty, not a formal
coverage certificate, and selected mixture probabilities are not frozen on
independent validation. The feasible frontier of these finite baselines is NOT
the global optimum over arbitrary policies. No claim of an actual randomized
deployment is made; numbers are expectations from saved pure-policy episodes.

Also report the same finite envelope at the declared5%target. If a feasible
mixture beats the primary at5%, retain it: cannot proclaim optimality within
the constraint from a win at the learner's lower actual outage. The diagnostic
addresses risk comparability but does not replace frozen independent comparison,
new-method sensitivity, standardized-channel checks or uncertainty limitations.
