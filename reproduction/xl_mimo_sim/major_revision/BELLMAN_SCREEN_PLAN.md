# Frozen closed-loop screen of the Bellman reconstruction

2026-09-21. Written before validation trajectories or policy outcomes are opened.
Prerequisite: both fixed fits finish under `BELLMAN_FIT_PLAN.md`. No additional
training, iteration-depth selection or price outside the frozen grid is allowed.

## Shared validation and controls

New stream 1110000000 + scenario*10000 + i, i=0..7. Three original scenarios,
eight external trajectories each, 24 in total. Original 140-slot scoring and
all physical, observation and prefix-action constraints remain unchanged.

Evaluate BOTH fixed models, all six prices and both depth16 and depth1: 576
episodes. Depth1 is the immediate-risk-only ablation using EXACTLY the frozen
risk weights of depth16, not a separately fitted or deliberately weak model.
Also run each frozen predecessor mix_pi with its own original lambda2/settings:
48 episodes. Retain all 624 records; do not only save selected prices.

Strong search: prs, cost, current-prs, current-cost, each with minimum1/4 and
margin0.90/1/1.04/1.08/1.16/1.24, plus full-array operation: 49*24=1176 episodes.
These are the SAME four search families used in preceding screens, including
exact current-active-CSI variants. No search or physics code is weakened.
Search has 48 tuning configurations versus six prices per learned variant;
this is conservative screening, not an equal-budget universal ranking.

Total 1800 complete evaluations on 24 shared trajectories. Neither 1800 rows
nor two training seeds imply 1800 independent environmental samples.

## Selection and statistics fixed in advance

For EACH seed/scenario/variant select lowest mean energy among prices with mean
outage<=5%; ties use outage then price. If none is feasible, select lowest outage
then energy then price for reporting and mark infeasible. The rule is applied
without looking at frontier gains. Scenario-wise tuning is disclosed; deployment
does not choose a price using the future of an individual test trajectory.

Similarly choose best feasible pure search among 48 points (all-on is a sanity
reference, not a search win). Include full-array in the finite49-point risk–energy
mixture frontier. Report each selected learner against this frontier at its own
mean outage. Episode-mixture weights are fitted EXPECTATIONS on validation,
not independently executed stochastic deployments or an oracle over all policies.

Myopic controls include all six prices plus full-array in a separate finite
frontier, so different amounts of outage cannot masquerade as long-term value.
Retain same-price results too, but do not select a weak myopic price for comparison.
Report paired mean energy difference, trajectory-level sample SD and number of
lower-energy trajectories. No significance claim or bootstrap of correlated
root/action records. Independent confirmation, if allowed by the gate, needs a
new protocol and untouched trajectories with all policies/mixtures frozen.

## Gate and stopping

Primary energy claim: nominal scenario, not a post-hoc favourable scenario.
For BOTH seeds the selected depth16 policy must:

1. meet <=5% mean outage in all three reported scenarios (pilot feasibility only);
2. use <=98% of the nominal best feasible pure search energy;
3. have nominal outage <= pure search outage +0.005;
4. use <=98% of the nominal matched-risk search-mixture energy;
5. use strictly less nominal energy than the matched-risk myopic frontier.

Every comparison must exist; missing feasible controls never mean automatic
promotion. Low/high gains are reported boundaries, not alternate success gates.
The 2% gate is a research-screening choice, NOT a conference acceptance rule or
a definition of statistical significance. Previous failed gates are unchanged.

Failure stops this candidate without more prices, depths, seeds or retraining.
Passing permits independent confirmation only, not replacement of manuscript
figures or declaring R1 complete. New R2 sensitivity and R3 standardized-channel
checks must use this model if eventually adopted. All adverse records are retained.

Use the existing project venv with at most two single-thread workers. No global
installation, host/server environment change, paper submission or external write.
