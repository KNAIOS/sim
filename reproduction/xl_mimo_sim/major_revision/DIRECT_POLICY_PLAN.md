# Direct episodic policy optimization: bounded reconstruction protocol

2026-09-21. Frozen before training. The failed Bellman candidate stays stopped.
Axes: communications; baseline/statistical/reproducibility; overclaimed-results;
planning. The purpose is reviewer R1, not a new generic RL algorithm.

## Claim and evidence contract

| Proposed claim | Required evidence | Present evidence | This stage | Risk |
|---|---|---|---|---|
| State-dependent control margins save energy at comparable reliability | Full closed-loop trajectories against strong calibrated search | No such advantage established | Two fixed training seeds and all three scenarios | Critical |
| State dependence, not extra scalar tuning, contributes | Same optimizer/budget with bias-only policy | None | Train bias-only alongside nine-weight policy | Major |
| Avoiding a fitted long-term critic fixes prior failure | Reliable new policy, not lower training loss | Prior diagnosis only motivates the hypothesis | No Q network or bootstrap targets | Major; not a causal proof of previous failure |

## Physical model and causal controller

Use unchanged validation configuration (16 subarrays, 3 users, 140 slots,
four-slot decisions, four-slot recovery, eight-slot causal predictions). All
operating, switching and pilot energy remains in measured episode totals.
Requirements used by the actual physical evaluation never change. Initial CSI
and subsequent active-only observations are the same for every method. No true
future channel, hidden motion, scenario label, trajectory seed or episode clock
enters the policy.

Reuse the existing cost-search hold predictions and ranking prefixes 1--16.
At a decision, determine its reference action at fixed margin 1.04. Form nine
fixed-scale causal features:

1. bias 1;
2. mean current requirement / 560 Mbps minus 1;
3. minimum reference predicted rate/requirement at slot 0, clipped [0,2], minus 1;
4. clipped minimum reference ratio at slot 7 minus that at slot 0;
5. mean min(CSI age,32)/32;
6. fraction of subarrays currently recovering;
7. fraction active minus 0.5;
8. mean absolute estimated support velocity / 0.55, clipped [0,1];
9. last observed user-slot outage fraction.

The controller outputs m=clip(m0*exp(clip(theta.phi,-2,2)),0.85,1.35).
Cost search then selects the lowest predicted-energy feasible prefix; if none
is feasible it uses the SAME minimum-shortfall fallback as the frozen search.
Thus theta=0 exactly contains cost search at m0. This is a nonlinear hybrid
policy with nine learned weights, not an end-to-end neural DQN, and not a
guaranteed-safe controller. Scaling a prediction threshold is not lowering
actual user demands. Bias-only retains only feature 1 with one learned weight.

## Established optimizer, explicit deviations

Read Secs.2--3 / Algorithm2 of Mania, Guy and Recht (2018),
https://arxiv.org/pdf/1803.07055 . Use an ARS-V1-t-style parameter perturbation
update, not a claim of exact benchmark replication or algorithmic novelty.
Four Gaussian directions, top two by better signed return, exploration std
0.08, step 0.04, and population standard deviation of those four selected
returns. Skip only if that standard deviation <=1e-12. No online feature
normalization, critic, replay buffer, imitation labels or optimizer restarts.
Adaptations: fixed physical features, bounded margin passed to discrete search,
and common exogenous trajectories for both signs and all directions.

Train seeds 11 and 22, each both state and bias-only variants. Exactly 12 updates
per fit, all start at zero, m0=1.04. At each update use two trajectories per
scenario, shared by all perturbations and variants/seeds. Training stream
1120000000 + scenario*10000 + 2*iteration + index, iteration 0--11, index 0--1.
This is 72 external trajectories total, reused across policies, NOT independent
per evaluation. Each fit consumes 12*8*6=576 full episodes (80640 slots), total
2304 episodes. Reward is -(episode energy/full-array energy + 4*episode outage),
averaged equally over six episodes. Price 4 is fixed, not selected later.
No constraint guarantee follows from this scalar training objective.

Save all 13 center weights and every perturbation return/episode metric. Only
the terminal update12 is eligible for validation; no best-checkpoint selection,
adaptive stopping, extra iterations or enlargement after seeing results.

## Validation, fair controls and stop rule

Fresh stream 1130000000 + scenario*10000 + index, eight trajectories per scenario.
All models see the same 24 trajectories. Tradeoff margins m0 in
(0.85,0.90,1.00,1.04,1.08,1.16,1.24,1.35), fixed before training.
State and bias models each get this complete grid, two seeds: 768 records.
Search controls: prs, cost, current-prs, current-cost, minimum count 1 or 4, same
eight margins, plus full array: 65 points/scenario, 1560 records. Dynamic method
gets training evaluations not given to fixed search; scalar learned ablation
therefore receives the exact same training budget. No equal-total-compute claim.

Select each method's per-scenario margin only by lowest mean energy subject to
mean outage <=5%; if none feasible select least outage then energy. This is
validation calibration, not independent confirmation. Retain all results.
Compute the finite pure/two-point episode-mixture search frontier at each
selected learned risk (a validation-fitted expected frontier, not a simulated
oracle or universal optimum). Also compute matched bias-only+full frontier.

Both state-policy seeds must satisfy all of the following to justify a separate
independent confirmation protocol:
- all three selected scenarios mean outage <=5%;
- nominal mean energy at least 2% below best feasible pure search;
- nominal outage no more than 0.5 percentage points above that pure search;
- nominal energy at least 2% below the matched-risk search frontier;
- nominal energy below the matched-risk bias-only frontier.

These are pilot promotion requirements, not statistical proof. Report raw
trajectory paired differences, SD and all failures. No automatic claim of
superiority, submission readiness or high-confidence constraint satisfaction.
If either seed fails, stop this candidate without more tuning. Old failed gates
remain unchanged. R2/R3 for the manuscript's original 72-input model do not
transfer to this new method. New figures/manuscript claims require independent
confirmation and new matching sensitivity/standard-channel evidence.

## Implementation and resource checks

Before training: neutral-policy/full search parity, real-demand immutability,
future/hidden-input exclusion, bounded finite margins, parameter update sign,
reproducible perturbations, and one full episode timing/accounting smoke test.
Archive source hashes and verify before/after every stage. Independently
recalculate update12 from saved directions/returns and recompute validation
selection/frontiers before interpreting performance. Use existing project venv,
no dependency installation or host/server changes; at most two single-thread
workers. Preserve all previous code, outputs and plans.
