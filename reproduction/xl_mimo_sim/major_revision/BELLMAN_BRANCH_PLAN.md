# Counterfactual decision-block Bellman reconstruction

2026-09-21. Author research protocol, not a submission claim. The user explicitly
authorized reconstruction with bounded validation and no guarantee of beating
search. All earlier gates and failed endpoints remain unchanged.

Detected axes: communications; baseline/ablation/statistical/reproducibility;
unfair-comparison/overclaimed-results; planning. Reviewer R1 remains open.

## 1. One hypothesis, not a new generic RL claim

The preceding experiment evaluated a first command and a fixed continuation for
16 physical slots. It found a small nominal signal, but neither the declared
promotion gate nor reliable high-load control. This new candidate instead learns
from actual four-slot transitions and a recursively updated successor value.
Changing the target can propagate consequences beyond a fixed teacher/student
rollout. It can also amplify approximation and off-policy errors; improvement is
not assumed. Root-state coverage alone does not prove successor-state coverage.

We retain the same 16 ranking-prefix actions, active-only causal CSI, hardware,
RZF evaluation, traffic, wake delay, switching/pilot energy and original three
speed/load scenarios. No power coefficients or demands are adjusted to win.
Known four-slot hardware energy is computed analytically. Unknown communication
loss and continuation value are learned separately. A no-continuation control
must use the SAME learned immediate-risk model, not a weaker unrelated DQN.

Fitted Q iteration and known/unknown dynamics decomposition are established:

- Ernst, Geurts and Wehenkel, JMLR 2005, pp. 503--556,
  https://jmlr.org/papers/volume6/ernst05a/ernst05a.pdf (Section 1 explains
  successive optimization horizons; Section 3 gives fitted Q iteration).
- Mastronarde and van der Schaar, TSP 2011,
  https://www.vanderschaar-lab.com/papers/36.pdf, studies known/unknown dynamics
  decomposition for energy-efficient wireless control.

These motivate implementation; their convergence/safety results are NOT claimed
for our neural network, finite sample, partially observed channel or simulator.
No final method name, novelty assertion or manuscript replacement is warranted
until comparisons support an application-specific contribution.

## 2. Stage A frozen data contract

This stage creates transition evidence, NOT new energy-performance curves.

- Existing project venv only; at most two single-thread workers. No host/server
  installation, global environment changes or other project writes.
- TRAIN stream 1100000000 + scenario*10000 + trajectory, trajectory=0..3.
  Twelve exogenous trajectories, shared across seeds 11 and 22.
- Two causal roll-ins per seed: existing frozen cost search mu, and that seed's
  frozen `policy_consistent/mix_pi` model. The failed model is only a coverage
  policy, not promoted or retrained. Its old lambda=2 and settings are unchanged.
- ALL 35 decision roots t=0,4,...,136 per trajectory. Thus 840 roots/seed,
  13440 action transitions/seed, 53760 branch slots/seed. Both seeds total
  1680 roots, 26880 transitions, 107520 branch slots. Shared mu roots are duplicate
  observations across seeds, NOT extra independent trials.
- At each root clone the policy-dependent state, execute each of the 16 legal
  prefix commands for four physical slots, collect energy and user-slot outage,
  then prepare the causal successor observation. No teacher/student continuation
  occurs inside the label. Store root and successor features and action index.
- Feature dimension stays 1500, but the two old search-setting features are fixed
  to margin=1 and minimum=1 in every NEW input. No scenario identity, simulator
  hidden state, future channel or time-to-evaluation-end is added.
- Generate 141 physical samples to obtain a REAL successor at t=140. The scored
  roll-in still runs 140 slots. A finite record cutoff is NOT a terminal state:
  all bootstrap flags remain true. No invented zero tail at the episode end.
- E4 is energy/(4*full-slot energy); O4 is the mean of 12 user-slot failure
  indicators. The first-slot outage is exactly known from current observed
  active CSI; the unknown part is the remaining three physical slots. These
  are simulator outcomes, not calibrated real-system probabilities.

Mandatory checks before fitting:

1. Original root state is unchanged by every counterfactual branch.
2. Analytic four-slot energy matches each actual branch within float32 storage
   tolerance 5e-7 in normalized units; current outage within 5e-7.
3. The branch for the roll-in's chosen action exactly reproduces its four-slot
   costs and successor feature vector (cost tolerance 1e-10).
4. Every action is represented once at every root. All observations finite.
5. t=136 has a real t=140 successor and is not labeled terminal.
6. Unit tests alter only unobserved future samples: root inputs stay unchanged;
   samples strictly later than t+4 cannot change the transition/next input.
7. Mu data are identical across seeds. Source, checkpoint and output hashes,
   seeds, physical config and counts are recorded. No cross-seed pseudo-replication.

Stop on semantic, accounting, causality or reproducibility failure. Fix only the
new implementation, preserve failed outputs and document the correction. Passing
Stage A authorizes only the separately specified training stage, not a paper claim.

## 3. Stage B design and gates to freeze before fitting

Intended architecture: a shared action-conditioned immediate-risk model with
known first-slot outage and exact four-slot energy, plus a price-conditioned
continuation network. Fit immediate risk once, then freeze it for both long-term
and myopic policies. Fit continuation via actual successor inputs and Bellman
targets; no fixed mu/pi suffix labels and no forecast output treated as truth.

Before training, a separate immutable protocol MUST fix exact network dimensions,
losses, iteration depth, epochs, optimizer, price grid, selection rules, random
streams and budget. No training is started under an unspecified protocol.
Two training seeds, one bounded fit per seed; no training until a favourable
curve appears. The proposed comparison is to strong four-family search, not
only to the failed predecessor or all-on operation.

Nominal is the primary development condition; low/high are reported boundaries.
Validation must report energy and outage together; compare both feasible pure
search and a disclosed finite search-mixture frontier at matched mean outage.
That frontier fitted on validation is not an independent deployment estimate.
It cannot be used to claim significance without new, frozen-policy confirmation.
Keep the old 2% gate marked failed. A new protocol must not retroactively change
it. Include the same-risk-model myopic ablation and frozen predecessor.

Even a successful pilot still needs independent confirmation, sensitivity tests
and standardized-channel evaluation for THIS method. Old R2/R3 results cannot
be relabeled as evidence for a new method.

## 4. Claim–evidence matrix

| Claim | Required evidence | This stage | Remaining risk |
|---|---|---|---|
| Action transitions are correct and causal | Physical replay, all-action energy identity, suffix isolation | New Stage A data and unit tests | Does not establish generalization |
| Long-term learning improves reliable energy use | Same immediate-risk ablation and strong search, paired closed loop | Not yet performed | Critical R1 gap |
| New controller satisfies 5% mean outage | Fresh evaluation in each stated condition | Not yet performed | No pointwise safety guarantee |
| Revision is complete | New method R1/R2/R3 and synchronized paper/reply | Not established | Do not mark ready to submit |
