# Bounded audit of spatial actions omitted by count-only control

Detected axes: communications; baseline/ablation/reproducibility; restricted
action comparison and overclaim risk; planning.

## Motivation and prior-work boundary

The completed price calibration does not establish a learned energy advantage.
Code inspection shows that both the count network and its residual variant
always execute `select_top(score, count)`. Keeping a previous set at the same
count, or exchanging one selected location, is not an independently available
action. A count change is therefore not a complete means of controlling spatial
switching and the subsequent active-only observations.

This is a limitation of our action parameterization, not a new physical law and
not evidence that expanding it will help. Energy-aware XL-MIMO antenna selection
already exists: [Marinello et al.](https://arxiv.org/abs/2009.02542). Historical
partial-CSI prediction and risk-aware antenna selection are also established by
[He et al., TWC 2024](https://ieeexplore.ieee.org/document/10478288/), DOI
10.1109/TWC.2024.3377733. The publisher abstract describes a channel tracker and
risk-aware MCTS. We have not reproduced that algorithm and must not label our
local search as RA-MCTS or claim that partial CSI plus prediction is novel.

## Claim–evidence matrix

| Question | Existing evidence | This audit | Interpretation boundary |
|---|---|---|---|
| Can count-only actions suppress an undesirable location change? | Fixed ranking determines every count mask | Construct a same-count retention action and check minimum command Hamming distance | Representational property, not energy superiority |
| Can a different same-count set improve predicted multiuser service? | RZF depends on channel conditioning, not gain alone | Check all single exchanges using the same causal forecast and readiness masks | Constructed predictor, not oracle CSI or a calibrated probability |
| Does either action improve realized energy and service? | No measurements for these actions yet | Paired single-action interventions and complete causal heuristic rollouts | Viewed development data, four trajectories per scenario; not learned-policy evidence |
| Does RL exploit these choices better than simple rules? | Unproven | Not tested in this first audit | Requires a separately fixed learner and matched-action non-learning controls |

## Fixed scope before running

- Preserve physical source, true requirements, initial full observation, recovery
  delay, energy accounting, four-slot decision clock and eight-slot forecast.
- Use the same anchor margins 1/1/1.04 as the residual study. This isolates action
  availability, not a new threshold sweep. Future policy comparisons must retain
  stronger calibrated search and mixture controls.
- Use the first four already-viewed development trajectories in each of the
  three existing scenarios (stream 973000000), not new purported test data.
- At slots 0, 32, 64 and 96, fork the causal PRS state. Execute the alternative
  command for one control block, then return to the identical PRS rule. Compare
  the immediate block and complete seven-second episode with the no-change fork.
- Retain all 48 starts per variant, including no-op choices. Do not pool these
  correlated starts as 48 independent trajectories or select favorable starts.
- Also run each deterministic variant at every decision epoch on the same 12
  complete trajectories. This tests whether repeated use erases isolated gains.
- Exactly two variants: `retain` and `exchange`. No threshold fitting, new neural
  network training or changed channel law in this audit. At most two single-thread
  workers in the existing local project venv; no installs or server changes.

## Causal alternatives

**Retain:** match the PRS-commanded count while changing as few previous command
bits as possible; use the existing score to resolve which members to add/remove.
This can retain a waking branch; hardware readiness remains enforced physically.

**Exchange:** include the original PRS mask and every one-for-one exchange.
Evaluate all using the same stale complex CSI, forecast amplitudes and future
readiness as PRS. Select lexicographically by predicted target violation,
normalized rate shortfall, modeled immediate transition energy, and preference
for the unmodified anchor on ties. No future simulated channel is used to choose.
This is a constructed deterministic diagnostic/control, not a new claimed method.

## Decision after this audit

Check whether any causal alternative offers credible headroom without changing
count or service definition, and whether a simple repeated rule already captures
it. Local hindsight improvements alone never justify a learning-gain claim.
Only if the mechanism is supported should an expanded-action learner be specified;
that study must compare with these exact non-learning actions, include failures,
and later undergo independent confirmation and final-model R2/R3 validation.
The main manuscript is not replaced by this diagnostic.

## Disclosed coverage amendment after the sparse-time pilot

The completed fixed-time pilot has 48/48 retention no-ops and 47/48 exchange
no-ops. Those particular snapshots do not exercise most possible changes, so
they cannot establish that spatial control is ineffective. Before collecting
further outcomes, add one census of every decision epoch of the same 12 PRS
trajectories: 420 states, both alternatives at each state, all 840 rows retained.
Only exact no-ops can reuse the already verified baseline outcome. For every
causally determined changed action, evaluate its full counterfactual continuation
with the unchanged PRS rule. No state is selected based on future energy or outage.
Report both the full-census and changed-event counts, and preserve the first
pilot. The amendment is exploratory and does not turn these traces into a test set.
