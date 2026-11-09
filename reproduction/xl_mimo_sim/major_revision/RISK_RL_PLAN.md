# Matched RL after the rate-risk correction screen

2026-09-22, frozen before RL initialization or generation of streams 120/121.
Prerequisite: verified estimator_screen_pass in RISK_CORRECTION_PLAN. That
screen passed prediction checks, but nominal calibrated search spent MORE
energy for lower outage; it is not an RL or net-energy superiority result.

## Matched intervention

Train two constrained, masked Double-DQN variants, each seeds 11 and 22:
- old: original filtered-current/point forecast and its hard failure indicator;
- calibrated: measured-current correction plus frozen calibrated rate risks.

Same physical model, 16 ranking prefixes, RA spatial ranking, root causal Gram
summary, architecture, training trajectories, seeds, reward, updates, and
checkpoint/selection budget. No new probing, queue, power control, or sleep
mode. No future truth enters either deployed controller. Search controls
receive the identical information, masks and selected operating parameters.

The 362-element common state is the existing causal Gram/tracker/hardware/
demand representation. Each of 16 candidates contributes its 16-bit mask,
8 available fractions, 8 normalized slot energies, 24 forecast rate ratios,
24 failure scores, and one allowed-action bit: 81 features, total 1658.
Old failure scores are deterministic 0/1 indicators, not probabilities.

Common encoder 362->64 (LayerNorm, SiLU); candidate encoder 81->32 (SiLU);
shared head 96->64->1 (SiLU). Initialize final residual head to zero, add prior
negative eight-slot mean normalized physical energy. Thus checkpoint zero
is exactly the corresponding cost-search action, apart from exact-energy
ties resolved by lower action index. Eligible actions satisfy the same
prediction threshold and minimum count as that search. If none passes,
both use that search's unique fallback action. Masking is an empirical
prediction filter, not guaranteed physical safety or a new contribution.

Operating parameters are frozen from stream-119 search tuning, per scenario,
and supplied to both search and RL; disclose scenario-specific tuning.
The rollout scenario identifier is NOT a network input. Both variants have
the same ability to select every prefix satisfying their respective filter.

## Fixed training and selection

- 120 episodes/variant/seed, four fits, 480 total. Train stream1200000000 +
  seed*10000 + episode; scenario=episode%3. Same external traces per paired
  variant. Episodes have 140 controlled slots plus an actual successor slot
  for continuing-task bootstrapping, no fictitious terminal reset label.
- One replay update per four-slot transition; batch64, replay20000,
  AdamW lr4e-4/decay1e-5, gamma.98, target sync100updates, gradient clip5.
- Reward = negative normalized actual block energy - 2*actual block outage.
  Fixed price2, no price sweep; all costs include wake/sleep/pilot/processing.
- Epsilon=.04+.46*exp(-4*episode/119), sampled only among allowed actions.
- Checkpoints 0,30,60,90,120. Use the same stream119 selection trajectories,
  four/scenario. Minimize worst-scenario excess over5% mean outage, then mean
  energy; exact ties retain earlier checkpoint. Restore exploration RNG after
  selection evaluation. Keep ALL checkpoints and selection results.

## Independent audit and stopping

Only after all four fits complete, generate stream1210000000+scenario*10000+i,
i=0..7:24 new external trajectories. Evaluate all four selected models and
all three selected searches (old/current-corrected/calibrated),168 episodes.
Report each seed, both prediction variants, absolute energy/outage, paired
deltas against its corresponding search, and corrected-versus-old RL.
Do not average seeds as additional independent trajectories. Bootstrap over
external trajectories, seed1180900001,10000resamples, descriptive95% intervals.

RL benefit requires lower energy at matched service, not comparison solely
with a search point failing the service target. The strict exploratory gate
requires BOTH calibrated RL seeds to have nominal outage<=5%, strictly lower
energy than calibrated search and no higher mean outage. Report high/low
failures regardless. Gate does not establish novelty, statistical significance,
standard-channel transfer, or whole-major-revision completion. If checkpoint0
is selected, label it search-equivalent, not learned improvement.
No extra epochs, changed reward, further seed selection or parameter resweeps
after seeing audit outcomes. No new standard-channel run unless a candidate
has actually shown a defensible control advantage.

Verify exact zero-initialization search parity, hidden-future independence,
valid action masks and target masks, matched training traces, all selected
policy replays, source/model/output hashes, and episode energy decomposition.
At most two single-thread workers; project venv only. All old sources/data/PDFs
stay untouched. These networks are new research candidates, not silently
substituted for the original manuscript's 72-input policy.
