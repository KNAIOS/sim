# Bounded predictive-residual control pilot

## Mechanism and limits

The completed single-action audits identify realized cases in which one extra
module now saves energy later, as well as cases in which a decrement incurs
later compensation. This is consistent with coupled hardware and observation
trajectories, but does not identify a guaranteed or already learnable benefit.
The next hypothesis is to let RL learn a local correction to a credible predictor
instead of relearning a global count map that currently keeps too many modules on.

Residual RL is established prior art, e.g.
[Johannink et al., Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201).
Its use here is a method-development candidate, not a new generic RL invention
or a demonstrated communications contribution.

## Definition

The causal PRS computes count m0 using the same active-only tracker, readiness
ranking and validation-selected margin as the retained baseline. A Double DQN
chooses delta in {-1,0,+1}; execute the ranked prefix of m0+delta. Invalid boundary
actions are masked in exploration, greedy selection and the Bellman target.
The zero residual exactly reproduces PRS, including pilots and switching energy.

The observation contains the existing 217 causal features plus m0/Q (218 total).
The same 64-unit hidden layers, optimizer, replay settings and energy/outage
reward remain. Initial output weights are zero and biases [-0.01,0,-0.01], so
the initial greedy policy is exactly the anchor, not a randomly inferior policy.
This prior and the reduced action space must be disclosed.

Training is the explicitly continuing-control interpretation from TIME_LIMIT_PLAN:
140 executed slots plus one causal successor observation for final bootstrapping.
Adoption would require a consistent manuscript objective and method rewrite.
No geometry, power, minimum count, demand, recovery or outage threshold changes.

## Prespecified execution and comparison

- Two seeds 11/22, each 300 episodes; price 4; no new price or network-width sweep.
- Same 971M training streams, 972M validation streams (12 traces/scenario), and
  973M already-viewed development streams (16 traces/scenario) as the previous
  diagnostic. Keep all unsuccessful checkpoints and both seeds.
- Evaluate checkpoint 0 (exact PRS) and every 50 episodes. Select minimum
  worst-scenario mean-outage violation, then minimum mean energy. If checkpoint
  0 wins, report fallback, not learning improvement. Selection does not use
  development outcomes.
- Retain the tuned PRS and switch-aware PRS. Also evaluate fixed residual -1
  and +1; they are necessary simple alternatives, not optional weak controls.
- Use at most two simultaneous single-thread workers, existing local venv only.
- Development improvement is only a gate. Before replacing the main paper,
  require independent held-out confirmation, more training seeds, matched-input
  absolute-action ablation, final-model sensitivity/timing, and standardized
  transfer. Do not reuse the original 72-dimensional model's timing.

This pilot is not evidence until it runs. It will not be promoted merely because
it reduces outage at higher energy or because its unchanged anchor performs well.
