# Frozen-policy single-action intervention audit

Axes: communications; baseline/ablation/reproducibility; audit; risk of attributing
an observed energy gap to an unverified learning mechanism.

## Claim-evidence gate

| Question | Required evidence | Current evidence | This bounded check |
|---|---|---|---|
| Does RL's extra activation purchase useful future reliability? | Same initial information, hardware and exogenous trajectory; change one action only | High-load operating energy explains 97.6% of the latest gap, but aggregate counts do not establish causation | Compare frozen RL with one PRS-command intervention followed by the same frozen RL |
| Are network action preferences aligned with realized consequences? | Record predicted action-value gap and the consequences of both choices | Q-values have not been audited | Record both values, energy and outage differences, including the immediate four-slot block |
| Would replacing decisions yield a better deployable controller? | An independently evaluated causal policy with tuning separated from tests | Not established | NOT answered by this audit; interventions at different times cannot be summed |

## Prespecified scope

- Latest continuing-control candidate, both selected seeds 11/22, frozen.
- First four already-viewed development trajectories in each of the three
  scenarios; four fixed decision slots 0, 32, 64 and 96. Total 96 paired starts.
- At each start, copy the full simulator, hardware and tracker state. Use the
  validation-selected PRS margin for that scenario, with no access to future CSI.
- Original branch: execute the RL action and then follow the same RL until the
  existing 140-slot window ends. Alternative: execute PRS's command once, then
  follow that same RL. Each branch acquires its own subsequent active-only CSI.
- Use the identical exogenous channel and demand sequence. This is valid only
  within this simulator's action-independent UE/channel model; it is not a field
  experiment or a certified action-value estimate under observation uncertainty.
- The branch engine must reproduce the existing engine's observations, actions
  and total energy/outage for unmodified policies, including countdown delays.
- Record every start including identical actions and unfavorable interventions.
  Do not select a favorable intervention as an online oracle policy.
- Q-value gaps are learned preferences, not confidence bounds. Finite realized
  returns are not equal to expected infinite-horizon Q-values; no absolute
  Q-error claim follows from this comparison.

No training, checkpoint selection, power modification or new dependency is
needed. One single-thread local process suffices. Raw outcomes and source hashes
are saved separately. If interventions reveal systematic avoidable activation,
they motivate a separate training/control hypothesis, not an energy-win claim.

Double DQN is already used. [The original paper](https://arxiv.org/abs/1509.06461)
addresses action-value overestimation in tested settings; it does not guarantee
correct action ordering or explain this communications result by itself.

## Follow-up after the first audit (specified before its execution)

The RL intervention audit is complete: 47 of 96 prescribed starts give lower
realized energy without additional outage, but this only identifies weaknesses
relative to PRS. To check whether PRS itself has a local opportunity for learned
correction, use its own state trajectories, the same first four traces/scenario
and the same four decision times (48 starts, no artificial training-seed repeats).
Reduce the commanded prefix by exactly one, floored at the common minimum four,
once; thereafter follow the same frozen PRS. Retain all failures and minimum-count
cases. A successful hindsight intervention is not proof of a learnable causal
opportunity. The search replay must exactly reproduce the existing PRS records.

The decrement audit then completed: no opportunities at low load (all prescribed
starts use the minimum four), three harmful decrements at nominal load, and four
realized dominating decrements at high load. High-load mean immediate energy
decreases by 0.231 J, but full-window energy increases by 0.119 J. Thus downstream
control/observation changes can reverse immediate savings. Before any residual
policy training, run the symmetric +1 count probe on exactly the same 48 starts,
clipped at sixteen. This tests whether paying for one extra subarray now can
improve later energy or reliability. It is the final local-action diagnostic,
not a search over many deltas. The decrement's original source bytes are archived
under their SHA-256 filename before adding the optional positive probe.
