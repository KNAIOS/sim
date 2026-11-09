# Joint subarray and power control: bounded development protocol

Authorization: the author explicitly approved joint subarray/ transmit-power
control on 2026-09-22. This supersedes the missing-scope decision, not the
requirements for accurate data and fair comparisons. Old experiments and the
four-page manuscript remain frozen. No host installation or remote operation.

Route: communications / baseline, ablation, statistical, reproducibility /
unfair-comparison, overclaimed-results / planning.

## Model and algorithm being tested

The fixed BS, moving UEs, geometric complex channel, bandwidth, noise, circuit
power, pilot charge, switching charge, and four-slot RF recovery remain those
of the original validation configuration. Decisions are every four 50-ms slots.
The total radiated-power ceiling is still 1.2 W, and PA efficiency is still .4.

Unlike the earlier postprocessing audit, variable power enters candidate
evaluation, realized transitions/rewards, training and validation. An outer
controller selects a ranked subarray prefix; the inner controller minimizes
instantaneous transmit power using current active-branch CSI. This is a
hierarchical joint policy, NOT a claim that an RL network directly outputs
continuous powers or jointly optimizes beam directions.

For fixed unit-norm RZF directions (regularization fixed at the 1.2-W ceiling),
let G_kj = |h_k w_j|^2 and gamma_k = 2^(d_k/((1-pilot)B))-1. Then:

    p >= F p + u; F_kj = gamma_k G_kj/G_kk (j != k),
    u_k = gamma_k noise/G_kk.

If rho(F)<1 and sum((I-F)^(-1)u)<=1.2 W, this is the componentwise minimum
power satisfying all user rate targets for these beams. Numerical guards are
checked against the original SINR equation. If all targets are infeasible,
use the same fallback for every method: retain at least the original equal-
power achievable rate up to the requested target. Every failed user-slot is
still counted. This fallback is NOT globally optimal admission or beamforming.
The power cap is total, not an invented per-user .4-W cap.

The arbitrary four-module floor is removed for ALL new joint methods:
counts 1..16 plus retaining the current command. A single module has eight
antennas for three users, but this does not guarantee rate feasibility.
No queues, traffic deferral, new PHY channel, or artificial search noise.

Predictor: last-observed complex CSI, causal gain forecasts and deterministic
hardware countdowns for t..t+7. Current active CSI is observed before commands;
sleeping branches are not scanned. The phase-one configuration requires
positive wake delay, so newly commanded branches cannot obtain instantaneous CSI.
Forecasted energy/rates are estimates, not known future ground truth.

Learning: shared action-scoring Double DQN with an analytic forecast-cost prior
and a learned residual. Inputs retain module absolute gains, complex user
correlations, observation age, hardware state, demand, motion and candidate
joint energy/outage features. Fixed price 8, gamma .98, AdamW .0004, hidden
64/64, Huber loss, batch32, replay5000, target copy100. Last layer starts at
zero so the initial policy is the declared forecast-cost prior. Same prior
without training is an explicit ablation. No claims of novelty from these
existing algorithmic ingredients alone.

The 229 inputs per candidate comprise 216 common context features and 13
candidate features. The scalar cost prior uses the predicted first four-slot
cost and repeats the next predicted four-slot cost over the remaining finite
episode with discount .98; this tail approximation is explicit, not an exact
value function. Exploration is .04+.46*exp(-4*episode/119), and updates start
after64 collected transitions. No host-level thread settings are changed;
process-local BLAS/PyTorch thread counts are one, with two training workers.

## Frozen phase-one comparisons

- Full activation, fixed power; full activation, shared power allocator.
- Original PRS, fixed power; original PRS with the shared allocator (same count
  domain 4..16): isolates common power-control benefit.
- Joint full search over counts 1..16 plus hold, minimizing predicted total
  energy under predicted average rate-outage <=5%; otherwise minimum rate
  shortfall then energy. Fixed ranking and finite horizon are limitations.
- Joint coarse search, counts {1,4,7,10,13,16} plus hold. This is a declared
  candidate-grid ablation, NOT an equal-runtime claim without measurements.
- Untrained forecast-cost prior and trained residual RL use the full candidate
  set and the same physical allocator as joint search.

Train seeds 11 and 22, 120 episodes each, nominal/low/high cases sampled evenly
in expectation; validate every30 episodes on 4 trajectories per scenario.
Choose checkpoints by worst-scenario 5% violation, then energy, with the
untrained checkpoint included. Retain last and selected checkpoints and all
failed runs. Training stream 2209220000 + seed*100000 + episode; validation
2309220000 + scenario*10000 + trajectory; development 2409220000 similarly.
Development: 4 trajectories per scenario, reused by every method/seed. These
are bounded development results, not submission-level confirmation. No tuning
on development outcomes, seed removal, data fabrication or altered search CSI.

Proceed to larger confirmation only if ALL three selected-policy scenario mean
outages are <=5%, nominal energy beats joint full search by >=2%, and neither
training seed has a nominal energy regression. Otherwise report why and do not
replace the manuscript or silently relax this gate. An unchanged selected
initial policy is not evidence of learning. This screen is not a significance test.

## Claim-evidence matrix

| Claim | Required evidence | Phase-one evidence | Remaining risk |
|---|---|---|---|
| Joint power calculation is correct | Interference-aware analytic and independent PHY tests | Unit/replay tests | Constant PA efficiency and ideal active CSI |
| Joint optimization saves energy | Same physical capabilities and service accounting | All named baselines | Development only |
| Learning adds value beyond search/prior | Same candidates, power solver, seeds and traces | Two-seed bounded screen | Independent multi-seed confirmation still required |
| R1 resolved | Reliable incremental benefit over strong baselines | Not presumed | Critical until confirmed |
| R2/R3 apply to new method | New sensitivity and standardized-channel runs | Deferred until candidate passes | Old results cannot be relabeled |

Primary background: [joint XL-MIMO selection/power allocation](https://arxiv.org/abs/2102.11438),
[hardware activation plus transmit-power minimization](https://arxiv.org/abs/2002.01504),
[known/unknown-dynamics RL decomposition](https://www.vanderschaar-lab.com/papers/36.pdf).
These support modeling choices, not this candidate's novelty or expected win.
