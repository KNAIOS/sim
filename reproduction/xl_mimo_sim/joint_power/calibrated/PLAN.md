# Phase two: shared short-term calibration and action-consistent value learning

Frozen before calibration/training/evaluation, 2026-09-22.
Route: communications / baseline, ablation, statistical, reproducibility /
unfair-comparison, overclaimed-results / planning.

The preceding turn was progress: a new authorized joint-power implementation,
physical replay and failed development screen established excessive circuit and
switching costs. Scope permission is no longer missing. No host installation,
remote operations, old-result edits or manuscript replacement in this phase.

## Hypothesis and unchanged physical problem

The phase-one action-dependent tail assumes an action's second four-slot cost
persists until episode end, though decisions change every four slots. Together
with uncalibrated short-term risk, this may mis-rank hardware decisions. This is
a hypothesis to test, not a proven causal attribution of the preceding gap.
Reuse the complete frozen joint-power physics, channel, hardware, all 17
count/hold actions, causal observations and total 1.2-W power cap. Do not
change PA efficiency, demand, switching charge or search observations to win.

## Shared calibration

Training stream 2509220000, calibration-validation 2609220000. For each of the
three existing scenarios, collect 16 training and 4 calibration-validation
trajectories. At each behavior-policy state, branch all 17 commands for exactly
four physical slots. Features are computed BEFORE the branch; later channels
are training labels only. No branching or future access during deployment.
Behavior families are original PRS, joint full, joint coarse and an exploratory
count policy, equally represented within each scenario. Branches share a
read-only trajectory, not mutable hardware, tracker, observations or records.

Fit one 64/64 SiLU predictor to first-interval normalized energy and user-slot
outage fraction. Energy is a bounded residual (+/- .2) around the analytic
forecast, risk a logit residual around the forecast clipped to [.01,.99].
Standardize features on TRAINING observations only (std floor .05, clip10).
Optimize BCE+10*MSE with AdamW .001, decay .0001, batch256, 40 epochs, seed
2026092203. Select lowest calibration-validation objective at epochs10/20/30/40,
including untrained epoch0. Archive all checkpoint scores and raw held-out bins.
Gate for downstream RL: noninitial checkpoint, validation Brier below the raw
analytic forecast, absolute validation mean risk error <=.05, finite predictions.
This gate does not prove per-state or distribution-free calibration.

The selected frozen predictor is given to BOTH calibrated search and RL.
Calibrated full/coarse search minimizes predicted four-slot energy subject to
predicted risk <= threshold, else minimizes predicted risk then energy. Use
thresholds {0,.02,.05,.10}, chosen only on control-validation, not development.
Coarse grid is fixed {1,4,7,10,13,16}+hold. Retain legacy and uncalibrated full
search as diagnostics. No equal-runtime claim: shared candidate construction
still evaluates all counts.

## RL correction, if calibration gate passes

Use scalar cost energy+price*outage with price in {1,2,4,8}. One price-conditioned
Double DQN per seed learns all four prices using stored separate energy/outage
returns. Q is calibrated immediate cost + action-INDEPENDENT finite-horizon
state value + centered learned action advantage, removing the fixed
action-dependent infinite-hold surrogate. State-value and advantage networks
each have two 64-unit SiLU hidden layers; zero initial advantages, initial
state-value .6. Inputs are standardized causal features, calibrated immediate
predictions and price/8. The action-independent component has the remaining
discounted tail weight, but cannot itself alter initial action ranking.

Three-decision-step returns, discount .98, AdamW .0004, decay .00001, Huber,
gradient clip5, batch32, replay5000, target copy100, two updates per collected
transition after64 samples. Exploration .03+.27*exp(-4*episode/159); 75% of
exploratory actions are adjacent counts (+/-2) or hold, 25% global. This changes
training behavior only, never the deployed action domain.

Train seeds11/22 for160 episodes; scenario and price schedules balanced in
expectation, independent fresh training stream2709220000+seed*100000+episode.
Control-validation stream2809220000+scenario*10000+i, 4 trajectories/scenario.
Validate epoch0/40/80/120/160 across all four prices. Choose a single checkpoint
and price per seed by worst scenario mean outage violation above5%, then mean
energy, then worst outage. Search chooses its threshold by the SAME key and
same control-validation trajectories. Disclose RL's greater offline training.

Development stream2909220000, 4 trajectories/scenario, never used to choose
hyperparameters or checkpoints. Compare full-fixed, full-power, PRS-power,
uncalibrated joint-full/coarse, calibrated full/coarse, untrained calibrated
cost policy (price selected by same validation rule), and two trained RL seeds.
Save complete slot, episode, validation and checkpoint records.

Proceed to larger independent confirmation only if: each RL seed is below5%
mean outage in every scenario; nominal mean RL energy is >=2% lower than BOTH
calibrated full search and original PRS-power, with no per-seed nominal energy
regression; selected checkpoints are noninitial. Baselines exceeding5% cannot
be called reliability-matched; report the full trade-off and keep the claim
open in that case. This is a new candidate gate, not a rewrite of phase one's
failed gate. No scientific completion or manuscript replacement based solely
on this bounded screen.

## Claim-evidence matrix

| Claim | Required evidence | Available now | Missing / risk |
|---|---|---|---|
| Better risk estimates | Trajectory-held-out proper scores and bins | Raw forecast error audit | New calibration; policy shift remains |
| Learning reduces hardware energy | Shared prediction, power and actions; independent seeds | Phase one contradicts advantage | Corrected value training; critical |
| Advantage beyond model calibration | Calibrated search and untrained cost ablation | None presumed | All named baselines |
| Reviewer R1 resolved | Reliable confirmed learning gain | Not established | Independent confirmation |
| R2/R3 support new method | New sensitivity and standardized-channel runs | Old-method evidence only | Run after a viable candidate |

## Reproducibility and boundaries

Run under project-local venv; process-local BLAS/Torch one thread and at most
two training workers. Output folders refuse overwrite. Hash input code,
protocol, old manuscript and phase-one outputs. Re-run physical tests and
independent replay before using any results. No threshold/seed removal or
new favorable subset after reading development outcomes.
