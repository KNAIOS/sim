# Full-count-domain revision experiment (fixed before running)

Detected axes: task_type=communications; evidence_type=baseline, ablation,
statistical, reproducibility; failure_mode=unfair-comparison, overclaimed-results;
stage=planning.

## Question and evidence gate

The old four-module floor is not a rank requirement of the implemented
fully digital, eight-antennas-per-module precoder. Does a freshly trained
count controller benefit from the physically admissible 1--16 count range,
relative to non-learning controllers with the **same** range?
The previous four-trace search diagnostic is not evidence of an RL advantage.

| Claim | Required evidence | Present gap | This bounded experiment |
|---|---|---|---|
| RL reduces energy at acceptable outage | Calibrated strong baselines, common count domain, paired independent traces | Old domain imposed an unnecessary floor; previous RL candidates failed | New-domain calibration/training and development screening |
| The hardware/accounting are unchanged | Replay at minimum four, energy-component equality, causal inputs | New adapters could introduce an implementation mismatch | Automated regression tests before training |
| Results justify a manuscript replacement | Stable learned improvement, final-model sensitivity and standardized-channel checks | No full-domain learned model yet | Promotion gate only; no automatic replacement |

## Fixed implementation and budget

- Physical channel, eight antennas/module, RZF, power, wake delay, observation
  restrictions, ranking and 140-slot/7-second accounting remain unchanged.
- New DQN: 235 inputs = original enriched state 139 + 16x3 fully-ready rate
  ratios + 16x3 actually-ready rate ratios. These are computed only from
  last-observed CSI and copied current hardware. Two 64-unit hidden layers,
  16 outputs for counts 1--16. No old checkpoint is relabelled or warm-started.
- Double-DQN, replay and optimizer follow the existing implementation. Train
  two seeds (11,22), three fixed outage prices (1,2,4), 300 episodes each.
  Save/check validation at episodes 100,200,300: 1800 training episodes maximum.
  Continuing-task training has one real successor sample, never charged as
  an extra executed slot. Epsilon .04+.76 exp(-4 ep/299).
- PRS and switch-aware finite-horizon search both enumerate 1--16; the latter
  also allows the unchanged previous command. Margins .90,1,1.04,1.08,1.16,1.24.
  True service requirements are never modified by these prediction margins.
- Same scenarios (.5 m/s,.85 load), (2,1), (3,1.15).
- Streams: training 981000000+seed*10000+episode; validation
  982000000+scenario*10000+i (12/scenario); development
  983000000+scenario*10000+i (16/scenario). None was used in the prior pilot.
  A genuinely independent confirmation stream is reserved but not run here.
- Checkpoint/price selection uses validation ONLY: minimum worst-scenario
  mean-outage excess above .05, then mean energy; one selected model per seed.
  Search gets one validation-selected margin per scenario, preferring feasible
  mean outage then least energy. This gives search explicit scenario-specific
  tuning; the DQN is one policy across scenarios. Report that difference.
- Persist the full selections before reading development results. Keep all
  training and validation rows, including failed arms. No finer sweep based on
  development outcomes. At most two single-thread workers in existing venv.

## Analysis and stopping rule

Report energy, outage, switching/operating/pilot components, per-training-seed
outcomes and paired trajectory bootstrap intervals. Compare both searches and
full array; old-domain data are contextual, not paired new-domain baselines.
Average the two trained models within each trajectory for the paired estimate;
also resample training seeds and trajectories as a limited, exploratory check.
Two seeds are not enough to establish general training stability.

Promote a candidate only if at least one prespecified scenario has mean outage
and its one-sided 95% trajectory-bootstrap upper endpoint at most 5%, lower
energy for **each** trained seed than both calibrated feasible search controls,
and the 95% paired energy-difference interval excludes zero against both.
Otherwise stop this bounded variant, preserve the negative result, and do not
substitute it into the manuscript. Passing triggers a separately frozen
five-seed/independent-trace protocol and final-model R2/R3; it does not by
itself close R1. No claim of deterministic reliability or globally optimal RL.
