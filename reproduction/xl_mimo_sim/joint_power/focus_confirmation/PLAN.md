# Independent focused confirmation -- locked before new data generation

Date: 2026-09-22. This is a local, recorded prespecification, not external trial
registration. The half-load NLOS focus was selected from earlier broad results;
the present new sample must be kept separate from that exploratory sample.

## Claim and fixed scope

Test whether the frozen BA-RL4 controller saves mean episode energy versus the
unchanged current-wideband-informed budget search WB-B4, with both below the
5% expected user-slot rate-failure target. No claim of equal reliability,
per-episode safety, algorithmic/global optimality, or arbitrary training-seed
generalization follows from this test.

- Existing Sionna 2.1.0 UMi NLOS generator, fixed topology/LSP and 2 m/s Doppler.
- Half-load requirements only (factor 0.5), 140 slots, 99 frequency samples.
- 4--16 requested subarrays for both methods; algorithmic domain, not hardware.
- Identical physical model, causal CSI access, allocation and energy accounting
  as joint_power.standard_transfer.control. No new policy implementation.
- Frozen Visitation-added checkpoints seeds 11/22 at epoch 20 and unchanged
  histogram helper, beta=0.65, discount=0.93, alpha=0.1. No retraining or tuning.
- Only the primary WBbudgetPRS4 reference is rerun. It is not weakened or given
  artificial noise. All earlier methods/results remain intact.

## Fresh data and bounded computation

- Exactly 40 independent channel-demand pairs, indices 0..39.
- Channel seed = 6009220000 + index; demand seed = 6109220000 + index.
  These namespaces were checked against prior recorded code/manifests before use.
- Both RL seeds and WB-B4 run every pair: 120 episodes / 16800 physical slots.
- Maximum two worker processes, each single-threaded CPU. Existing project-local
  Sionna venv/run.ps1 only. No installs, SSH, global config or environment changes.
- No outcome-based early stopping, sample expansion, deletion, or seed replacement.
  A technical failure is reported and diagnosed; it does not authorize a new
  random sample or a policy change. Completed traces are always retained.
- Save all per-slot and decision ledgers. Keep channel hashes for every drop and
  channel arrays for drops 0/39 for independent post-save replay.

## Analysis specified before execution

1. Report new-sample method means and separate RL-seed means for energy, rate
   outage, and the existing energy components. No pooling with the earlier 20.
2. Primary energy difference: BA-RL4 minus WB-B4, pairing on each of the 40
   exogenous drops, averaging the two frozen learned policies. Report percent
   saving relative to WB-B4.
3. Use 10000 crossed bootstrap resamples, random seed 2026092260, resampling 40
   drops with replacement and the two learned seeds with replacement. Retain
   pairing within every resample. The two-sided 95% interval is percentile
   [0.025,0.975]. Only this one primary baseline contrast is tested.
4. Use the same resampling index arrays for energy and outage. Report the paired
   outage-difference interval, plus each method's one-sided 95% upper bound for
   its mean outage. Lack of a significant difference is not equivalence.
5. Prespecified positive evidence rule: upper endpoint of energy-difference
   interval < 0 AND the one-sided 95% upper mean-outage bounds of BOTH methods
   <= 0.05. This conservative operational rule is not a calibrated finite-sample
   safety certificate. No required minimum saving is imposed after seeing results.
6. If this rule fails, report which condition failed and all numeric outcomes;
   do not claim independent confirmation or automatically tune/rerun.

## Verification and evidence boundary

Verify source/model hashes before/after; replay every physical slot independently
from logged powers, active masks and original channels. Audit written aggregates,
decision lower bounds, complete sample indices and summary recomputation.
The evidence concerns these two frozen policies under the compact-panel transfer
model; two training seeds and unmeasured hardware power remain limitations.

| Claim | Required evidence | Current step | Not justified |
|---|---|---|---|
| The selected saving repeats | Fresh paired fixed-policy comparison | This run | Universal RL superiority |
| Mean reliability constraint is met | New-sample outage and upper bound | This run | Per-episode safety/equal service |
| Energy numbers follow the model | Independent physical replay | Every slot | Measured device power |
