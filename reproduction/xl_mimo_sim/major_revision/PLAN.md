# Major revision pilot: frozen development protocol (2026-09-20)

Status: implementation/pilot, NOT a final revised method or submission evidence.

## Audit findings

1. Existing energy reward already charges switching and uses discounted returns.
   Adding these is not a new contribution.
2. Normalized rank scores discard common absolute gain changes. A unit test
   confirms that uniform 6-dB attenuation leaves the original state unchanged
   at initialization, while the enriched state distinguishes it. This is a
   representation limitation, not proof that it causes all observed losses.
3. PRS uses complex-channel information to predict RZF rates; the DQN's original
   compressed state does not retain equivalent information. New absolute-gain
   features only partially address this; user-channel phase/correlation remains
   compressed. Do not claim full information equivalence of representations.
4. Original model selection pools validation scenarios. Pilot selects by the
   worst of three scenario mean outages before minimizing average energy.
   This is not a statistical/per-slot feasibility guarantee.

## Changes being tested

- Same original simulator, power model, switch energies, 50-ms slots, 4-slot
  action clock and 8-slot prediction horizon. No inflated switch costs.
- Same DQN width, optimizer, discount and energy/outage reward. Original state
  dimension 72; enriched dimension 139: add 48 causal absolute-gain/noise
  features, 16 observation-age features, and 3 changes in required user rate.
  More inputs mean more parameters: this is not a matched-parameter ablation.
- Both DQN variants are trained afresh under equal episode budgets and matching
  trajectory streams. Five-seed/full-budget confirmation is deferred.
- PRS unchanged; hysteresis PRS keeps new activation commands for 8 slots;
  switch-aware PRS rolls out 13 rank prefixes plus holding the old mask, charges
  transition/pilot/operating energy and selects feasible minimum forecast energy.
  It includes the current unavailable slot and exact hardware countdown. Thus
  the strengthened baseline differs in both objective and forecast alignment;
  these factors would need separation for a mechanistic publication claim.
- Pilot does not claim an information-rich DQN has lower runtime than PRS.

## Locked development settings

- DQN methods: Original-DQN, Enriched-DQN; seeds 11,22; 80 episodes each.
- Scenarios (speed m/s, rate multiplier): (0.5,0.85), (2,1), (3,1.15).
  These coupled scenarios are screening cases, not a factorial speed/load study.
- Validation: 3 independent trajectories per scenario; checkpoints every 20 episodes.
- Development evaluation: 8 new trajectories per scenario shared by all policies.
- Original/Enriched DQN, PRS, hysteresis PRS, switch-aware PRS, full array.
- Report total, operating, pilot and switching energy; outage; transition count;
  reactivation within eight slots of shutdown. Do not reinterpret only reduced
  switch count as an energy gain.
- Code/config hashes and seed streams saved before executing the pilot.
- Development outcomes may inform the next design; never label this same set
  as untouched final test data after inspecting or tuning against it.

## Claim/evidence gates

| Claim | Evidence required | Pilot role | Missing for publication |
|---|---|---|---|
| Added state helps | old/new DQN under equal reliability | screening only | 5+ seeds, fair capacity ablation, paired intervals |
| RL saves over search | energy/outage frontier against strong PRS | screening only | validated comparable reliability, new test stream |
| Savings arise from less cycling | energy breakdown and churn | diagnostic | controlled mechanism ablation |
| Generality beyond geometric model | calibrated standardized or measured channel | not started | standard channel implementation/calibration/evaluation |

Do not enlarge training simply to force a winning result. Stop and reassess
if no useful energy/reliability tradeoff appears. Both local and remote tests
must pass before result interpretation. Remote execution uses the existing
package-local venv and one worker at nice level 10; no host environment changes.
