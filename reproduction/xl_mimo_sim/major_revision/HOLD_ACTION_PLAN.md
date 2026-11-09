# Audit of the omitted current-command action

2026-09-22. Freeze this protocol before running the new audit. This is a
diagnostic and baseline-completeness correction, not a new RL contribution.

Detected axes: communications / baseline,ablation,statistical,reproducibility /
unfair-comparison,overclaimed-results / audit.

## Why this audit is needed

The early `FullDomainRollout.command(mode='cost')` includes the previous command
in addition to ranking prefixes. Later `action_value.baseline_command` and the
risk-correction interface contain only 16 prefixes. Re-ranking can therefore
remove the literal no-change option. Existing retention tests kept the PRS
count and used the older predictor; they did not test a held previous count
under the newly calibrated risk score. No claim that holding is novel.

| Claim | Needed evidence | Existing evidence | This audit | Risk |
|---|---|---|---|---|
| Missing no-change action causes avoidable switching | Previous command absent, chosen with identical objective, physical paired benefit | Source omission only;108.5 nominal switches with new risk | Census of all decision epochs and causal hold-action intervention | High if inferred from switch counts alone |
| Restoring this action improves full-policy energy at adequate service | Paired complete episodes and all adverse scenarios | None for the new risk interface | Prefix versus prefix-plus-current-command for each predictor | High; risk/energy tradeoff is not a win |
| RL adds benefit | Fair new learner and equally capable search on fresh data | Last four fits select checkpoint0 | Not tested; no training in this audit | R1 remains open |

## Unchanged setting and inputs

Use the existing geometric channel, demands, power/transition accounting,
observation timing, four-slot command clock, four-slot recovery and H=8.
Predictors: old filtered-current, anchored measured-current, calibrated risk.
Use exactly the previously frozen stream119 parameters per scenario. The
calibrated model is unchanged. It was fitted on prefix candidates, so its
calibration on new held-mask states is NOT guaranteed.

For each predictor compare:
1. the existing 16-prefix search;
2. the identical selection rule with the current hardware COMMAND appended.

The appended mask includes waking branches, not only ready branches. Compute
its energy and risk with exactly the same hold prediction and feasibility rule.
Enforce the existing minimum on the mask's actual cardinality. Duplicate masks
are harmless; existing prefix wins exact ties. No new parameter, switching
penalty, forced dwell, truth-based safety filter, or physical-model change.

## Viewed-data diagnosis, not independent confirmation

Replay all24 already-viewed stream121 trajectories (8/scenario) without fitting.
Three predictors x two action sets =144 complete episodes. Verify all72 original
prefix episodes against the previous archived search results. All35 decision
epochs per prefix episode are surveyed (2520 correlated roots total).

At each prefix root log current/prefix/extended commands, candidate risks and
energy, duplicate/eligibility flags, number of on/off commands and simultaneous
replacement pairs min(on,off). At every root where the extended choice differs,
execute that causal choice for one four-slot block, then follow the ORIGINAL
prefix policy for the rest of the episode. Keep every such branch, including
adverse outcomes. Its difference is relative to the complete original episode;
root effects cannot be added or presented as independent trajectories.

Save the causal action table before computing any counterfactual labels. Future
channel/demand may be used only by the physical rollout, not action selection.
Compute full-policy paired deltas and descriptive trajectory-bootstrap95% CIs,
10000 resamples, seed1220900001. Do not pool training seeds or root branches as
independent observations. Replaying viewed data cannot establish generalization.

## Checks and stopping

- First16 candidate arrays/actions exactly reproduce the frozen interface.
- Scalar original PHY/hardware agrees for the additional mask at every root
  with a full eight-slot suffix (up to t=132); no invented terminal data.
- Hidden-future perturbation does not change causal predictions or choices.
- All144 policy episodes and all changed-action branches independently replay;
  all72 original prefix episodes match saved traces and metrics.
- Source/parameter/model/trace hashes, no root mutation, energy decomposition.
- Existing plus new unit tests; project venv only, at most two one-thread jobs.

The exploratory gate for considering another method design requires calibrated
extended search to have nominal mean outage<=5%, no greater mean outage than
its prefix control, and strictly less mean energy. A pass would NOT establish
RL superiority, novelty, independent significance or R1 closure. Report all
scenarios regardless. A failure ends this mechanism candidate without new RL,
extra tuning or a standard-channel run. Original sources, checkpoints, figures,
paper and reviewer responses remain untouched. Author reports may be updated.
