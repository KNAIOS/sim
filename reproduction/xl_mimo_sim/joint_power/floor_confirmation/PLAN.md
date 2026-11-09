# Locked development check after floor-projection screening

2026-09-22. Written BEFORE accessing development traces. Phase-seven validation
passed the previously declared screening gate for Visitation-added only:
nominal116.674J, versus119.800J budgetPRS, both seeds under5% in each scenario.
This is not matched-outage dominance: learning outage3.958%,PRS0%, and high
scenario learning is more costly. Do not hide either result.

Use the reserved stream4009220000, four traces/scenario, as declared in the
earlier CONTROL_PLAN.md. An output-manifest/report search found no previous
execution of this stream. It is independent of all training/selection traces,
but is called DEVELOPMENT, not an untouchable final test. Once inspected it
cannot be relabeled as pristine data for a future revision. No parameter or
checkpoint updates during or after this run.

Retain BOTH seed11/22, epoch20 Visitation-added models. Alpha=.1 for corrected
Value, corrected Myopic and uncorrected Value. Fixed-price Visitation-added
uses the previously selected lambda4 for both seeds. The PRS-added family
failed screening and is fully preserved in phase-seven reports, not chosen
for this confirmation. Selection of the Visitation-added family is disclosed.

Non-learning controls: PRS original=.05 (identical to validation-selected
PRS tuned, report this alias), PRS budget alpha1, Raw full threshold.05,
Raw coarse threshold0 with stride3, Full power, Raw budget alpha0 and
corrected Raw budget alpha0. All powers, energy, information, fallback and
candidate sets stay as implemented. Coarse subset is NOT an equal-compute
claim. Eight learned and seven non-learning configurations =>180 episodes.

Prespecified confirmation: every seed's mean outage in EVERY scenario <=5%,
nominal average energy >=2% lower than the best feasible non-learning control,
neither seed regresses against that control. Also report every trajectory and
pathwise5% exceedance, all scenario means, paired energy and outage differences,
training-seed effects, energy components, and descriptive whole-trajectory
resampling95% intervals. Twelve trajectories and two seeds are a bounded
development check, not a broad deployment or statistical-superiority proof.
If intervals overlap zero, explicitly avoid a significant-improvement claim.

Independent antenna-domain physical/energy verification for every slot.
Check selected physical floors for corrected rules and reproduce metric
summaries from raw slots. Frozen weights, source and previous artifact hashes
before/after. No installs, SSH, environment changes, manuscript claims or
submission triggered by the outcome. If gate fails, no tuning or enlarged
confirmation of this frozen candidate; retain failure and report R1 open.
