# Major revision evidence and completion gates

This file tracks the complete requested revision, not merely the latest pilot.
No result below guarantees acceptance or authorizes omitting adverse baselines.

Latest completed consistency audit (2026-09-21): SERVICE_EQUIVALENCE_REPORT.md
checks all864 archived roots. Commands with identical four-slot available-mask
sequences have EXACTLY equal sampledC4 outage labels (max error0), but all four
networks violate that equality. Parameter-free group averaging obeys the basic
orthogonal projection identity (error2.22e-16) and lowers absoluteC4 riskMSE by
only0.607--1.044% on viewed validation. Nominal free-policy sample weighted costs
do not improve consistently; this is not a closed-loop result. Complete-action/Q
equivalence and a long-horizon error guarantee are explicitly false; a C16
counterexample is unit tested. No new simulation/training/promotion,125 tests
pass, jobs exited, PDFs/environment unchanged. This is completed diagnostic
progress; R1/full revision remain incomplete, not an external blocker.

Latest completed follow-up (2026-09-21): ABSOLUTE_RETURN_SCREEN_REPORT.md
tests the fixed free absolute-return models on24 new106-stream trajectories.
No retraining or learned-price sweep. Four searches x12 settings plusmu/full
produce1200 baseline records; two seeds xC4/C16 produce96 learned records.
Matched-current-CSI searches are verified causal; original searches replay
unchanged. NominalC16 is133.22J/4.17% versus cost138.34J/3.27%; both seeds use
more outage than the frozen0.5pp tolerance. ALL SIX promotion cells fail.
Post-hoc49-point expected episode-mixture frontiers match each learner's risk:
nominal RL-minus-mixture is+0.953/-0.500J, not consistent superiority. Mixture
probabilities are validation-fitted expectations, not independent deployments.
123 tests pass; jobs exited; PDFs/environment unchanged. R1 remains unresolved.
This turn completed new evidence and is progress, not a blocker/wait. Original
failed structural/closed-loop gates remain intact; no confirmation permitted.

Latest (2026-09-21): STRUCTURAL_VALUE_REPORT.md completes the known-cost and
conditional-future audit: 864 roots' exact C4 energy, 48 fixed roots x16 future
draws, 1536 paired records. Low/high mean cost-sign errors persist under repeated
futures; selected prediction errors are mainly priced-outage underestimation.
FACTORED_VALUE_REPORT.md then records a new frozen known/unknown-cost candidate:
108 causal trace replays, 1500 features, two variants x two seeds x100 epochs.
C16 outage-difference MSE drops 43.7--46.0% versus the old model, NOT an energy
saving. The free ablation has smaller overall errors; structured seed22 has
positive nominal energy difference. Final gate FAILS, no closed-loop permission.
All variants retained, no post-hoc promotion of the free ablation. A float64
precision audit clarifies a NumPy scalar-rounding diagnostic (max1.99e-8), without
changing frozen inputs or checkpoints. 117 tests passed, owned jobs exited,
PDF hashes unchanged, environment untouched. These are substantive completed
diagnostics/reconstruction work, not a repeated external blocker. R1 and the
overall goal remain active/incomplete; subsequent candidates need a new bounded
evidence-based protocol, not relaxed gates or extra epochs on this failure.

Latest completed reconstruction (2026-09-21): RECOVERY_CAUSE_REPORT.md checks
840 states across 24 exact replays; recovery-limited predicted states are
7/40/59 for H8 versus 7/8/5 for calibrated cost search. This is a conditional
hardware counterfactual, not a deployable zero-delay strategy or proof that RL wins.
ROLLOUT_VALUE_REPORT.md then records a separate frozen paired-return study:
864 root states, 13824 branch labels, four fixed 100-epoch fits, 936 baseline
and 720 learned closed-loop validation evaluations. Both C16 seeds pass the
regression gate but ALL SIX performance promotion cells fail. C16 mean energies
114.02/145.95/219.74 J and outages 1.36/4.85/11.42% do not beat strong feasible
search. No confirmation is permitted. Post-hoc action-ranking checks already
show sign errors on reference-policy states; single-future labels do not prove
conditional-expectation bias. A report-only mixed-CSV schema error was fixed
without retraining/reselection; complete audit is closed_audit_v2, partial first
report and its source snapshot retained. All simulation/fit jobs exited; PDFs
unchanged; R1 and the full manuscript goal remain incomplete. This turn made
progress and is not an external blocker or verified wait.

Latest completed H8 prior screen (2026-09-21): HORIZON_PRIOR_REPORT.md records
84 fixed-policy evaluations. Validation outage improves to 3.13/10.54/20.93%,
but only one scenario meets 5%; the frozen training gate FAILS. No H8 training,
horizon/price sweep, or manuscript replacement is allowed for this candidate.
Development is 101.09/114.00/140.65 J with 3.29/10.18/24.58% outage. A declared
post-hoc audit on 12 viewed trajectories/420 decisions finds ZERO choices of
predicted-infeasible actions when a feasible candidate exists. States with no
predicted-feasible candidate are 7/40/75 out of 140 per scenario. This refutes
the suggested local-choice explanation on this sample, not all penalty-based
control, and does not prove physical infeasibility or any trained-RL mechanism.
Both jobs exited, 99 tests passed, both manuscript PDF hashes unchanged.
R1 and the overall revision remain open; see report for next evidentiary boundary.

Latest completed stage B: ACTION_VALUE_REPORT.md records two 200-episode
action-conditioned Double DQN runs, 936 baseline validation records, 360 learned
validation records and 288 development evaluations. Both select episode150;
all six screening cells fail. Development RL uses 134.26/156.08/185.98 J with
0.42/1.99/12.25% outage; explicit cost search uses 101.01/131.32/207.30 J with
3.39/3.12/5.89%. A post-hoc audit of 840 viewed states and 10219 adjacent
sleep-module action pairs verifies that the four-slot prior sees 0.33 J extra
cost and no service before recovery. This does not establish the cause of
learned-policy failure or a better replacement. HORIZON_PRIOR_PLAN.md freezes a
separate eight-slot prior screen before any new training. R1 remains unresolved.

Latest completed diagnostic: DEMONSTRATION_REPORT.md audits the strengthened
nested PRS, 3360 teacher training labels, 1260 held-out labels, two network seeds
and two fixed own-state aggregation rounds. Both seeds FAIL the prespecified
initialization gate (selected held-out agreements 56.11% and 55.56%); no RL
fine-tuning is launched. On 48 new development traces, selected imitation uses
106.67/133.85/195.18 J with 4.87/5.89/6.85% outage; matched nested PRS uses
107.11/130.64/185.41 J with 2.43/3.88/3.08%. No energy advantage at comparable
reliability is established. All three endpoints and failures are retained.
Checkpoint recovery changes NumPy scalar metadata only, preserving all original
files and identical weights; strict loading, receipts and selection are audited.
All processes have exited; 85 tests pass. No system/environment/physical-model
changes, no main-paper substitution, and no assertion that R1 is resolved.
The author has now explicitly chosen core-method reconstruction with bounded
validation and no guaranteed superiority. CORE_REBUILD_PLAN.md freezes stage A
before execution. No failed imitation endpoint is promoted or fine-tuned.

Core-rebuild stage A is now complete: GRAM_INTERFACE_REPORT.md records exact
per-subarray Gram/RZF computation on 420 causal states and 161280 forecast rates,
maximum absolute discrepancy 3.77e-10 Mbit/s, zero threshold mismatches and 12/12
matched physical replays. A constructed pair has identical old 235-feature
inputs but a 10.33 Mbit/s forecast-rate difference, distinguished by the new
362-feature representation. This is an information-retention check, not evidence
that the pair occurs in generated data, that it caused earlier learning failure,
or that the new method saves energy. No new RL training has been launched;
stage B requires its own frozen action-value learning protocol and strong controls.

Latest completed continuation: FULL_DOMAIN_REPORT.md records all 1800 full-domain
training episodes, 432 baseline validation rows, 648 learned validation rows and
240 development rows. None of the three scenarios passes the promotion gate.
RL energy is 162.45/188.84/231.32 J with 0.14/2.54/11.94% outage; PRS is
101.85/133.87/266.30 J with 3.10/3.20/6.10%. An explicitly post-hoc, all-trace
physical-bound audit checks 84 fixed traces: the high-load development equal-power
unavoidable-outage lower bound is 4.807%, versus full-RZF 5.342%. The bound is
below 5%, so it does not prove strict infeasibility or excuse RL's additional
failures. All jobs have exited. Main-paper model/results remain unchanged; a
yellow wording correction makes predicted rather than true future CSI explicit.
R1 remains unresolved and the overall goal is not complete.

Earlier running note (now completed): FULL_DOMAIN_PLAN.md fixes a fresh 1--16-count
study before outcomes: a new 235-input/16-output Double DQN, two seeds, prices
1/2/4, 300 episodes per arm, plus common-domain PRS and switch-aware search.
All use new validation/development streams; the selected price/checkpoint must
be validation-only. The initial baseline process was deliberately interrupted
before cost-search results because code review found a filtered-current-gain
forecast mismatch. FULL_DOMAIN_BASELINE_CORRECTION.md explains the preserved
partial output and exact `baselines_v2` replacement. The learner does not call
that branch. Six adapter and two statistical-report unit tests have passed.
No new superiority claim or main-manuscript substitution is made while running.

Latest continuation: SPATIAL_ACTION_REPORT.md completes two same-count action
diagnostics and an explicitly disclosed all-epoch coverage amendment (420 causal
states, 840 candidate rows, 27 changed-action continuations). The local action
signals are small and do not establish a learned advantage. COUNT_DOMAIN_REPORT.md
then identifies the imposed K+1-module floor: it is not a physical rank condition
in the implemented full-digital, Ns=8 model. A 24-rollout domain audit gives lower
energy when PRS may select 1--3 modules, but nominal/high-load outages fail 5%.
The next technical priority is a common full count domain with fresh validation
calibration for ALL methods, not another fine price sweep or a claimed RL gain
from domain expansion. Frozen sources and original-model data are preserved.
The manuscript now explicitly discloses the action restrictions and cites the
verified He et al. TWC 2024 prior work; main results remain the original model's.
All 68 tests pass. R1 and the full major-revision goal remain unresolved.

Current continuation: the completed validation reward audit confirms that price 4
ranks costlier, lower-outage checkpoints above PRS even when PRS is mean-feasible.
REWARD_CONSTRAINT_REPORT.md records 180 replays and the finite-discount limitation.
PRICE_CALIBRATION_PLAN.md specified the now-completed bounded {1,2,4} comparison,
reusing the finished price-4 arm, plus expanded search margins and a non-learning
mixture. PRICE_CALIBRATION_REPORT.md records the result: selected high-load RL
172.91 J / 3.83% versus PRS 174.70 J / 3.42%; paired excess -1.79 J with interval
[-5.75, 1.83] J. The mixture expectation is 172.11 J / 4.41%, with an outage
interval crossing 5%. No scenario passes the predeclared promotion gate. All
61 tests pass and the bounded workers have exited. No finer price search started.
Historical statements below that runs have exited refer to their own completed
studies. No new candidate has been promoted and R1 remains unresolved.

| Reviewer requirement / deliverable | Authoritative evidence | Current status | Remaining completion gate |
|---|---|---|---|
| R1: establish a useful learned-policy advantage against PRS | Original sensitivity, independent candidate check, constraint/readiness/time-limit development reports | Independent confirmation failed reliability; later mean-feasible candidates still consume more energy than both search baselines | Author decision on narrower computation/performance claim or a separately specified new method-development plan; do not label R1 resolved |
| R2: heuristic sensitivity | `results/reviewer_20260919/sensitivity_raw.csv` and summary, 1920 records | Completed for original five frozen policies and two OAT ranking parameters | If final model changes, repeat on final model; distinguish ranking heuristics from calibrated probabilities |
| R3: standardized/measured validation | `../../standard_channel_20260920/results/manuscript_validation_v1`, `checks_v1` | Completed 20 new LOS + 20 NLOS realizations, two loads, 720 evaluations; 560 original-model/baseline records in Table III; geometry metadata corrected | Compact-array/fixed-topology validation added with limitations; do not extrapolate to moving near-field VRs or universal reliability |
| Channel numerical validity | `checks_v1/checks.json`, `transfer_screen_v1/audit.json`, `major_revision/test_wideband.py` | Pathloss/LSP/reproducibility and quadrature spot checks pass | Use 99-bin frequency quadrature for formal runs; check outage sensitivity and broaden trace/parameter coverage; do not claim complete multi-cell 3GPP calibration |
| Fair physical accounting | `future_vr/system.py`, pilot/wideband tests, raw CSV energy identities | Common power, pilots, switching and active-only observations preserved | Carry accounting through final code; disclose ideal active-CSI RZF, analytic power proxy and prediction information differences |
| Claim-supporting figures/tables | Original figures + Tables II and III in 20260920 revision | Original 72-dimensional method retained; no candidate substitution; all added table values verified | Scientific R1 gap remains despite correct presentation |
| Complete revised manuscript | `../../ICECAI2026_Revision_20260920` | Seven-page author-review draft with abstract, introduction, experiments and conclusion aligned to evidence | Author decision on scientific scope; verify venue page allowance before submission |
| Yellow-highlighted changes | 20260920 highlighted and clean PDFs, shared source | Both compiled; extracted normalized character multisets match (float/line order differs); fonts embedded; final-page columns balanced; all 14 rendered pages inspected | No portal-compliance claim; verify venue page allowance |
| Point-by-point response | `../../ICECAI2026_Revision_20260920/RESPONSE_TO_REVIEWERS.md` | Updated with completed evidence and explicit unresolved R1; contains author-only decision note | Author review required; not a claim that all concerns are resolved |
| Environment isolation | Server `/tmp/bupt-energy-20260920-ZNIAVc/.venv`; local `standard_channel_20260920/.venv`, lock and verification | No system/driver/global-package changes | Keep all future installs, caches, outputs and jobs scoped; preserve originals |

## Independent confirmation supersedes the development signal

The fixed-price sweep (600 episodes) and rate-context ablation (200 episodes)
are complete. Their four-trajectory development signal did not survive independent
confirmation on 40 new trajectories: rate-context DQN has 195.49 J / 8.50% outage,
whereas validation-tuned PRS has 206.72 J / 3.60%. Energy reduction comes with
failure of the 5% target. See CANDIDATE_CONFIRMATION_REPORT.md; all unsuccessful
candidates and baselines remain recorded. The original five frozen policies were
also evaluated on 40 new standardized UMi realizations. Table III retains all
four state/load conditions, including reliability failures. No training or
evaluation job remains active. Further work needs a defensible scope decision,
not an indefinite search for favorable test results.

## Subsequent bounded diagnostic and next concrete method issue

The 1200-episode fixed-price/longer-budget diagnostic and its 336-record fresh
development screen are complete. All selected policies are mean-feasible in the
development scenarios, but both prices consume more energy than both tuned search
baselines. High-load price-4 DQN is 213.12 J / 2.28% outage versus PRS
174.70 J / 3.42%; paired excess energy is 38.42 J [24.85, 50.70]. See
CONSTRAINT_TRAINING_REPORT.md. No positive energy signal passed the screen, so
there is no justification to promote that candidate into the manuscript.

A constructive signal/hardware test identifies a remaining information alias:
the same 178-dimensional state can correspond to 349.64 versus 368.70 Mbit/s
for the first two users after the same switch, at a 360-Mbit/s target. The
readiness-conditional feature implementation distinguishes this pair without
future CSI or physical-model changes. See READINESS_STATE_ALIAS.md. This is a
specific next modeling hypothesis, not empirical proof of an energy benefit.
The full major-revision objective remains active; no narrower completion claimed.

## Non-negotiable final reporting

### New causal-action audit and bounded residual pilot

The frozen-policy intervention audit completed 96 paired starts. In 47, one
causal PRS action followed by the original RL reduced realized energy without
extra outage. A separate PRS-local audit completed 48 decrement and 48 increment
starts, retaining all failures. At high load an increment costs 0.341 J in the
first block but changes whole-window energy by -0.750 J on average; this is a
small correlated diagnostic, not a deployable-policy result. See
POLICY_INTERVENTION_REPORT.md and SEARCH_HEADROOM_REPORT.md.

PREDICTIVE_RESIDUAL_PLAN.md specified a two-seed, 300-episode-per-seed pilot:
PRS anchor plus learned {-1,0,+1} correction, same physical model, reward and
causal observations. A checkpoint-zero exact anchor fallback and fixed +/-1
controls prevent attributing the baseline's success to learning. Four new
controller tests verify anchor identity, boundary masking, finite TD updates and
real successor handling. The pilot is now complete: both seeds select checkpoint
zero, exactly PRS. All 96 fallback records match the anchor; fixed -1 has high-load
167.59 J / 6.13% outage, and fixed +1 has 186.72 J / 1.65%. No learning gain is
established. See PREDICTIVE_RESIDUAL_REPORT.md. All 51 tests pass and all jobs have
exited; no main-paper substitution, independent confirmation or R1 closure.

### Completed readiness-feature and time-limit studies

The 217-dimensional readiness-context ablation is complete: two training seeds,
600 episodes, and 96 new evaluations on the shared development traces. Against
PRS its low/nominal/high-load mean excess energies are 10.32/33.74/41.74 J;
all paired development intervals are positive. Against the 178-dimensional
learner the energy-difference intervals all cross zero. See
READY_ABLATION_REPORT.md; the main manuscript remains unchanged.

TIME_LIMIT_PLAN.md specifies a separate continuing-control training audit.
The final replay target now has an opt-in actual successor observation rather
than a false terminal flag. The default old engine remains unchanged, and
22 major-revision unit tests pass, including prefix identity and successor
equality against uninterrupted replay. Both single-thread, 300-episode runs
finished. High-load continuing DQN is 219.92 J / 1.33% outage versus PRS
174.70 J / 3.42%; paired excess is 45.22 J [34.49, 55.21]. It also consumes
more mean energy than the terminal-window learner in all three scenarios.
See TIME_LIMIT_REPORT.md. About 97.6% of its high-load excess energy is operating
energy, not switching charges. No method is promoted into the manuscript, no
new superiority claim is justified, and all training processes have exited.

- Saving against an always-on full array is not superiority over PRS.
- A lower outage at higher energy is a tradeoff, not an energy win.
- Meeting a small-sample mean outage threshold is not a reliability guarantee.
- Do not reuse viewed development data as untouched final testing.
- Compact standard MIMO does not validate near-field XL visibility dynamics.
- Standard pathloss plus a homemade fading process is not a complete standard
  channel; here the pinned external UMi implementation generates both.
- Never reduce error bars by filtering unfavorable seeds or relabeling them.
- If no energy advantage survives fair comparison, revise the claim explicitly
  and explain the remaining reviewer risk; do not manufacture an advantage.
