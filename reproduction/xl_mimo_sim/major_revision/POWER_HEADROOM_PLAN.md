# Fixed-power headroom diagnostic (2026-09-22)

Scope: a bounded, non-learning diagnostic of an optional power-control extension.
The existing manuscript, simulator, policies and saved results remain unchanged.
Author permission is still pending for integrating joint power/subarray control.
This diagnostic cannot establish RL superiority or close review comment R1.

Use all 24 already-viewed stream-121 trajectories (8 per scenario), 140 slots.
Replay the frozen old search, calibrated search, and full-array schedules. Search
settings and risk model are unchanged. No fitting, parameter sweep or selection.
Save every slot, including failures and numerical fallbacks.

At each slot, keep the actual active subarray set and original normalized RZF
beam directions (regularization based on the original peak 1.2 W). Only evaluate
the counterfactual allocation among these fixed beams. Active-branch current CSI
is already observed by this simulator; neither sleeping CSI nor future CSI is
provided to the allocation solver. Recovery delay remains four slots.

For gain matrix g[k,j]=|h_k w_j|^2 and original powers p0=Pmax/K, set target rate
to min(original achievable rate, required rate), with a tiny numerical interior
margin for previously successful users. This preserves the original service
failures instead of claiming that power control repairs an infeasible slot.
Let gamma=2^(target/[B(1-pilot_fraction)])-1. Solve (I-F)p=u with
F[k,j]=gamma[k]*g[k,j]/g[k,k] for j!=k, u[k]=gamma[k]*noise/g[k,k].
For fixed beams and positive noise, if spectral radius(F)<1, this is the
componentwise minimum feasible power vector. Verify nonnegativity, peak power,
rate targets, and the exact original success/failure indicators. Fall back to p0
if any numerical check fails. A 1e-8 blend toward p0 is numerical interior only,
not an uncertainty calibration. No SINR or channel-model simplification is used.

Energy difference is dt*(Pmax-sum(p))/PA_efficiency only. Preserve all circuit,
sleep, wake, switching, pilot and existing processing energy. Added allocation
computation energy and a nonconstant PA-efficiency curve are not measured and
must be disclosed. Per-episode absolute PA-saving ceiling: 7*1.2/0.4=21 J.
The adjusted result is counterfactual schedule accounting, not a trained joint
policy and not an end-to-end benchmark of that optional extension. Achievable
sum rate can decrease while minimum-rate satisfaction stays the same; do not
claim unchanged throughput/goodput or guaranteed energy-efficiency improvement.

Checks: original PHY equivalence; linear inequalities and powers; per-user
failure identity; recorded old/calibrated trajectory replay; disjoint state
access; analytic orthogonal and interference examples; zero/rank-deficient
channels; source/input hashes. Existing project venv only, one worker/thread.

Required output: all slot allocations, episode/scene summaries, paired
trajectory bootstrap descriptions, limitations, no-learning marker. Stop after
the bounded diagnostic regardless of direction; do not launch RL training.

Claim-evidence matrix:

| Claim | Evidence required | This study | Remaining gap |
|---|---|---|---|
| Fixed power contains removable slack | Exact same beams/masks and rate satisfaction | Direct conditional calculation | Hardware and imperfect-CSI validation |
| Joint control can improve full closed-loop energy | Fairly reoptimized controls | Not tested | Author-approved model and matched baselines |
| RL beats search | Learned nonzero policy on independent tests | Not tested | R1 remains open |

Literature boundary (abstracts checked 2026-09-22): Chien et al.,
https://arxiv.org/abs/2002.01504, already jointly consider hardware and transmit
power for energy-efficient activation in cell-free MIMO; de Souza et al.,
https://arxiv.org/abs/2102.11438, already study antenna selection/power allocation
in XL-MIMO (spectral-efficiency objective). These are related, different models,
not a novelty claim, full replication, or validation of our fixed-beam formula.
