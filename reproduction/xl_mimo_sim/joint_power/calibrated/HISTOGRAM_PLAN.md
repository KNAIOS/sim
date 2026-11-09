# Phase 2b: lower-capacity shared risk calibration

Written after phase2 neural calibration failed, before this candidate is fit.
Do not change or remove the phase2 failure. Its best checkpoint was epoch0;
no RL control training or development evaluation was launched under that gate.

The high-dimensional 64/64 residual network generalized worse after training.
Keep the accurate analytic immediate energy forecast unchanged. Replace ONLY
the shared risk predictor by a shrunk histogram, using the SAME 48 training
trajectories already collected. Do not use the 12 previous validation
trajectories as additional training data. New calibration-validation stream
3009220000, four trajectories/scenario (12 total), not previously generated.

Groups: predicted four-slot user-outage bin (nearest multiple of1/12),
command count {1},{2..3},{4..7},{8..16}, and whether any requested branch is
already waking or a new wake is commanded. Fit 104 group means, shrunk toward
the corresponding global predicted-outage-bin mean. The global bin mean is
shrunk with32 interval-equivalents toward clipped analytic rate [.001,.999].
Test local shrink strengths {32,128,512}, choose by held-out Brier score then
BCE. This is regularized fractional-outcome regression, not independent
Bernoulli observations or a Bayesian confidence bound. Missing groups use
the global bin prediction. No hand-adjusted per-method risks or probabilities.

Reuse training-feature mean/std for RL features. Preserve the Predictor API,
but runtime risk uses only the named group features and a frozen lookup table.
Calibration gate: selected prediction Brier must be below BOTH raw analytic
and the old .01/.99-clipped analytic baseline on this NEW validation split;
absolute mean risk error <=.05; finite output. If it fails, stop this candidate.
Report trajectory bootstrap and policy-relevant group bins; the small-sample
gate is not a reliability certificate or proof of calibration at every state.

If gate passes, use exactly the RL algorithm, grids, baseline selection,
160-episode two-seed budget, control-validation/development streams and
advantage gate from PLAN.md. Those downstream streams have not been used.
Use a new immutable results/joint_power_20260922_phase2b folder. Its launcher
explicitly binds the reusable study driver to this folder and histogram loader;
it does not edit frozen phase-one/phase-two source or results. No host changes.

The claim-evidence matrix from PLAN.md remains applicable. Current R1 still
open, and no paper result replacement is authorized by a calibration gate.
