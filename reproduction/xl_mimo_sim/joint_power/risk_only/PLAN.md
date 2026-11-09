# Strong-reference risk-only reconstruction: bounded collection and fit

2026-09-22, before new data or model initialization. Communications;
baseline/ablation/reproducibility; overclaimed-results; planning.

## Why this is a new, bounded check

Joint-power experiments repeatedly found the learned long-energy ranking inferior
to a matched myopic-energy ablation. Their labels, however, continued with
minimum4 PRS, and the old model mixed energy and failure regression in one trunk.
The new question is whether a dedicated risk-to-go critic, referenced to the
strong minimum1 PRS, yields useful causal control. This is not a claim that
risk learning itself is novel or that a failed high-load hypothesis has passed.
Earlier fixed-power risk/Bellman experiments remain failed and are not relabeled.

## Stage A: training labels

Training stream5409220000: eight external traces per each of the unchanged three
scenarios (24 total). Prediction-validation stream5509220000: four per scenario
(12 total). Use both behaviors on each trace: Expanded PRS, and the frozen old
Energy Myopic minimum1 with Visitation-added epoch20, alpha0.1, model seed11 for
even trace indices and22 for odd. Behavior changes state coverage, not labels.
The old histogram helper is used by that behavior only, not new critic features.

At slots0/36/72/108 on both behavior trajectories, evaluate all16 causal ranking
prefixes plus the previous command. Execute the intervention for4slots, then
continue with fixed-threshold0.05 Expanded PRS (minimum1) until slot140, updating
each branch's own observations and hardware. Save energy components and failures
at4slots,16slots and full suffix. Training:192 roots; validation:96 roots.
Repeated behaviors/actions/roots are correlated, not extra independent traces.

Causal input: existing physical candidate features, module-wise post-command
hardware features, and already observed failure fraction;342 dimensions.
No true future, scenario identifier, policy identity, or outcome label is input.
The critic estimates full-suffix outcomes under the stated reference, not its
own learned future policy or a hard reliability guarantee. The feature summary
is not asserted to be a sufficient Markov state.

Keep every candidate label. Verify parent-state isolation, no future in features,
energy-component sums, duplicate commands, exact PRS suffix identity on PRS
behavior, and independent antenna-domain rates/energy for predetermined branches
(reference, counts1/8/16) plus the actual behavior episode. Reuse existing venv,
two local single-thread workers only. Do not add old test traces to training.

## Stage B: matched fitting

Two arms, each seeds11/22: risk-only and joint energy/risk. Same342->64->64 SiLU
trunk; one output for risk-only, two for joint. Trunk initial weights and batch
order are identical within each seed. Input normalization uses training only.
Risk output is3*(remaining physical slots)*sigmoid(logit), predicting the expected
number of failed user-slots in the full reference suffix. Initialize its constant
rate to the training-label frequency, clipped to[1e-4,1-1e-4].

Risk loss is capacity-weighted fractional BCE, identical in both arms. It is a
regression objective, not independent Bernoulli likelihood or calibration proof.
The joint arm adds the previous energy-difference Huber term normalized by one
full-array slot energy; energy is anchored to the strong reference candidate.
This explicitly isolates the auxiliary energy loss within the new matched pair.

Exactly80epochs, root batch16, AdamW lr0.001/wd0.0001, gradient clip5. Common
seed-specific shuffle generator independent of output-head shape. Checkpoints
0/20/40/60/80; select minimum common prediction-validation risk loss, earlier
epoch on ties. Report all checkpoints, risk MAE/bias and per-scenario prediction
metrics. Save models and exact reload checks. Do not select on a control win.

## Evidence boundaries and next gate

| Claim | This stage supplies | Still required |
|---|---|---|
| Strong-reference labels correspond to implementation | Physical replay and suffix checks | Model generalization |
| Dropping energy loss changes risk fitting | Matched heads/data/updates/selection | Closed-loop ablation |
| Risk learning saves energy | Nothing yet | New causal validation against strong searches |
| Reviewer R1/R2/R3 closed | Nothing yet | Final control evidence, sensitivity and standardized channel |

Do not use prediction loss alone as a performance gate or copy earlier controller
results onto these models. Before new closed-loop validation, freeze a separate
protocol covering common energy-based action selection, strong minimum1 and
original minimum4 PRS, identical physical information, parameter budgets, all
scenarios, and stopping. Stream5609220000 is reserved for that development stage,
not generated here and not a final independent test. The nominal primary claim
must not be silently changed to the previously favorable high scenario.
No broad training expansion, host modifications, installations, SSH, manuscript
overwrite, or publication performance claims are authorized by this fit plan.
