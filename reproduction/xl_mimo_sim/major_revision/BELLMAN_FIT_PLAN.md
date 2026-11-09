# Bounded continuation-value fitting protocol

2026-09-21. Frozen before any fitting or new validation under this method.
Prerequisite: `BELLMAN_BRANCH_PLAN.md` Stage A audit passes. This protocol does
not change earlier failed gates or claim generic FQI novelty.

## Model and objective

Let x be the unchanged causal 1500-dimensional feature vector, a one of 16 prefix
counts and lambda one of {0.5,1,2,4,8,16}. One decision block is four 50-ms slots.
e(x,a) is exact block energy divided by four all-on slot energies. o_now(x,a) is
the known current-slot user outage fraction. Labels o4 are actual four-slot
user-outage means. Fit r_phi(x,a) to the remaining-three-slot fraction
(o4-o_now/4)/0.75, using binary cross entropy on fractional empirical targets.
Then o_hat=o_now/4+0.75*sigmoid(r_phi). Fractional targets are not independent
Bernoulli samples; BCE is only a regression loss, not a calibration guarantee.

Common input: existing 362 fields plus the two constant settings fields, 364.
Action input: existing 67 fields plus current three rate ratios and outage, 71.
Both encoders use Linear(364,64)+LayerNorm+SiLU and
Linear(71,32)+LayerNorm+SiLU. Immediate-risk head is Linear(96,64)+SiLU+Linear(64,1).
Continuation uses a SEPARATE encoder pair and a head
Linear(97,64)+SiLU+Linear(64,2), appending log2(lambda)/4.
The separate immediate-risk network is frozen after pretraining for EVERY
subsequent iteration and the myopic ablation. Its output cannot depend on lambda.

Depth d=1 uses Q_E=e and Q_O=o_hat, with zero continuation. At d>1,
W_d=sum(gamma^j,j=1..d-1), gamma=0.98 per decision block;
Q_E=e+W_d*softplus(v_E), Q_O=o_hat+W_d*sigmoid(v_O).
For each stored action transition to x', select a'=argmin_a[Q_E+lambda*Q_O]
using a FROZEN copy of depth d-1, then fit the two mean-continuation targets
gamma*[Q_E(x',a'),Q_O(x',a')]/W_d. No reward or outage constraint constant is
subtracted: a policy-independent constant would not affect the chosen action.
For fixed lambda this is a cost-form approximate Bellman iteration, not a proof
of constrained optimality. A lambda grid alone does not guarantee 5% outage.

The intended optimization depth is exactly 16 decision blocks (64 physical
slots), receding at deployment. It is not an infinite-horizon convergence claim.
No remaining-episode-time input or terminal zero is used. Approximation, partial
observability, limited coverage and greedy-selection bias may still cause failure.

## Frozen training budget

- Independent initializations 11 and 22; each uses its own 840-root Stage A data.
  No validation, former validation labels or branch continuation returns are used.
- Root risk: 100 epochs, shuffled root batches of 64, all 16 actions each root;
  AdamW lr=4e-4, wd=1e-5, gradient-norm clip=5. Constant output initialization
  uses only the training-label mean. No endpoint selection.
- Freeze root network. Tail initialization uses training means only.
- Depth 2 through 16, exactly five epochs per depth over all 840*6 root/price
  pairs; batch=64, all 16 actions. New AdamW at each depth with the same settings;
  warm-start weights from the previous depth. Targets computed once before each
  depth using an immutable copy. Loss is equally weighted mean squared error of
  the two normalized continuation costs. No extra rank loss or price-weighted
  sample selection. The finite price grid is training conditioning, not six
  independent random models.
- Save one final model per seed, immediate-risk-only control, training diagnostics
  and immutable hashes. Fixed depth 16 only; no best-depth selection or rerun
  until a favourable result. No host/venv dependency installation.

Checks: known energy identity, bounded outage values, strict reload equivalence,
immutable root-risk weights during tail fitting, immutable targets at each depth,
greedy selection and scalarized Bellman identity on a hand-computable toy case,
and zero continuation at depth 1. Unit/fit checks are NOT energy-performance proof.

## Evaluation boundary

This file authorizes fitting only. Before new policy validation is generated,
freeze a separate closed-loop protocol with the primary scenario, price selection,
strong search calibration budget, risk-matched comparisons, myopic ablation and
stopping gate. Do not inspect new validation and then choose these rules.
The previous policy-consistent candidate remains failed. Publication figures and
R1/R2/R3 replies remain unchanged until evidence for this new method exists.
