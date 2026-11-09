# Reliability-price calibration after an observed objective mismatch

## Why this is a separate bounded study

The fixed-price-4 residual pilot completed and failed; it remains archived.
An additional validation-only replay now evaluates the actual gamma=0.98
finite-window reward components. The strongest nonzero feasibility-selected
checkpoint from each seed spends more energy than PRS but has a better price-4
cost. Their discounted tradeoff crossover prices versus PRS are approximately
2.107 and 1.588. Thus the previous failure is consistent with an objective that
pays for reliability below the required operating point, not merely an inability
to optimize its scalar reward. This does not prove that a new price will work.

The distinction between specifying constraints and shaping a scalar reward is
established in [Achiam et al., CPO, ICML 2017](https://proceedings.mlr.press/v70/achiam17a.html).
We are NOT implementing CPO or inheriting its guarantees. The earlier universal
value-function literature was inspected, but no new conditioned architecture is
needed for this first isolation experiment.

## Fixed protocol, before seeing new training outcomes

- Keep the exact residual architecture, 218 inputs, {-1,0,+1} legal actions,
  initialization/anchor, reward normalization, causal observations and physical model.
- Add prices 1 and 2, bracketing the observed validation crossover range; reuse
  the completed price-4 runs. No finer grid or adaptive favorable-case search.
- For each new price use both seeds 11/22 and 300 episodes, the same 971M training
  streams, 972M validation streams and 973M viewed development streams. Total
  new training is 1200 episodes, not unbounded training to a desired result.
- Retain checkpoint 0 and every 50 episodes. Use the same worst-scenario
  mean-outage feasibility (5%), then mean energy selection. Keep all checkpoints,
  prices, seeds, and failures. Compare prices on validation only.
- A selected checkpoint 0 is PRS fallback, not learning. At final price selection,
  break identical-performance ties in favor of the anchor, then lower price and
  earlier episode; no development-driven reselection.
- Run at most two single-thread workers in the existing local venv. No installs.

## Stronger fair controls

Because residual learning is allowed to reduce a count, search calibration must
not be artificially restricted to conservative margins above one. Extend both
PRS and switch-aware PRS with margins {0.90,0.95,0.98}, in addition to all 13
previous settings. Only the predictor's threshold changes; true demands and
physical rate/outage evaluation do not. Select on the same validation traces.
Report the complete 16-point calibration, including infeasible settings.

Also include a non-learning episode-level randomized mixture of the old anchor
and fixed-minus-one policy. Choose its probability solely from validation to
minimize expected energy subject to the same mean constraint in each scenario.
Expected test metrics can be integrated exactly from both component trajectories;
label them as mixture expectations, not independent simulated seeds. This tests
whether a candidate merely exploits an unused average outage allowance.

## What is and is not evidence

The audit uses normalized discounted costs in the observed 35 decision blocks;
it does not estimate the unobserved infinite tail. The paper's outage metric is
undiscounted and a worst-scenario validation gate is stricter than a single
pooled constraint. Do not conflate these three objectives or claim feasibility
from a reward increase alone.

The price study remains development-only. Promotion requires benefit over the
recalibrated strong controls at a common reliability requirement, followed by
untouched confirmation and final-model evidence. Price calibration is a training
correction, not a new communications innovation by itself.
