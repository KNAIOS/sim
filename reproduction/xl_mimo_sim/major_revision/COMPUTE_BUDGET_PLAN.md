# Equal online latency budget: frozen protocol (2026-09-22)

User authorizes coarse search using count strides. No joint transmit-power
control, queue model, artificial corruption of one baseline, or predetermined
RL-superiority conclusion is authorized/inferred. Original results stay intact.

Detected axes: communications; baseline/complexity/statistical/reproducibility;
unfair-comparison/overclaimed-results; planning.

## Meaning of compute budget

Same existing CPU, single process, NumPy/BLAS and Torch configured to one thread
within the launched process only. No global environment/package/server changes.
Measure the entire decision function, including causal forecast, ranking,
state construction and inference or search. Exclude common observation updates,
channel generation, physical precoding/energy simulation, initialization and
offline training. Training is not free: its cost is outside this ONLINE budget
and is not reclassified as measured total electrical energy.

The old17.3x timing ratio was for particular Python implementations. Remove
duplicated ranking in RL and use batched Gram-domain RZF in search, preserving
their outputs. Keep legacy timing as a diagnostic, not the sole strong baseline.

The primary empirical budget B* is max over the five original RL seeds of their
calibration p95 decision time. Also publish budget slices0.25,0.5,1,2,5,10,20,200ms.
This is an online-latency comparison, not a new real deployment deadline: the
original physical decision period remains200ms. p95 eligibility is not a hard
real-time guarantee or FLOP equality. Do not fabricate missed-deadline penalties.

## Controllers, fixed before outcomes

- All five original72-input RA-DQN checkpoints11/22/33/44/55, no new training.
- Uniform count strides delta=1,2,3,4,6,12 over4..16, always include16.
  Candidate counts are13,7,5,4,3,2, respectively. delta1 is full original PRS.
- Same-candidate-budget local search: endpoints4/16 plus previous count and
  neighboring counts (lower before upper), for strides2,3,4,6. Endpoints and
  fill rules fixed in code. Avoid handicapping search by uniform grids alone.
- Full-array physical-energy reference. Legacy original PRS/RL for calibration
  timing and action equivalence only; no duplicate plotted superiority claim.
- All search candidates use all8 original forecast slots and the same exact
  RZF equations, outage threshold, ranking and active-only CSI permissions.
  Approximation comes ONLY from skipped count candidates, not added noise,
  shorter forecasts, looser physics or method-specific CSI errors.

At each budget, choose the largest candidate count passing calibration p95
within each search family, BEFORE energy evaluation. Full search remains eligible
whenever it fits. No performance-based choice of stride or selective budget plot.
Publish all raw variants as well as the budget selections, even if none favors RL.

## Disjoint data and bounded execution

- Calibration stream1600000000+j*10000+i;3 existing scenario definitions,2 traces
  each,140slots. One trace per scenario driven by original PRS, the other by
  original RL seed11.35 causal decision states per trace=210 paired states.
- Clear forecast caches before EVERY timed call; warm every path once per state.
  Three interleaved repetitions with deterministic rotating method order.
  Save every timing, including slow measurements; no outlier deletion.
- Evaluation stream1610000000+j*10000+i;8 new traces per scenario=24. Five RL
  seeds+10 search variants+full array=16 policies/trace=384 episodes. Save all
  slot masks, energy, outage and held-out decision times. No checkpoint or
  parameter selection from these outcomes. No power postprocessing this time.
- Same physical scenario definitions (.5m/s,.85load),(2m/s,1load),(3m/s,1.15load).
- Independently reproduce each selected mask using original antenna-domain RZF,
  not the optimized Gram evaluator, and replay hardware/physical accounting.
  Verify both counts and locations; recompute aggregate/paired statistics.
- Crossed5-seed/8-trajectory paired bootstrap,10000 draws, fixed seed20260924.
  Retain infeasible operating points. Report energy AND rate outage AND latency.
  Never infer equal service from sharing a5% target or trade outage for a claim
  of energy superiority. Development-sized evidence is not full confirmation.
- Stop at the bounded study/report, regardless of outcome. If a candidate looks
  promising, a frozen independent confirmation and author-approved revised claim
  are still needed before replacing manuscript claims. No automatic training.

| Claim | Required evidence | This stage | Boundary |
|---|---|---|---|
| Coarser count grids reduce measured computation | Full path timing, optimized reference | All stride/local variants | Not guaranteed by candidate count alone |
| RL improves energy under equivalent online budget | Calibrated budget, held-out latency, paired reliable energy | Bounded fair comparison | No predetermined winner or hard deadline claim |
| Full-search comparison can be discarded | Technical inapplicability | No such evidence | Full search retained |
| R1/major revision resolved | Independent advantage and consistent manuscript | Not assumed | Report actual results before any claim change |
