# Exact completed-sleep-cycle accounting audit

2026-09-22, before executing the new audit. No retraining or physical change.
Detected axes: communications / ablation,reproducibility / overclaimed-results /
audit. The preceding hold-action screen failed its prespecified promotion rule.

## Question and evidence contract

| Claim | Needed evidence | Current evidence | This audit | Boundary |
|---|---|---|---|---|
| Avoiding short cycles saves energy | Full operating/pilot/transition comparison, not switch count | More switches observed, but no cycle accounting | Closed-form single-module difference and physical replay of EVERY completed cycle | Does not establish a deployable rule |
| A cycle has a communication cost | Same channel/demand, exact RZF comparison | Not counted per cycle | Prescribed-schedule bridge replay, including adverse rate changes | Holding changes observations; no reoptimization is performed |
| RL beats search | Matched learned policies on independent trajectories | No new supporting evidence | Not tested | R1 remains open |

## Analytic object

Compare one completed active->sleep(L slots)->wake(d slots)->active cycle
with keeping that one module active over the same L+d slots. All other commands
are identical. Keep constant transmit power, fixed load/number of users, pilot
timing and the simulator's linear active-antenna processing term. Before switch
CSI pilots are observed, so L+d-1 pilots differ for d>0. For d=0, explicitly
include the immediate-wake slot's additional pilot, giving L pilot differences.
Define per-module
processing power pc=8*Nq*K^2*(B/symbols)/(compute_efficiency*1e9).

Energy(kept active)-Energy(cycled) = dt*[L*(Pa+pc-Ps)+d*(Pa+pc-Pw)]
 + (L+d-1+1[d=0])*pilot_energy - (sleep_energy+wake_energy).

This is a model identity, not a novel RL theorem or measured device behavior.
The command clock is4 and default recovery is4. Thus every complete cycle in
the present traces has L>=4; confirm that shutdown starts ready and recovery
is not aborted. Record open/censored cycles separately, never fabricate their
eventual wake or silently label them complete.

## Data and replay

Use ALL144 prescribed command schedules from the prior hold-action audit:
24 already-viewed stream121 trajectories x3 predictors x2 action sets. Do not
select only useful modules or trajectories. They are not independent test data.
Replay original schedules with the unchanged scalar HardwareState and XLSystem;
match archived per-slot energy/outage and totals. Keep commands fixed even if
the counterfactual changes observed CSI; this isolates hardware/service effects,
not a causal online policy with extra information.

For every completed cycle of every module, keep that module active until its
original ready time and preserve every other module's original schedule. Save
energy difference, original/bridge failed-user-slot counts, interval times and
state endpoints. Other modules and the post-cycle hardware state must agree.
Cycles within a trajectory overlap; do not add their service effects or treat
them as independent samples. Paired difference means refer to these diagnostic
cycles only, not whole-policy gains. No statistical significance claim.

Fast bridge accounting uses archived availability arrays and original scalar
PHY. Independently verify EVERY cycle by applying modified commands through
HardwareState from its pre-shutdown snapshot, with scalar PHY and original
pilot timing, then checking endpoint equivalence and energy identity. Synthetic
tests cover different durations/delays, aborted recovery rejection, and endpoint
censoring. Derivative and break-even expressions are checked algebraically and
against scalar simulation; no physical parameter sweep for improved performance.

At most two one-thread workers; existing project venv. Preserve all old sources,
figures, checkpoints and PDFs. Archive source/input/output hashes. No new RL,
new standardized-channel run, or new causal anti-cycling policy is authorized
by an energy-accounting result alone. If all realizable minimum-length cycles
already save energy versus staying active, do not continue claiming switch
suppression alone is an energy-saving mechanism under this model. This says
nothing about globally optimal service-constrained control or real hardware.

## External validity check

Ericsson's 2023 first-party energy-model explanation distinguishes whole-gNB
micro/light/deep sleep, relative power and total transition time; it does not
calibrate our14GHz subarray's absolute2/1.2/.05W powers or200ms recovery. Source:
https://www.ericsson.com/en/blog/2023/8/breaking-the-energy-curve
Do not inflate wake energy or replace these parameters by whole-BS numbers to
manufacture a favorable RL regime. Measured subarray calibration remains absent.
