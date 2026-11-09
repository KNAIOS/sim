# Demonstration-initialization diagnostic (fixed before execution)

Detected axes: task_type=communications; evidence_type=baseline, ablation,
statistical, reproducibility; failure_mode=unfair-comparison, weak-ablation,
overclaimed-results; stage=planning.

Previous turn classification: progress. It completed a bounded full-domain
study, rejected all candidates, and identified the near-binding equal-power
rate limit on high-load development traces. It did not resolve R1.

## Hypothesis and minimum evidence

The fresh DQN commands 5.79 modules where causal search commands 1.39 at low
load. Before another RL sweep, test whether the same network/observation can
learn the existing causal count map from explicit action labels, including
states reached by its own actions. Successful imitation is **not** an RL gain
or a new communications contribution. Failure changes what must be debugged.

| Claim | Test | Controls / scope | Gate |
|---|---|---|---|
| The 235-feature network can represent useful count decisions | Causal expert labels, held-out action agreement, closed-loop energy/outage | Same architecture, domain, ranking, physical model | Listed below |
| Own-state exposure matters | Teacher-only behavioral cloning vs two aggregation rounds | Same data source and fixed optimizer; disclose extra examples | Descriptive, no automatic novelty claim |
| RL adds something beyond imitation | Only after the initialization gate; retained imitation model vs fine-tuned model and expert | Separate prespecified fine-tuning protocol, untouched confirmation | Not authorized by an imitation result alone |

## Strengthen expert first

Retain full-domain PRS and cost-search records. Add minimum-four PRS calibration
on the **same** 982000000 validation stream, 12 traces/scenario and the same six
margins (.9,1,1.04,1.08,1.16,1.24). Select minimum 1 or 4 plus margin per scenario
by mean outage <=.05 then least energy; otherwise least outage then energy.
The original minimum-four policy is admissible inside the common 1--16 domain.
No development result selects the teacher. This is a strengthened constructed
baseline, not reproduction of a published state-of-the-art controller.

## Bounded imitation stage

- Same 235-input, 64/64 hidden, 16-output network; outputs here are action
  classification logits, **not** learned Q-values. AdamW 4e-4, weight decay
  1e-5, cross entropy, batch 256, gradient clipping at 5.
- Two network seeds 11,22. Teacher demonstrations: 32 trajectories/scenario,
  stream 991000000+j*10000+i = 96 trajectories and 3360 decision labels.
- Held-out imitation validation: 12 trajectories/scenario on
  992000000+j*10000+i; never added to training. Save every causal teacher state
  and label; also evaluate the student's actual closed loop on these traces.
- 100 teacher-only epochs. Then **two** fixed aggregation rounds; each rolls
  out the student on 12 new training trajectories/scenario, queries teacher
  labels at student-visited states, appends them and fits another 50 epochs.
  Streams 994000000+seed*100000+round*10000+j*1000+i (round 1,2).
  Teacher labels are queried in training only, not used by deployed decisions.
- Save the three stage endpoints, all losses and action-agreement statistics,
  plus all validation closed-loop rows. Select using maximum scenario-mean
  outage excess above the expert (tolerance .01), then mean energy. Persist
  the choice before inspecting development trajectories.
- New development stream 993000000+j*10000+i, 16/scenario, evaluates the selected
  imitator, the teacher-only endpoint, the expert and full-array reference.
  All are development diagnostics, not independent final confirmation.
- Gate for each seed: >=90% held-out expert-state action agreement; <=1 percentage
  point excess mean outage over the expert in **each** scenario; and <=10%
  excess energy in **each** scenario on validation. The two seeds must both
  pass before a separately frozen RL fine-tuning experiment is justified.
  Mean outage agreement is not a statistical reliability certificate.
- Do not extend the count of aggregation rounds or epochs until this gate
  passes. Retain failures. No changed power constants, channel simplification,
  trajectory exclusions or new global/system packages. At most two workers.

## Literature basis and boundary

[Hester et al., DQfD, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11757)
combines TD learning and supervised action information; ordinary classification
pretraining alone is not a calibrated Q-function or a reproduction of DQfD.
[Ross et al., DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)
motivates labeling states induced by the learned policy. We use a finite
data-aggregation diagnostic, not its theoretical guarantee. These are existing
training ideas; any communications contribution still needs separate evidence.
