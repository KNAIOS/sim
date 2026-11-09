# Full-count spatial-action audit (no learner or model retuning)

2026-09-21. Frozen before outcomes. Previous candidates and gates remain stopped.
Axes: communications; baseline/ablation/reproducibility; overclaimed-results; audit.

## Why this is not the old experiment again

`SPATIAL_ACTION_REPORT.md` tested local same-count retention/exchange at a 4--16
count floor, old PRS continuation and a different predictive selection rule.
Later corrections established that the physical full-digital model permits
1--16 subarrays. Latest count and margin learners still use a single fixed
ranking prefix per count. Whether useful arbitrary-mask choices remain under
this full-count domain has not been measured exhaustively.

| Question | Existing evidence | This audit | Not established |
|---|---|---|---|
| Does a fixed ranking omit lower predicted-cost spatial masks? | Earlier local old-domain search changes few actions | Enumerate every nonempty 16-bit mask at fixed causal roots | True optimal energy or learning benefit |
| Does a single exchange capture those alternatives? | Old 4--16 results only | Restrict the SAME candidate table to one-exchange masks | General action-space sufficiency |
| Does the causal choice survive actual channel evolution? | No current-domain all-mask labels | Hold-command and first-block intervention replays, all outcomes retained | Independent whole-policy superiority |

Antenna selection for XL-MIMO energy efficiency already exists (Marinello et al.,
https://arxiv.org/abs/2009.02542). Incomplete-CSI tracking/risk-aware selection also
exists (He et al., TWC2024, https://ieeexplore.ieee.org/document/10478288/).
This enumeration is not either published algorithm's reproduction or a new
communication contribution. It is a computationally costly diagnostic control.

## Frozen roots and common information

- Use already-viewed validation stream113, first TWO trajectories per scenario,
  six shared external trajectories total. Fixed roots t=0,32,64,96:24 roots.
  They are correlated development states, not24 independent tests.
- Roll in and continue with the unchanged full-count cost search (minimum1).
  Margins fixed at1.00/1.08/1.04 for low/nominal/high, respectively. These are
  the previous validation-selected pure-search margins; high's selected floor4
  is NOT inherited. The baseline here is the common-domain cost rule, not the
  strongest calibrated policy or a new comparison claiming superiority.
- Before branching, verify every baseline episode against its exact matching
  cost/minimum1/margin record in the existing complete113 table.
- Preserve current model, observations, actual requirements, hardware/pilot
  accounting, four-slot decisions, four-slot wake delay and eight-slot forecast.
  Decisions receive only the already available tracker and hardware state.

## Exhaustive causal choices

At each root evaluate all65535 nonempty masks held for8 slots, using the SAME
filtered-current / forecast Gram matrices and hardware transitions as the
existing cost search. All masks start from the SAME hardware/CSI state. No true
future is available to this selection. Store predicted energy, outage and
normalized shortfall for every mask, along with root causal inputs and masks.

Select three alternatives to the existing prefix reference:
1. all same-count masks;
2. same-count masks differing by one selected/unselected exchange (plus reference);
3. all nonempty masks, jointly allowing count and location changes.

Use the common lexicographic key: infeasible(predicted outage>5%), then energy
if feasible or shortfall if infeasible, then energy. Resolve identical keys by
minimum Hamming distance to the reference and then ascending bitmask integer.
Including the reference ensures no worse PREDICTED primary key, not realized
performance. This tie rule is declared before seeing outcomes.

Enumerate in bounded batches; preserve exact per-column-normalized RZF algebra.
Test arbitrary-mask predictions against the original scalar hardware/RZF model,
and all16 prefix entries against frozen `hold_predictions`. Empty mask is not
an action. This is not dynamic programming or optimal control over future masks.

## Actual outcomes and interpretation

Cache all selected commands before consulting any physical future outcomes.
For every distinct selected mask, including the reference:
- hold it for8 true slots, record physical energy/outage/shortfall and compare
  predicted hardware energy (must agree up to roundoff);
- apply it for the first4 true slots, then return to the SAME causal full-count
  cost policy until the original episode end. Compare total suffix energy and
  user-slot outage against the reference branch from the identical root.

Reference suffix must reproduce the original baseline's logged energy/outage.
Deduplicate identical masks, but retain all four labeled choices in the table.
Report all24 roots and every no-op, not only favorable changes. Report same-count
and unrestricted results separately. Do not add individual intervention gains
together, multiply by decision count, or call them achievable whole-policy gain.

There is no new training, margin sweep, full-episode all-mask deployment or
independent confirmation in this protocol. Any further learner needs its own
justification/protocol plus matched-action nonlearning baselines. Neither a
positive sparse intervention nor a negative result can decide all spatial RL.
No manuscript result substitution or R1 closure is authorized by this diagnostic.

## Verification and resource boundary

Independent scalar replay of all chosen masks, exact selection reconstruction
from saved all-mask costs, mask coverage/uniqueness, prefix equality, baseline
episode equality, and immutable root inputs/source hashes. Two workers at most,
each one existing-venv CPU thread, split by trajectory index. No installs or
host/server environment changes. All previous files remain untouched.
