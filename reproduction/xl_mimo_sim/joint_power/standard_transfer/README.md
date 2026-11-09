# Current-policy wideband transfer preparation

2026-09-22. This folder does not replace the frozen geometric model, weights,
old DQN standardized-channel results, or the new sensitivity study.

R3 requires evaluating the actual new policy on standardized channels. The
existing UMi examples use 99 frequency bins; a centre-frequency failure count
is not a valid lower bound on frequency-averaged service failures. This adapter
therefore computes physical rates and the current-slot failure component using
the same per-bin RZF and shared flat-PSD power allocator.

Information contract:
- Observations arrive before switching and only on currently active subarrays.
- The existing learned features/forecast remain centre-frequency quantities.
- Per-bin active CSI is available to all compared controllers and their common
  power allocator; the adapter never exposes sleeping current CSI or future CSI
  to the current-slot failure calculation.
- Positive recovery delay prevents a newly requested sleeping subarray from
  supplying service or observations in the current slot.
- Later fully unavailable slots contribute unavoidable failures only while
  configured user demands are strictly positive. This is a lower bound, not a
  chance-constraint certificate or an upper bound on future risk.

Tests here establish numerical/causal consistency, not policy superiority.
Cached LOS/NLOS traces are previously seen numerical fixtures, not fresh test
data. A new protocol must lock channel/demand streams, fixed policies, method
information use, baselines and sample counts before any new performance run.
In particular, any comparison that adds wideband candidate screening to RL must
also include an appropriately wideband-informed search reference; do not call a
centre-only versus wideband-informed comparison a pure learning contribution.

Scope of any later UMi evaluation: standardized far-field compact-panel fading
transfer, not calibration of the 12 m near-field visibility-region process, not
measured energy, not continuous-mobility spatial-consistency validation. The
existing Sionna venv and generator may be reused without installation or host
configuration changes.
