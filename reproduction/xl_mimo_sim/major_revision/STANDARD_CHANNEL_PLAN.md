# Standard-channel validation gate (2026-09-20)

Status: source verification only; no standardized-channel results yet.

## Verified implementation and installation boundary

- NVIDIA Sionna 2.1.0 provides TR 38.901 models. Its official installation
  instructions require Python >=3.11 and recommend PyTorch >=2.9.
- The restored server package uses Python 3.10 / PyTorch 2.6. Do NOT upgrade
  its interpreter or overwrite that environment. Use a second independent
  project-local environment, potentially on the local Python 3.12 machine,
  and export immutable channel traces for the server experiment runner.
- Use `sionna-no-rt`, not the ray-tracing package: no CUDA, LLVM, driver,
  shell-profile, system-Python, or OS installation is required by this plan.
- All dependencies and caches must remain in the new package directory.

## Important scientific limitation found in official documentation

The current spatial-consistency implementation generates jointly correlated
random fields for terminal positions in a topology snapshot. It explicitly
does NOT implement the stateful mobility procedure of Section 7.6.3.2.
Velocities give Doppler evolution within a generated impulse response, but
do not move topology, evolve LOS states, or implement cluster birth/death.
Repeated `set_topology` calls must not be misrepresented as a physically
continuous realization. Track IDs do not preserve fields across those calls.

Therefore a standard-channel test can validate performance under the declared
TR 38.901 fading scenario, but cannot by itself validate the original moving
near-field visibility-region mechanism. Keep these claims separate.

## Minimum next evidence

1. Pin the actual installed package version and inspect its API/source.
2. Select and disclose a supported UMi or CDL scenario, antenna geometry,
   carrier, velocities, bandwidth, sampling interval, delay profile, pathloss
   and shadowing settings. Never silently normalize away pathloss.
3. Validate dimensions, units, deterministic seed reproduction, temporal
   correlation/Doppler behavior and relevant profile statistics. Distinguish
   implementation checks from a complete 3GPP calibration exercise.
4. Retain common traces, power accounting, active-only observations and
   reliability evaluation across all policies. Declare flat-fading versus
   frequency-selective approximations; a central-subcarrier test alone is
   not wideband OFDM validation.
5. Evaluate frozen geometric-channel policies as transfer evidence; if
   retraining is necessary, report it separately with equal training and
   validation budgets. Retain adverse results.

## Official sources checked

- https://nvlabs.github.io/sionna/installation.html
- https://nvlabs.github.io/sionna/phy/api/channel/wireless/tr38901_spatial_consistency.html
- https://pypi.org/project/sionna/
- https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.00.00_60/tr_138901v180000p.pdf

Sionna current documentation references parameter editions other than the
manuscript's V18.0.0. Record the actual supported edition rather than claiming
an unverified version match.
