# Post-hoc validation-price boundary check (not a changed original gate)

2026-09-22, after absolute-critic control-validation, before these extra
evaluations. Both absolute critics selected the lower boundary lambda4;
worst-scenario mean outages3.39%/4.35% remain below5%, while nominal energies
133.69/149.19J exceedPRS119.98J. Increasinglambda substantially raised energy.
The original gate remains failed and its files are immutable.

Test the missing lower side on the SAME tuning/validation split3909220000:
lambda{0,.25,.5,1,2}, both fixed signed and absolute critics, both seeds.
No retraining or new checkpoint selection. This is explicit post-hoc parameter
tuning, NOT an independent confirmation. Keep all old options{4,8,16,32} and
all baseline tuning records; select by the unchanged actual5% constraint and
energy key. No baseline capability is changed or result suppressed.

Zero gives an energy-only limiting case, not a reliability-feasible solution
by definition. Do not assume outage is monotone inlambda in closed loop.
Save all240 additional validation episodes in a separate price_extension
folder. Report the original failed screen as well as the extended screen.

The same1% mean nominal advantage/no-seed-regression screen applies. Only if
passed may a separately frozen new development run be launched; do not call
the old control.py development path with overwritten original choices. If
failed, stop this bounded Monte Carlo candidate and diagnose the remaining
model/data issue rather than continuing to add prices until a chance win.
