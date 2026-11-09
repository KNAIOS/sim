# Post-screen baseline-grid diagnostic (not a rewritten screen)

Phase2b has FAILED: nominal RL 137.61 J exceeds PRS-power122.68 J and seed11
has5.30% high-case outage. Keep that decision and all source data unchanged.
The apparent21.27% saving versus calibrated four-slot full search needs a
grid-resolution audit because validation selected threshold0 (fallback to
minimum predicted risk), leading to roughly8 active modules.

Derive a diagnostic threshold grid exclusively from the frozen histogram:
all unique risk predictions in the ORIGINAL [0,.10] range, plus the original
{0,.02,.05,.10}. With piecewise-constant predicted risks, these are the
decision-feasibility breakpoints in that range; do not choose them using
development results. Select one threshold per full/coarse policy on the same
control-validation12 trajectories and the same worst-scenario violation,
energy, outage key. Then evaluate ONLY those selected policies on the already
used development12 trajectories. These results are retrospective development
diagnosis, not fresh confirmation. Save all thresholds and validation outcomes.

Do not alter the old baseline selection, frozen comparison table, screen,
training, or paper. Even if this diagnostic changes apparent gains, R1 stays
open because the original PRS-power and per-seed reliability failures remain.
The separate eight-slot calibrated-search requirement also remains open.
