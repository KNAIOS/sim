# Baseline implementation correction before cost results

The initial `full_domain_study --part baselines` process was interrupted with
Ctrl-C (exit 1) while still evaluating PRS validation margins. Its partial raw
records are preserved in `results/major_revision_20260920_full_domain/baselines`.
There is no completed cost arm, selected margin file, or development result
there. This is a code-correctness correction, not selection after cost outcomes.

On code review, the new adapter's cost-search forecast for the current slot used
the raw last-observed complex channel. The frozen original `pilot.cost_search`
rescales that channel to current **filtered** tracker gain, which can differ
from the last-observed magnitude. Short action-replay unit tests passed but did
not prove algebraic equivalence of that forecast.

The exact implementation is now `full_domain_baselines.baseline_episode`, with
the historical filtered-gain scaling and only the minimum count changed. It
writes `baselines_v2`; all original files remain intact and all validation and
development sampling, margins, selection and metrics stay as preregistered.
The full-domain DQN does not call either search implementation, so its training
and its source hashes are unaffected. Do not cite the initial adapter's cost
branch as an exact reproduction or use its incomplete results in a comparison.
