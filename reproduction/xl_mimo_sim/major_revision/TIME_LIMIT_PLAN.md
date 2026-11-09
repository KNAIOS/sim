# Continuing-control time-limit audit and bounded comparison

The completed 217-dimensional feature ablation does not beat PRS. Further
feature/capacity sweeps are not justified by that result. This next check instead
addresses a separate semantic issue in the training target.

## Evidence and scope

`ready_episode` and the earlier engines treat the last sample of every seven-second
trace as a terminal state (`done=True`). The network has no remaining-time input.
The paper describes continuously operating base-station activation but evaluates
finite windows. If the seven seconds are a genuine finite task, remaining time
is missing from a Markov state; if they are only sampled windows of a continuing
task, suppressing the continuation value is inappropriate. These are different
objectives, not interchangeable implementations.

This distinction is established by [Pardo et al., ICML 2018](https://proceedings.mlr.press/v80/pardo18a.html)
and explained in the [Gymnasium time-limit documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
Neither source proves a performance gain in this communications experiment.
The current main paper is not silently reinterpreted: the following is explicitly
a continuing-control candidate. Any adoption requires updating its objective
and training description, not merely substituting favorable curves.

## Prespecified test

- Compare the completed 217-dimensional terminal-window learner with a continuing
  learner; same state, price 4, architecture, seeds 11/22, 300 episodes each,
  selection, channels, power model, action clock and evaluation windows.
- Generate one additional sample for training only. All first 140 samples must
  exactly match the old trajectory with the same seed. No action uses that sample
  before its physical time; it supplies only the final transition's successor.
- At truncation, bootstrap from that actual next causal observation. Never use
  the pre-action state or a newly reset episode as its substitute.
- Unit-test the successor against the observation at the next decision of an
  uninterrupted replay with the same actions. Accounting for the first 140 slots
  must remain identical; the extra observation belongs to a potential next block.
- Keep the same training, validation and already-viewed development streams.
  The result remains exploratory; no final-test claim or new novelty claim.
- Preserve both search baselines and all seeds. No price, load or power sweep.
  Run at most two single-thread workers in the existing local venv.

The original `ready_episode` bytes are archived under `source_snapshots` using
their SHA-256 name. Its new option defaults to the old terminal behavior, and
existing trajectory/replay regression tests continue to cover that default.

Even if stability improves, independent energy/reliability confirmation is still
needed before closing reviewer Comment 1. A lower outage bought by more energy
does not pass that gate.
