"""Causal, paid CSI probing and a four-slot information-withholding diagnostic."""
import numpy as np

from major_revision.action_value import baseline_command
from major_revision.full_domain import FullDomainRollout


class ProbeRollout(FullDomainRollout):
    """Observe/charge physical active branches; optionally withhold one from control."""
    def __init__(self, cfg, tr, minimum=1):
        super().__init__(cfg, tr, minimum)
        self.withheld_q = -1
        self.withheld_start = self.withheld_until = -1

    @classmethod
    def from_root(cls, root, withheld_q=-1, start=-1, until=-1):
        run = cls.__new__(cls)
        run.__dict__ = root.clone().__dict__
        run.withheld_q = int(withheld_q)
        run.withheld_start, run.withheld_until = int(start), int(until)
        return run

    def prepare(self):
        if self.t >= self.cfg.episode_steps:
            raise ValueError('Episode is complete')
        if self.prepared:
            return
        self.observed = self.hw.active.copy()  # Actual paid pilots, not assimilated ones.
        assimilated = self.observed.copy()
        if self.withheld_q >= 0 and self.withheld_start <= self.t < self.withheld_until:
            assimilated[self.withheld_q] = False
        self.age = np.where(assimilated, 0., self.age + 1)
        self.tracker.update(self.tr.gain_db[self.t], assimilated)
        antennas = np.repeat(assimilated, self.cfg.antennas_per_subarray)
        self.tracker.channel_estimate[:, antennas] = self.tr.channel[self.t][:, antennas]
        self.req = self.tr.requirements_mbps[self.t]
        self.prepared = True


def probe_choices(root, reference):
    """Only causal observation/state; return all candidates and two frozen choices."""
    _, score = root.base_observation()
    eligible = np.flatnonzero(~root.hw.active & (root.hw.countdown == 0) & ~reference)
    if not len(eligible):
        return eligible, dict(score=-1, age=-1)
    by_score = min(eligible, key=lambda q: (-float(score[q]), int(q)))
    by_age = min(eligible, key=lambda q: (-float(root.age[q]), -float(score[q]), int(q)))
    return eligible, dict(score=int(by_score), age=int(by_age))


def probe_command(root, reference, q):
    mask = np.asarray(reference, dtype=bool).copy()
    if q >= 0:
        if mask[q] or root.hw.active[q] or root.hw.countdown[q] > 0:
            raise ValueError('Probe must be an unselected fully sleeping module')
        mask[q] = True
    return mask


def branch(root, reference, q, margin, minimum, withhold=False, trace=False):
    """One root intervention then the same reference controller, through episode end."""
    if root.t % 4 or root.t + root.cfg.wake_delay_slots + 4 > root.cfg.episode_steps:
        raise ValueError('A complete probe window at a decision epoch is required')
    first = probe_command(root, reference, q)
    start = root.t
    ready = start + root.cfg.wake_delay_slots
    run = ProbeRollout.from_root(root, q if withhold else -1, ready, ready + 4)
    records, snapshots = [], []
    first_block_energy = 0.
    while run.t < run.cfg.episode_steps:
        run.prepare()
        if run.t % 4 == 0:
            command = first if run.t == start else baseline_command(run, 'cost', margin, minimum)
        else:
            command = None
        if trace:
            snapshots.append((run.t, run.hw.active.copy(), run.hw.countdown.copy(),
                              run.tracker.channel_estimate.copy(), run.age.copy()))
        before = run.energy
        run.step(command)
        if run.t <= start + 4:
            first_block_energy += run.energy - before
        if trace:
            records.append((run.t - 1, run.energy - before, run.outages[-1],
                            run.pilot, run.switching, run.hw.command.copy()))
    result = run.metrics()
    result.pop('method')
    result.update(first_block_energy_j=first_block_energy,
                  first_block_outage=float(np.mean(run.outages[start:start+4])),
                  suffix_energy_j=run.energy-root.energy)
    return (result, records, snapshots) if trace else result


def independent_reference(root, reference, margin, minimum):
    """No new adapter: use the original rollout for baseline parity."""
    run = root.clone()
    start = run.t
    while run.t < run.cfg.episode_steps:
        command = None
        if run.t % 4 == 0:
            command = reference if run.t == start else baseline_command(run, 'cost', margin, minimum)
        run.step(command)
    return run.metrics()


def causal_snapshot(root, reference, choices):
    _, score = root.base_observation()
    return dict(reference=reference, score=score, active=root.hw.active,
                countdown=root.hw.countdown, command=root.hw.command, age=root.age,
                requirements=root.req, channel_estimate=root.tracker.channel_estimate,
                gain_db=root.tracker.gain_db, velocity=root.tracker.velocity,
                uncertainty=root.tracker.uncertainty, probability=root.tracker.probability,
                score_choice=np.array(choices['score']), age_choice=np.array(choices['age']))
