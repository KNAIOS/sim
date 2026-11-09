"""Snapshot-capable, evaluation-only replay for single-action interventions.

This adapter never changes the physical model or frozen training sources.
Regression tests compare observations and all accounting with ready_episode.
"""
import copy
import numpy as np
from future_vr.belief import VisibilityBelief
from future_vr.controllers import select_top
from future_vr.system import HardwareState, XLSystem
from major_revision.pilot import state
from major_revision.rate_context import candidate_rate_context
from major_revision.ready_rate_context import ready_rate_context


class AuditRollout:
    def __init__(self, cfg, trajectory):
        self.cfg, self.tr = cfg, trajectory
        self.system, self.hw = XLSystem(cfg), HardwareState(cfg)
        self.tracker = VisibilityBelief(cfg, trajectory.gain_db[0])
        self.tracker.channel_estimate = trajectory.channel[0].copy()
        self.age = np.zeros(cfg.subarrays)
        self.last_off = np.full(cfg.subarrays, -10000)
        self.previous_req = trajectory.requirements_mbps[0].copy()
        self.last_outage = 0.
        self.t, self.prepared = 0, False
        self.energy = self.switching = self.pilot = 0.
        self.transitions = self.churn = 0
        self.outages, self.counts = [], []
        self.slot_energy = []

    def clone(self):
        # Share only immutable-in-use exogenous inputs; all policy-dependent
        # tracker/hardware arrays and accounting histories remain independent.
        return copy.deepcopy(self, {id(self.tr):self.tr, id(self.cfg):self.cfg})

    def prepare(self):
        if self.t >= self.cfg.episode_steps:
            raise ValueError('Episode is complete')
        if self.prepared:
            return
        self.observed = self.hw.active.copy()
        self.age = np.where(self.observed, 0., self.age+1)
        self.tracker.update(self.tr.gain_db[self.t], self.observed)
        antennas = np.repeat(self.observed, self.cfg.antennas_per_subarray)
        self.tracker.channel_estimate[:,antennas] = self.tr.channel[self.t][:,antennas]
        self.req = self.tr.requirements_mbps[self.t]
        self.prepared = True

    def observation(self):
        self.prepare()
        if self.t % 4:
            raise ValueError('Observation requested outside a decision epoch')
        base, score = state(self.cfg, self.tracker, self.hw, self.req,
                            self.previous_req, self.age, self.last_outage, True)
        obs = np.concatenate((base,
            candidate_rate_context(self.cfg,self.tracker,score,self.req),
            ready_rate_context(self.cfg,self.tracker,self.hw,score,self.req))).astype(np.float32)
        return obs, score

    def step(self, command=None):
        self.prepare()
        if self.t % 4 == 0:
            if command is None:
                raise ValueError('A decision command is required')
            self.previous_req = self.req.copy()
        else:
            if command is not None and not np.array_equal(command,self.hw.command):
                raise ValueError('Command cannot change between decision epochs')
            command = self.hw.command.copy()
        changed_on = command & ~self.hw.command
        changed_off = ~command & self.hw.command
        self.churn += int(np.sum(changed_on & ((self.t-self.last_off) <= self.cfg.horizon)))
        self.last_off[changed_off] = self.t
        active, switch, changes = self.hw.apply(command, 1.)
        result = self.system.evaluate(self.tr.channel[self.t], active,
            self.hw.countdown>0, switch, False, self.req, observed_count=int(self.observed.sum()))
        self.hw.tick()
        self.energy += result.energy_j
        self.switching += switch
        self.pilot += self.cfg.scan_energy_j_per_subarray*self.observed.sum()
        self.transitions += changes
        self.outages.append(result.outage)
        self.counts.append(command.sum())
        self.slot_energy.append(result.energy_j)
        self.last_outage = result.outage
        self.t += 1
        self.prepared = False

    def finish(self, controller, first_command=None):
        first = True
        while self.t < self.cfg.episode_steps:
            if self.t % 4 == 0:
                obs, score = self.observation()
                if first and first_command is not None:
                    command = first_command.copy()
                else:
                    _, count = controller.choose(obs, 0.)
                    command = select_top(score,count)
                self.step(command)
            else:
                if first and first_command is not None:
                    raise ValueError('Intervene only at a decision epoch')
                self.step()
            first = False
        return self.metrics()

    def metrics(self):
        return dict(method='Enriched-DQN', energy_j=self.energy,
            outage=float(np.mean(self.outages)), switching_j=self.switching,
            pilot_j=float(self.pilot), operating_j=self.energy-self.switching-self.pilot,
            transitions=self.transitions, short_sleep_reactivations=self.churn,
            mean_count=float(np.mean(self.counts)))
