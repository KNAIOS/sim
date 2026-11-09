"""Shared physical wideband adapter with strictly causal active-array CSI."""
import numpy as np

from joint_power.model import Environment, effective_gains, slot_energy
from joint_power.wideband_power import allocate


class WidebandEnvironment(Environment):
    def __init__(self, cfg, trajectory, physical_channel):
        physical = np.asarray(physical_channel)
        expected = (cfg.episode_steps, cfg.users, cfg.antennas)
        if (physical.ndim != 4 or physical.shape[::2] != expected[:2]
                or physical.shape[-1] != expected[-1] or physical.shape[1] % 2 != 1):
            raise ValueError('Expected time, odd frequency count, user, antenna')
        if not np.isfinite(physical).all():
            raise ValueError('Nonfinite physical channel')
        if not np.array_equal(physical[:, physical.shape[1]//2], trajectory.channel):
            raise ValueError('Control channel must equal the physical centre bin')
        if not np.all(trajectory.requirements_mbps > 0):
            raise ValueError('Failure bound requires strictly positive demand')
        super().__init__(cfg, trajectory)
        self.physical = physical
        self._phy_cache, self._cache_slot = {}, -1

    def current_phy(self, available):
        """Current observed subset only; cache shared between floor and execution."""
        self.prepare()
        available = np.asarray(available, bool)
        if available.shape != (self.cfg.subarrays,) or np.any(available & ~self.observed):
            raise ValueError('Current PHY cannot observe sleeping/waking subarrays')
        if self._cache_slot != self.t:
            self._phy_cache, self._cache_slot = {}, self.t
        key = available.tobytes()
        if key not in self._phy_cache:
            h = self.physical[self.t][:, :, np.repeat(available, self.cfg.antennas_per_subarray)]
            gains = effective_gains(self.cfg, h@h.conj().swapaxes(-2, -1))
            self._phy_cache[key] = allocate(self.cfg, gains, self.req)
        power, achieved, metadata = self._phy_cache[key]
        return power.copy(), achieved.copy(), dict(metadata)

    def step(self, command):
        self.prepare()
        cfg = self.cfg
        active, switching, _ = self.hw.apply(command, 1.)
        power, achieved, metadata = self.current_phy(active)
        components = slot_energy(cfg, int(active.sum()), int((self.hw.countdown > 0).sum()),
            float(power.sum()), switching, int(self.observed.sum()))
        outage = float(np.mean(achieved < self.req))
        bits = lambda mask: sum(int(b) << q for q, b in enumerate(mask))
        self.records.append(dict(slot=self.t, command_bits=bits(command), active_bits=bits(active),
            active_count=int(active.sum()), waking_count=int((self.hw.countdown > 0).sum()),
            observed_count=int(self.observed.sum()), outage=outage,
            numerical_fallback=int(metadata['mode'] in ('equal_fallback', 'equal_numerical_fallback')),
            allocator_mode=metadata['mode'], allocator_iterations=metadata['iterations'],
            full_target_status=metadata.get('full_target_status', 'zero_channel'),
            **dict(zip(('energy_j', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j'), map(float, components))),
            **{f'power_{k}': float(v) for k, v in enumerate(power)},
            **{f'rate_{k}': float(v) for k, v in enumerate(achieved)}))
        self.last_outage = outage
        self.hw.tick()
        self.t += 1
        self.prepared = False
        return components[0]/self.full_slot, outage


def failure_floor(env, masks, interval=4):
    """Known wideband current failures plus subsequent complete hardware blackout."""
    env.prepare()
    cfg = env.cfg
    masks = np.asarray(masks, bool)
    if cfg.wake_delay_slots <= 0 or min(cfg.demand_levels_mbps) <= 0 or interval < 1:
        raise ValueError('Positive recovery delay, user demand and interval required')
    if masks.ndim != 2 or masks.shape[1] != cfg.subarrays:
        raise ValueError('Invalid command masks')
    active = np.broadcast_to(env.hw.active, masks.shape).copy()
    countdown = np.broadcast_to(env.hw.countdown, masks.shape).copy()
    off = ~masks & (active | (countdown > 0))
    on = masks & ~active & (countdown == 0)
    active[off], countdown[off], countdown[on] = False, 0, cfg.wake_delay_slots
    lower = np.array([np.sum(env.current_phy(available)[1] < env.req)
                      for available in active], dtype=int)
    for _ in range(1, min(interval, cfg.episode_steps-env.t)):
        ready = countdown == 1
        countdown[countdown > 0] -= 1
        active[ready] = True
        lower += cfg.users*(~active.any(1))
    return lower
