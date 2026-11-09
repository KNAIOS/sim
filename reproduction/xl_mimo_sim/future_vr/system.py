from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class HardwareState:
    cfg: Config

    def __post_init__(self) -> None:
        self.active = np.ones(self.cfg.subarrays, dtype=bool)
        self.countdown = np.zeros(self.cfg.subarrays, dtype=int)
        self.command = np.ones(self.cfg.subarrays, dtype=bool)

    def readiness(self) -> np.ndarray:
        return np.where(self.active, 0, np.where(self.countdown > 0, self.countdown, self.cfg.wake_delay_slots))

    def apply(self, command: np.ndarray, switch_scale: float) -> tuple[np.ndarray, float, int]:
        command = command.astype(bool)
        newly_off = (~command) & (self.active | (self.countdown > 0))
        newly_on = command & (~self.active) & (self.countdown == 0)
        self.active[newly_off] = False
        self.countdown[newly_off] = 0
        self.countdown[newly_on] = self.cfg.wake_delay_slots
        if self.cfg.wake_delay_slots == 0:
            self.active[newly_on] = True
        self.command = command.copy()
        transitions = int(newly_off.sum() + newly_on.sum())
        energy = switch_scale * (
            self.cfg.sleep_energy_j * newly_off.sum() + self.cfg.wake_energy_j * newly_on.sum()
        )
        return self.active.copy(), float(energy), transitions

    def tick(self) -> None:
        completing = self.countdown == 1
        self.countdown[self.countdown > 0] -= 1
        self.active[completing] = True


@dataclass(frozen=True)
class StepResult:
    rates_mbps: np.ndarray
    energy_j: float
    outage: float
    shortfall: float
    delivered_mbit: float
    power_w: float


class XLSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        thermal_dbm = -174.0 + 10.0 * np.log10(cfg.bandwidth_mhz * 1e6) + cfg.noise_figure_db
        self.noise_w = 10.0 ** ((thermal_dbm - 30.0) / 10.0)

    def achievable_rates(self, channel: np.ndarray, available: np.ndarray) -> np.ndarray:
        """Net achievable rate after a fixed pilot overhead, not packet goodput."""
        indices = np.flatnonzero(np.repeat(available, self.cfg.antennas_per_subarray))
        if len(indices) < self.cfg.users:
            return np.zeros(self.cfg.users)
        h = channel[:, indices]
        gram = h @ h.conj().T
        regularizer = self.cfg.users * self.noise_w / max(self.cfg.tx_power_w, 1e-12)
        precoder = h.conj().T @ np.linalg.inv(gram + regularizer * np.eye(self.cfg.users))
        precoder /= np.maximum(np.linalg.norm(precoder, axis=0, keepdims=True), 1e-15)
        effective = h @ precoder
        per_user_power = self.cfg.tx_power_w / self.cfg.users
        desired = per_user_power * np.abs(np.diag(effective)) ** 2
        interference = per_user_power * (np.sum(np.abs(effective) ** 2, axis=1) - np.abs(np.diag(effective)) ** 2)
        sinr = desired / np.maximum(interference + self.noise_w, 1e-30)
        return self.cfg.bandwidth_mhz * (1.0 - self.cfg.pilot_fraction) * np.log2(1.0 + sinr)

    def evaluate(
        self,
        channel: np.ndarray,
        available: np.ndarray,
        waking: np.ndarray,
        transition_energy_j: float,
        scan: bool,
        requirements_mbps: np.ndarray,
        observed_count: int | None = None,
    ) -> StepResult:
        rates = self.achievable_rates(channel, available)

        active_count = int(available.sum())
        waking_count = int(waking.sum())
        active_antennas = active_count * self.cfg.antennas_per_subarray
        flops = 8.0 * active_antennas * self.cfg.users**2 + (8.0 / 3.0) * self.cfg.users**3
        blocks_s = self.cfg.bandwidth_mhz * 1e6 / self.cfg.symbols_per_processing_block
        processing_w = flops * blocks_s / (self.cfg.compute_efficiency_gflop_w * 1e9)
        power_w = (
            self.cfg.fixed_power_w
            + self.cfg.tx_power_w / self.cfg.pa_efficiency
            + active_count * self.cfg.active_subarray_w
            + waking_count * self.cfg.waking_subarray_w
            + (self.cfg.subarrays - active_count - waking_count) * self.cfg.sleeping_subarray_w
            + processing_w
        )
        # Pilots are acquired before switching and only from active branches.
        # The initially active full array provides the common initial observation.
        if scan:
            raise ValueError("Instantaneous sleeping-chain scans are not supported")
        pilot_count = active_count if observed_count is None else observed_count
        scan_energy = self.cfg.scan_energy_j_per_subarray * pilot_count
        energy = power_w * self.cfg.slot_s + transition_energy_j + scan_energy
        outage = float(np.mean(rates < requirements_mbps))
        shortfall = float(np.mean(np.maximum(requirements_mbps - rates, 0.0) / np.maximum(requirements_mbps, 1e-9)))
        return StepResult(rates, float(energy), outage, shortfall, float(rates.sum() * self.cfg.slot_s), float(power_w))
