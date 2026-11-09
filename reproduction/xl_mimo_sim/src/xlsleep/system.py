from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass(frozen=True)
class LinkResult:
    rates_mbps: np.ndarray
    sinr: np.ndarray
    power_w: float
    energy_j: float
    outage: float
    energy_efficiency_mbit_j: float


class XLSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        thermal_dbm = -174.0 + 10.0 * np.log10(cfg.bandwidth_mhz * 1e6) + cfg.noise_figure_db
        self.noise_w = 10.0 ** ((thermal_dbm - 30.0) / 10.0)

    def antenna_indices(self, mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.repeat(mask.astype(bool), self.cfg.antennas_per_subarray))

    def evaluate(
        self,
        channel: np.ndarray,
        mask: np.ndarray,
        previous_mask: np.ndarray,
        switch_scale: float,
        scan: bool = False,
        csi_seed: int = 0,
        rate_requirement_mbps: float | np.ndarray | None = None,
    ) -> LinkResult:
        indices = self.antenna_indices(mask)
        if len(indices) < self.cfg.users:
            rates = np.zeros(self.cfg.users)
            return self._finish(rates, np.zeros_like(rates), mask, previous_mask, switch_scale, scan, rate_requirement_mbps)
        true_h = channel[:, indices]
        rng = np.random.default_rng(csi_seed)
        pilot_snr = self.cfg.tx_power_w * np.mean(np.abs(true_h) ** 2) / max(self.noise_w, 1e-30)
        error_ratio = 1.0 / np.sqrt(max(pilot_snr * self.cfg.pilot_symbols, 1.0))
        error_std = np.sqrt(np.mean(np.abs(true_h) ** 2)) * min(error_ratio, 0.25)
        # Draw a full-array innovation and then select antennas.  This gives
        # competing policies common random numbers on a paired trajectory.
        innovation = rng.normal(size=channel.shape) + 1j * rng.normal(size=channel.shape)
        error = error_std / np.sqrt(2.0) * innovation[:, indices]
        estimated = true_h + error
        gram = estimated @ estimated.conj().T
        regularization = self.cfg.users * self.noise_w / max(self.cfg.tx_power_w, 1e-12)
        precoder = estimated.conj().T @ np.linalg.inv(gram + regularization * np.eye(self.cfg.users))
        precoder /= np.maximum(np.linalg.norm(precoder, axis=0, keepdims=True), 1e-15)
        effective = true_h @ precoder
        per_user_power = self.cfg.tx_power_w / self.cfg.users
        desired = per_user_power * np.abs(np.diag(effective)) ** 2
        interference = per_user_power * (np.sum(np.abs(effective) ** 2, axis=1) - np.abs(np.diag(effective)) ** 2)
        sinr = desired / np.maximum(interference + self.noise_w, 1e-30)
        prelog = 1.0 - self.cfg.pilot_symbols / self.cfg.coherence_symbols
        rates = self.cfg.bandwidth_mhz * prelog * np.log2(1.0 + sinr)
        return self._finish(rates, sinr, mask, previous_mask, switch_scale, scan, rate_requirement_mbps)

    def _finish(
        self,
        rates: np.ndarray,
        sinr: np.ndarray,
        mask: np.ndarray,
        previous_mask: np.ndarray,
        switch_scale: float,
        scan: bool,
        rate_requirement_mbps: float | np.ndarray | None,
    ) -> LinkResult:
        active_antennas = int(mask.sum()) * self.cfg.antennas_per_subarray
        q = max(int(mask.sum()), 1)
        k = self.cfg.users
        # RZF formation and data multiplication, converted with the cited 75 Gflop/s/W efficiency.
        flops_per_block = 8.0 * active_antennas * k * k + (8.0 / 3.0) * k**3
        blocks_per_second = self.cfg.bandwidth_mhz * 1e6 / self.cfg.coherence_symbols
        processing_w = flops_per_block * blocks_per_second / (self.cfg.compute_efficiency_gflop_per_w * 1e9)
        power_w = (
            self.cfg.fixed_power_w
            + self.cfg.local_oscillator_w
            + active_antennas * self.cfg.antenna_circuit_w
            + k * self.cfg.user_circuit_w
            + self.cfg.tx_power_w / self.cfg.pa_efficiency
            + processing_w
        )
        if scan:
            sleeping_antennas = self.cfg.antennas - active_antennas
            power_w += sleeping_antennas * self.cfg.antenna_circuit_w * self.cfg.pilot_symbols / self.cfg.coherence_symbols
        transition_j = (
            np.count_nonzero(mask != previous_mask)
            * self.cfg.switch_energy_j_per_subarray
            * switch_scale
        )
        energy_j = power_w * self.cfg.slot_s + transition_j
        delivered_mbit = float(np.sum(rates) * self.cfg.slot_s)
        requirement = self.cfg.rate_requirement_mbps if rate_requirement_mbps is None else rate_requirement_mbps
        outage = float(np.mean(rates < np.asarray(requirement)))
        ee = delivered_mbit / max(energy_j, 1e-12)
        return LinkResult(rates, sinr, float(power_w), float(energy_j), outage, ee)
