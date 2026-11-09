from __future__ import annotations

import numpy as np

from .config import Config


def _shift(values: np.ndarray, displacement: float) -> np.ndarray:
    q = np.arange(values.shape[0], dtype=float)
    output = np.empty_like(values)
    for k in range(values.shape[1]):
        output[:, k] = np.interp(q - displacement, q, values[:, k], left=values[0, k], right=values[-1, k])
    return output


class VisibilityBelief:
    """Full-support advection/birth-death belief from partial subarray CSI."""

    def __init__(self, cfg: Config, initial_gain_db: np.ndarray):
        self.cfg = cfg
        self.gain_db = initial_gain_db.copy()
        self.probability = self._gain_to_probability(self.gain_db)
        self.previous_probability = self.probability.copy()
        self.velocity = np.zeros(cfg.users)
        self.uncertainty = np.full_like(self.gain_db, 0.12)
        self._forecast_cache = {}

    @staticmethod
    def _gain_to_probability(gain_db: np.ndarray) -> np.ndarray:
        relative = gain_db - np.max(gain_db, axis=0, keepdims=True)
        return 1.0 / (1.0 + np.exp(-(relative + 8.0) / 1.8))

    def update(self, observation_db: np.ndarray, observed_q: np.ndarray) -> None:
        self._forecast_cache.clear()
        observed = np.repeat(observed_q[:, None], self.cfg.users, axis=1)
        predicted = np.empty_like(self.gain_db)
        for k in range(self.cfg.users):
            predicted[:, k] = _shift(self.gain_db[:, [k]], self.velocity[k])[:, 0]
        self.gain_db = np.where(observed, 0.75 * observation_db + 0.25 * predicted, predicted)
        measured_probability = self._gain_to_probability(self.gain_db)
        new_probability = np.where(observed, measured_probability, 0.96 * self.probability + 0.04 * measured_probability)
        q_index = np.arange(self.cfg.subarrays, dtype=float)
        for k in range(self.cfg.users):
            old_mass = max(float(self.probability[:, k].sum()), 1e-9)
            new_mass = max(float(new_probability[:, k].sum()), 1e-9)
            old_center = float(np.dot(q_index, self.probability[:, k]) / old_mass)
            new_center = float(np.dot(q_index, new_probability[:, k]) / new_mass)
            innovation = np.clip(new_center - old_center, -0.55, 0.55)
            self.velocity[k] = 0.72 * self.velocity[k] + 0.28 * innovation
        self.previous_probability = self.probability
        self.probability = np.clip(new_probability, 0.01, 0.995)
        self.uncertainty = np.where(observed, 0.08, np.minimum(self.uncertainty + 0.035, 0.65))

    def predict(self, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if horizon in self._forecast_cache:
            return self._forecast_cache[horizon]
        h = np.arange(1, horizon + 1)[:, None, None]
        position = np.clip(np.arange(self.cfg.subarrays)[None, :, None]
                           - h * self.velocity[None, None, :], 0, self.cfg.subarrays - 1)
        left = np.floor(position).astype(int)
        right = np.minimum(left + 1, self.cfg.subarrays - 1)
        fraction = position - left
        users = np.arange(self.cfg.users)[None, None, :]
        def interpolate(values):
            lo = values[left, users]
            return lo + fraction * (values[right, users] - lo)
        gain = interpolate(self.gain_db)
        transported = interpolate(self.probability)
        neighbour = (np.roll(transported, 1, axis=1) + transported
                     + np.roll(transported, -1, axis=1)) / 3.0
        neighbour[:, 0] = (transported[:, 0] + transported[:, 1]) / 2.0
        neighbour[:, -1] = (transported[:, -2] + transported[:, -1]) / 2.0
        probability = np.clip(0.97**h * transported + (1.0 - 0.97**h)
                              * (0.8 * neighbour + 0.02), 0.01, 0.995)
        uncertainty = np.minimum(self.uncertainty[None] + 0.045 * h, 0.8)
        self._forecast_cache[horizon] = gain, probability, uncertainty
        return gain, probability, uncertainty
