"""Measured-current CSI consistency and empirical future rate-risk estimates."""
from dataclasses import dataclass

import numpy as np
import torch

from future_vr.controllers import select_top
from major_revision.action_value import hold_predictions, slot_energy
from major_revision.full_mask_audit import causal_forecast
from major_revision.gram_control import subarray_grams, rates_from_gram


FEATURE_NAMES = ('log_rate_margin', 'squared_margin', 'horizon', 'margin_horizon',
    'weighted_csi_age', 'age_horizon', 'unobserved_fraction', 'active_fraction',
    'requirement', 'requirement_change', 'support_velocity', 'uncertainty',
    'filtered_measured_gain_difference', 'margin_age')


def hardware_path(cfg, hw, masks):
    """Exact hold-command availability/cost; no channel or future observation."""
    active = np.broadcast_to(hw.active, masks.shape).copy()
    countdown = np.broadcast_to(hw.countdown, masks.shape).copy()
    available, energy = [], []
    for _ in range(cfg.horizon):
        observed = active.sum(1)
        off = ~masks & (active | (countdown > 0))
        on = masks & ~active & (countdown == 0)
        active[off], countdown[off] = False, 0
        countdown[on] = cfg.wake_delay_slots
        if cfg.wake_delay_slots == 0:
            active[on] = True
        transition = cfg.sleep_energy_j*off.sum(1)+cfg.wake_energy_j*on.sum(1)
        energy.append(slot_energy(cfg, active.sum(1), (countdown > 0).sum(1), observed, transition))
        available.append(active.copy())
        ready = countdown == 1
        countdown[countdown > 0] -= 1
        active[ready] = True
    return np.stack(available, axis=1), np.stack(energy, axis=1)


def predictions(cfg, tracker, hw, requirements, previous_requirements, age, score):
    """Pure causal interface. Correct h=0 observed CSI; do not change h>0."""
    old = hold_predictions(cfg, tracker, hw, score)
    forecast, _ = causal_forecast(cfg, tracker)
    measured = subarray_grams(cfg, tracker.channel_estimate)
    forecast = forecast.copy()
    forecast[0, hw.active] = measured[hw.active]
    available, energy = hardware_path(cfg, hw, old['masks'])
    grams = np.einsum('ahq,hqkl->ahkl', available, forecast)
    rates = rates_from_gram(cfg, grams, available.sum(2)*cfg.antennas_per_subarray)
    features = risk_features(cfg, tracker, hw, requirements, previous_requirements,
                             age, available, rates, measured)
    return dict(masks=old['masks'], available=available, energy=energy,
                old_rates=old['rates'], rates=rates, features=features)


def risk_features(cfg, tracker, hw, req, prev_req, age, available, rates, measured):
    shape = rates.shape
    def broad(value):
        return np.broadcast_to(value, shape)
    power = np.maximum(np.diagonal(measured, axis1=-2, axis2=-1).real, 1e-30)
    contribution = available[..., None]*power[None, None]
    weights = contribution/np.maximum(contribution.sum(2, keepdims=True), 1e-30)
    csi_age = np.minimum(age, 128.)/32.
    weighted_age = np.sum(weights*csi_age[None, None, :, None], axis=2)
    uncertainty = np.sum(weights*tracker.uncertainty[None, None], axis=2)
    observed_gain = 10*np.log10(power/cfg.antennas_per_subarray)
    gain_difference = np.sum(weights*(tracker.gain_db-observed_gain)[None, None], axis=2)/10.
    margin = np.clip(np.log(np.maximum(rates, 1e-6)/req[None, None]), -4., 4.)
    horizon = broad(np.arange(cfg.horizon)[None, :, None]/cfg.horizon)
    unobserved = np.sum(available & ~hw.active[None, None], axis=2)/np.maximum(available.sum(2), 1)
    values = (margin, margin**2, horizon, margin*horizon, weighted_age,
        weighted_age*horizon, broad(unobserved[..., None]),
        broad(available.mean(2)[..., None]), broad(req[None, None]/560.),
        broad((req-prev_req)[None, None]/100.), broad(np.abs(tracker.velocity)[None, None]),
        uncertainty, gain_difference, margin*weighted_age)
    result = np.stack(values, axis=-1)
    assert result.shape == (*shape, len(FEATURE_NAMES)) and np.isfinite(result).all()
    return result


def true_hold_outcomes(cfg, channel, requirements, available):
    """OFFLINE labels only. Never called by an online controller."""
    grams = np.stack([subarray_grams(cfg, h) for h in channel])
    sums = np.einsum('ahq,hqkl->ahkl', available, grams)
    rates = rates_from_gram(cfg, sums, available.sum(2)*cfg.antennas_per_subarray)
    return rates, (rates < requirements[None]).astype(float)


@dataclass
class RateRiskModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    temperature: float = 1.

    @classmethod
    def fit(cls, x, y):
        mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
        z = torch.as_tensor((x-mean)/scale, dtype=torch.float64)
        target = torch.as_tensor(y, dtype=torch.float64)
        weight = torch.zeros(x.shape[1]+1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight], lr=1., max_iter=150,
                                     tolerance_grad=1e-8, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            logits = z@weight[:-1]+weight[-1]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss = loss+1e-4*weight[:-1].square().mean()
            loss.backward()
            return loss
        optimizer.step(closure)
        return cls(mean, scale, weight[:-1].detach().numpy().copy(), float(weight[-1].detach()))

    def logits(self, x):
        return (np.asarray(x)-self.mean)/self.scale@self.coefficient+self.intercept

    def calibrate(self, x, y):
        logits = torch.as_tensor(self.logits(x), dtype=torch.float64)
        target = torch.as_tensor(y, dtype=torch.float64)
        log_temperature = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], lr=1., max_iter=80,
                                     tolerance_grad=1e-8, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            value = logits/torch.exp(torch.clamp(log_temperature, -5., 5.))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(value, target)
            loss.backward()
            return loss
        optimizer.step(closure)
        self.temperature = float(torch.exp(torch.clamp(log_temperature.detach(), -5., 5.)))

    def probability(self, x):
        logit = self.logits(x)/self.temperature
        return np.exp(-np.logaddexp(0., -logit))

    def predict(self, data, requirements):
        shape = data['rates'].shape
        output = self.probability(data['features'].reshape(-1, len(FEATURE_NAMES))).reshape(shape)
        output[:, 0] = data['rates'][:, 0] < requirements
        output[data['available'].sum(2) == 0] = 1.
        return output

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, scale=self.scale, coefficient=self.coefficient,
                            intercept=self.intercept, temperature=self.temperature)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z['mean'].copy(), z['scale'].copy(), z['coefficient'].copy(),
                       float(z['intercept']), float(z['temperature']))


def select_action(data, requirements, mode, parameter, minimum=1, model=None):
    energy = data['energy'].sum(1)
    if mode in ('old', 'anchored'):
        rates = data['old_rates'] if mode == 'old' else data['rates']
        req = requirements[None, None]*parameter
        risk = (rates < req).mean((1, 2))
        fallback = (np.maximum(req-rates, 0.)/req).mean((1, 2))
        threshold = .05
    elif mode == 'calibrated' and model is not None:
        risk = model.predict(data, requirements).mean((1, 2))
        fallback, threshold = risk, parameter
    else:
        raise ValueError('Invalid prediction mode or missing risk model')
    def key(i):
        feasible = risk[i] <= threshold
        return (not feasible, energy[i] if feasible else fallback[i], energy[i], i)
    return min(range(minimum-1, len(energy)), key=key)
