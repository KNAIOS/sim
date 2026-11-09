from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .channel import generate_trajectory
from .config import Config


class ProbabilisticLSTM(nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.features = features
        self.lstm = nn.LSTM(2 * features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * features)
        with torch.no_grad():
            self.head.weight.mul_(0.05)
            self.head.weight[:features].zero_()
            self.head.bias[:features].zero_()
            self.head.bias[features:].fill_(-3.0)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.lstm(sequence)
        parameters = self.head(output[:, -1])
        delta, log_variance = parameters.chunk(2, dim=1)
        mean = sequence[:, -1, : self.features] + delta
        return mean, torch.clamp(log_variance, -7.0, 5.0)


@dataclass
class TrainedPredictor:
    model: ProbabilisticLSTM
    mean: np.ndarray
    std: np.ndarray
    sigma_scale: float
    validation: dict[str, float]

    def predict(self, values: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = (values - self.mean) / self.std
        x = np.concatenate((normalized, observed.astype(float)), axis=1)
        with torch.no_grad():
            mu, log_var = self.model(torch.as_tensor(x[None], dtype=torch.float32))
        mu_np = mu.numpy()[0] * self.std + self.mean
        sigma_np = np.exp(0.5 * log_var.numpy()[0]) * self.std * self.sigma_scale
        return mu_np, sigma_np


def _forward_fill(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for t in range(1, len(filled)):
        filled[t] = np.where(mask[t], filled[t], filled[t - 1])
    return filled


def _make_examples(cfg: Config, seeds: range, speed_choices: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, seed in enumerate(seeds):
        speed = speed_choices[index % len(speed_choices)]
        gain = generate_trajectory(cfg, seed, speed).subarray_gain_db.reshape(cfg.episode_steps, -1)
        rng = np.random.default_rng(seed + 91_337)
        subarray_mask = rng.random((cfg.episode_steps, cfg.subarrays)) < 0.65
        subarray_mask[:: cfg.scan_period] = True
        mask = np.repeat(subarray_mask[:, :, None], cfg.users, axis=2).reshape(cfg.episode_steps, -1)
        filled = _forward_fill(gain, mask)
        for t in range(cfg.history, cfg.episode_steps - cfg.prediction_horizon):
            x = np.concatenate((filled[t - cfg.history : t], mask[t - cfg.history : t]), axis=1)
            sequences.append(x)
            targets.append(gain[t + cfg.prediction_horizon - 1])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def train_predictor(cfg: Config, seed: int = 17) -> TrainedPredictor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    count = cfg.predictor_trajectories
    split = max(2, int(0.8 * count))
    train_x, train_y = _make_examples(cfg, range(10_000, 10_000 + split), cfg.speeds_mps)
    val_x, val_y = _make_examples(cfg, range(20_000, 20_000 + max(2, count - split)), cfg.speeds_mps)
    features = cfg.subarrays * cfg.users
    mean = train_y.mean(axis=0)
    std = np.maximum(train_y.std(axis=0), 0.5)
    train_x[:, :, :features] = (train_x[:, :, :features] - mean) / std
    train_y = (train_y - mean) / std
    val_x[:, :, :features] = (val_x[:, :, :features] - mean) / std

    model = ProbabilisticLSTM(features, cfg.predictor_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    batch_size = min(128, len(train_x))
    val_tensor = torch.as_tensor(val_x, dtype=torch.float32)
    with torch.no_grad():
        initial_mu, _ = model(val_tensor)
    val_y_n = (val_y - mean) / std
    best_rmse_n = float(np.sqrt(np.mean((val_y_n - initial_mu.numpy()) ** 2)))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    mean_improved = False
    model.train()
    for _ in range(cfg.predictor_epochs):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            xb = torch.as_tensor(train_x[indices], dtype=torch.float32)
            yb = torch.as_tensor(train_y[indices], dtype=torch.float32)
            mu, log_var = model(xb)
            squared_error = (yb - mu) ** 2
            nll = 0.5 * (log_var + squared_error / torch.exp(log_var)).mean()
            loss = squared_error.mean() + 0.05 * nll
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            candidate_mu, _ = model(val_tensor)
        candidate_rmse = float(np.sqrt(np.mean((val_y_n - candidate_mu.numpy()) ** 2)))
        if candidate_rmse < best_rmse_n:
            best_rmse_n = candidate_rmse
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            mean_improved = True
        model.train()

    if mean_improved:
        model.load_state_dict(best_state)
    else:
        # Keep the trained recurrent uncertainty head but fall back to the
        # validation-selected persistence mean instead of reporting degradation.
        with torch.no_grad():
            model.head.weight[:features].zero_()
            model.head.bias[:features].zero_()
    model.eval()
    with torch.no_grad():
        mu_n, log_var = model(torch.as_tensor(val_x, dtype=torch.float32))
    mu = mu_n.numpy() * std + mean
    sigma = np.exp(0.5 * log_var.numpy()) * std
    error = val_y - mu
    standardized = np.abs(error) / np.maximum(sigma, 1e-6)
    sigma_scale = float(np.quantile(standardized, 0.90) / 1.645)
    # Conservative mobility-stratified calibration: use the largest scale
    # required by any validation speed instead of averaging away the hardest regime.
    conditional_scales: list[float] = []
    for speed_index, speed in enumerate(cfg.speeds_mps):
        cal_x, cal_y = _make_examples(
            cfg,
            range(25_000 + 100 * speed_index, 25_002 + 100 * speed_index),
            (speed,),
        )
        cal_x[:, :, :features] = (cal_x[:, :, :features] - mean) / std
        with torch.no_grad():
            cal_mu_n, cal_log_var = model(torch.as_tensor(cal_x, dtype=torch.float32))
        cal_mu = cal_mu_n.numpy() * std + mean
        cal_sigma = np.exp(0.5 * cal_log_var.numpy()) * std
        ratio = np.abs(cal_y - cal_mu) / np.maximum(cal_sigma, 1e-6)
        conditional_scales.append(float(np.quantile(ratio, 0.90) / 1.645))
    sigma_scale = max([sigma_scale, *conditional_scales])
    sigma *= sigma_scale
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    coverage90 = float(np.mean(np.abs(error) <= 1.645 * sigma))
    last_value = val_x[:, -1, :features] * std + mean
    last_rmse = float(np.sqrt(np.mean((val_y - last_value) ** 2)))
    return TrainedPredictor(
        model=model,
        mean=mean,
        std=std,
        sigma_scale=sigma_scale,
        validation={"rmse_db": rmse, "mae_db": mae, "coverage90": coverage90, "last_value_rmse_db": last_rmse},
    )


class BeliefTracker:
    def __init__(self, cfg: Config, predictor: TrainedPredictor, initial_gain: np.ndarray):
        self.cfg = cfg
        self.predictor = predictor
        flat = initial_gain.reshape(-1)
        self.values = np.repeat(flat[None], cfg.history, axis=0)
        self.observed = np.ones_like(self.values, dtype=bool)

    def update(self, gain_db: np.ndarray, mask: np.ndarray, full_scan: bool) -> None:
        visible_q = np.ones(self.cfg.subarrays, dtype=bool) if full_scan else mask.astype(bool)
        visible = np.repeat(visible_q[:, None], self.cfg.users, axis=1).reshape(-1)
        latest = self.values[-1].copy()
        actual = gain_db.reshape(-1)
        latest[visible] = actual[visible]
        self.values = np.vstack((self.values[1:], latest))
        self.observed = np.vstack((self.observed[1:], visible))

    def belief(self) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = self.predictor.predict(self.values, self.observed)
        return mu.reshape(self.cfg.subarrays, self.cfg.users), sigma.reshape(self.cfg.subarrays, self.cfg.users)
