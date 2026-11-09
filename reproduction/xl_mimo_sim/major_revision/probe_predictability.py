"""Fixed small predictors of paid-probe outcomes, using causal snapshots only."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from future_vr.belief import VisibilityBelief
from future_vr.system import HardwareState
from major_revision.full_mask_audit import causal_forecast, predicted_batch


FEATURE_NAMES = (
    'remaining_fraction', 'csi_age_fraction', 'age_minus_array_mean',
    'reference_fraction', 'active_fraction', 'waking_fraction',
    'mean_requirement_ratio', 'requirement_relative_spread',
    'relative_ra_score', 'ra_rank_fraction', 'target_uncertainty',
    'mean_absolute_support_velocity', 'mean_relative_gain',
    'worst_relative_gain', 'mean_absolute_gain', 'target_support_score',
    'distance_to_reference', 'forecast_reference_energy',
    'forecast_incremental_energy', 'forecast_reference_outage',
    'forecast_outage_delta', 'forecast_reference_shortfall',
    'forecast_shortfall_delta',
)
KINDS = ('linear', 'kernel')
RIDGE = 1.
FIRST_COST = .33


def eligible_from_snapshot(snapshot):
    return np.flatnonzero(~snapshot['active'].astype(bool)
                         & (snapshot['countdown'] == 0)
                         & ~snapshot['reference'].astype(bool))


def features(cfg, snapshot, t):
    """Return [eligible module ids, 23 causal features]. Never accepts truth data."""
    qids = eligible_from_snapshot(snapshot)
    if not len(qids):
        return qids, np.empty((0, len(FEATURE_NAMES)))
    ref = np.asarray(snapshot['reference'], bool)
    if not np.any(ref):
        raise ValueError('Reference mask must be nonempty')
    tracker = VisibilityBelief(cfg, snapshot['gain_db'])
    for key in ('gain_db', 'velocity', 'uncertainty', 'probability', 'channel_estimate'):
        setattr(tracker, key, np.asarray(snapshot[key]).copy())
    hw = HardwareState(cfg)
    for key in ('active', 'countdown', 'command'):
        setattr(hw, key, np.asarray(snapshot[key]).copy())
    forecast, _ = causal_forecast(cfg, tracker)
    masks = np.tile(ref, (len(qids)+1, 1))
    masks[np.arange(1, len(qids)+1), qids] = True
    pred = predicted_batch(cfg, hw, forecast, snapshot['requirements'], masks)
    base, delta = pred[0], pred[1:]-pred[0]
    score = snapshot['score']
    gain = snapshot['gain_db']
    relative = gain-np.max(gain, axis=0, keepdims=True)
    req = snapshot['requirements']
    rows = []
    for k, q in enumerate(qids):
        rows.append([
            (cfg.episode_steps-t)/cfg.episode_steps,
            snapshot['age'][q]/cfg.episode_steps,
            (snapshot['age'][q]-np.mean(snapshot['age']))/cfg.episode_steps,
            ref.mean(), snapshot['active'].mean(), (snapshot['countdown'] > 0).mean(),
            req.mean()/460., (req.max()-req.min())/req.mean(),
            score[q]/max(float(np.max(np.abs(score))), 1e-12),
            np.mean(score > score[q]), snapshot['uncertainty'][q].mean(),
            np.abs(snapshot['velocity']).mean(), relative[q].mean()/20.,
            relative[q].min()/20., (gain[q].mean()+90.)/20.,
            snapshot['probability'][q].mean(),
            np.min(np.abs(np.flatnonzero(ref)-q))/cfg.subarrays,
            base[0], delta[k, 0], base[1], delta[k, 1], base[2], delta[k, 2],
        ])
    values = np.asarray(rows, dtype=float)
    assert values.shape == (len(qids), len(FEATURE_NAMES)) and np.isfinite(values).all()
    return qids, values


def kernel(x, z):
    distance = np.maximum((x*x).sum(1)[:, None]+(z*z).sum(1)[None]-2*x@z.T, 0.)
    return np.exp(-distance/(2*x.shape[1]))


@dataclass
class OutcomePredictor:
    kind: str
    mean: np.ndarray
    scale: np.ndarray
    target_mean: np.ndarray
    train: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(cls, x, outcomes, kind):
        if kind not in KINDS or len(x) == 0:
            raise ValueError('A supported model and nonempty training set are required')
        x, y = np.asarray(x, float), np.asarray(outcomes, float).copy()
        if x.ndim != 2 or y.shape != (len(x), 2):
            raise ValueError('Expected two-dimensional features and two outcome columns')
        y[:, 0] -= FIRST_COST
        mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
        z = (x-mean)/scale
        target_mean = y.mean(0)
        centered = y-target_mean
        if kind == 'linear':
            weights = np.linalg.solve(z.T@z+RIDGE*np.eye(z.shape[1]), z.T@centered)
            train = np.empty((0, x.shape[1]))
        else:
            weights = np.linalg.solve(kernel(z, z)+RIDGE*np.eye(len(z)), centered)
            train = z
        return cls(kind, mean, scale, target_mean, train, weights)

    def predict(self, x):
        z = (np.asarray(x, float)-self.mean)/self.scale
        design = z if self.kind == 'linear' else kernel(z, self.train)
        result = design@self.weights+self.target_mean
        result[:, 0] += FIRST_COST
        return result

    def save(self, path):
        np.savez_compressed(path, kind=np.array(self.kind), mean=self.mean,
            scale=self.scale, target_mean=self.target_mean, train=self.train,
            weights=self.weights)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(str(data['kind']), *[data[key].copy() for key in (
                'mean', 'scale', 'target_mean', 'train', 'weights')])


def choose(qids, prediction):
    """A point-prediction filter, not a guaranteed reliability constraint."""
    feasible = [i for i in range(len(qids))
                if prediction[i, 0] < 0 and prediction[i, 1] <= 0]
    if not feasible:
        return -1
    chosen = min(feasible, key=lambda i: (prediction[i, 0], int(qids[i])))
    return int(qids[chosen])
