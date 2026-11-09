"""Prepared-state decision timing, AFTER training, not policy energy evidence."""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.budgeted import evaluate
from joint_power.calibrated.histogram import load_predictor
from joint_power.model import Environment
from joint_power.postdecision.encoding import decision
from joint_power.postdecision.learning import Learner
from joint_power.postdecision.study import ROOT, intact
from joint_power.study import digest, make_trace, write_rows
from major_revision.budget_search import search


def main():
    torch.set_num_threads(1)
    assert (ROOT/'training_complete.json').exists(), 'Do not benchmark while training'
    intact()
    output = Path('results/joint_decision_budget_checks_20260922')
    output.mkdir(exist_ok=False)
    checkpoint = ROOT/'PDS-branch_seed11/selected.pt'
    inputs = [Path(__file__), Path('joint_power/budgeted.py'),
              Path('joint_power/model.py'), checkpoint,
              ROOT/'manifest.json', ROOT/'training_complete.json']
    hashes = {str(p): digest(p) for p in inputs}
    cfg, predictor, learner = make_config('validation'), load_predictor(), Learner(11)
    saved = torch.load(checkpoint, weights_only=True)
    learner.online.load_state_dict(saved['state'])
    samples, equivalence = [], []
    rng = np.random.default_rng(2026092215)
    for trajectory in range(2):
        env = Environment(cfg, make_trace(cfg, 3509220000+trajectory, (2., 1.)))
        for state_index, count in enumerate((4, 10, 6, 16)):
            reference = env.candidates()
            for stride in (1, 2, 3, 4, 8):
                result = evaluate(env, stride)
                errors = {k: float(np.max(abs(getattr(result, k)-getattr(reference, k)[result.ids])))
                          for k in ('energy', 'outage', 'shortfall')}
                assert max(errors.values()) < 1e-9
                equivalence.append(dict(trajectory=trajectory, state=state_index, stride=stride,
                    candidates=len(result.ids), candidate_slots=result.candidate_slots, **errors))

            def raw(stride):
                candidates = evaluate(env, stride)
                return candidates.masks[candidates.choose()]

            def rl():
                state = decision(env, env.candidates(), predictor)
                return state.masks[learner.choose(state, saved['parameter'])]

            operations = {
                'PRS power selection': lambda: search(cfg, env.tracker, env.hw, env.req),
                'Joint full selection': lambda: raw(1),
                'Joint coarse selection': lambda: raw(3),
                'PDS-branch selection': rl,
            }
            # Warm each branch; randomize subsequent order to reduce drift bias.
            for operation in operations.values():
                for _ in range(5):
                    operation()
            methods = list(operations)
            for repeat in range(50):
                for index in rng.permutation(len(methods)):
                    method = methods[index]
                    start = time.perf_counter_ns()
                    operations[method]()
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    samples.append(dict(trajectory=trajectory, state=state_index,
                                        repeat=repeat, method=method, milliseconds=elapsed))
            env.block(np.arange(16) < count)
    summary = []
    for method in operations:
        values = [r['milliseconds'] for r in samples if r['method'] == method]
        summary.append(dict(method=method, samples=len(values), median_ms=float(np.median(values)),
                            p95_ms=float(np.quantile(values, .95))))
    write_rows(output/'timings.csv', samples)
    write_rows(output/'summary.csv', summary)
    write_rows(output/'equivalence.csv', equivalence)
    assert all(digest(p) == h for p, h in hashes.items())
    intact()
    report = dict(summary=summary, numerical_cases=len(equivalence), input_hashes=hashes,
        python=platform.python_version(), platform=platform.platform(), cpu=platform.processor(),
        torch=torch.__version__, numpy=np.__version__, torch_threads=torch.get_num_threads(),
        scope='Prepared-state selection including forecasts/features/solves/network; '
              'excludes shared CSI update, actual per-slot power/beam execution, data generation and training. '
              'Local descriptive wall time, not controller electricity or energy-performance validation.',
        seed_selection='Seed11 fixed in benchmark code, not fastest seed selected after timing',
        prior_data_unchanged=True)
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
