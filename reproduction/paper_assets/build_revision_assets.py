"""Build revision evidence from frozen records, keeping discovery and confirmation separate."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'output/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from build_focus_assets import HERE, ROOT, SENS, METHODS, NAMES, read, focused, digest, table, save_figure

CONF = HERE.parent / 'xl_mimo_sim/results/joint_power_20260922_focus_confirmation'
METRICS = ('energy_j', 'outage', 'mean_active', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j')


def validate(episodes, summaries, methods, drops):
    for method in methods:
        rows = [r for r in episodes if r['method'] == method]
        seeds = (11, 22) if method == 'Value4' else (0,)
        assert len(rows) == drops * len(seeds)
        assert {(int(r['trajectory']), int(r['seed'])) for r in rows} == {
            (i, seed) for i in range(drops) for seed in seeds}
        for metric in METRICS:
            np.testing.assert_allclose(np.mean([float(r[metric]) for r in rows]),
                float(summaries[method][metric]), rtol=0, atol=1e-10)
        np.testing.assert_allclose(sum(float(summaries[method][k]) for k in
            ('circuit_j', 'pa_j', 'switching_j', 'pilot_j')),
            float(summaries[method]['energy_j']), rtol=0, atol=1e-9)


def main():
    inputs = [root / f'{name}.csv' for root in (ROOT, CONF)
              for name in ('episodes', 'summary', 'paired')]
    inputs += [CONF / 'report.json', SENS / 'paired.csv', HERE / 'focus_sensitivity.pdf']
    hashes = {str(p): digest(p) for p in inputs}
    old = {r['method']: r for r in read(ROOT/'summary.csv') if focused(r) and r['method'] in METHODS}
    new = {r['method']: r for r in read(CONF/'summary.csv') if int(r['seed']) in (-1, 0)}
    validate([r for r in read(ROOT/'episodes.csv') if focused(r)], old, METHODS, 20)
    validate(read(CONF/'episodes.csv'), new, ('Value4', 'WBbudgetPRS4'), 40)
    pairs = ([r for r in read(ROOT/'paired.csv') if focused(r) and r['reference'] == 'WBbudgetPRS4'],
             read(CONF/'paired.csv'))
    for summaries, rows in zip((old, new), pairs):
        assert len(rows) == 2
        for row in rows:
            np.testing.assert_allclose(float(row['difference']),
                float(summaries['Value4'][row['metric']])-float(summaries['WBbudgetPRS4'][row['metric']]),
                rtol=0, atol=1e-10)

    lines = [r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
        r'Method & Energy (J) & Outage (\%) & Active\\', r'\midrule']
    for heading, summaries, methods in (
            ('Exploration: 20 drops', old, METHODS),
            ('Independent confirmation: 40 new drops', new, ('Value4', 'WBbudgetPRS4'))):
        lines.append(r'\multicolumn{4}{@{}l}{\textit{'+heading+r'}}\\')
        for method in methods:
            r = summaries[method]
            lines.append(' & '.join((NAMES[method], f"{float(r['energy_j']):.2f}",
                f"{100*float(r['outage']):.3f}", f"{float(r['mean_active']):.2f}"))+r'\\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    (HERE/'revision_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    terms = ('circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'energy_j')
    labels = ('Circuit/processing', 'Power amplifier', 'Switching', 'Pilots', 'Total')
    components = []
    for term, label in zip(terms, labels):
        a, b = float(new['Value4'][term]), float(new['WBbudgetPRS4'][term])
        components.append([label, f'{a:.3f}', f'{b:.3f}', f'{a-b:+.3f}'])
    table('revision_components.tex', r'Component & BA-RL4 & WB-B4 & Difference', components, '@{}lrrr@{}')

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.65), constrained_layout=True)
    for ax, metric, title in zip(axes, ('energy_j', 'outage'),
            ('(a) Energy difference (J)', '(b) Outage difference (pp)')):
        factor = 100 if metric == 'outage' else 1
        for i, (rows, color, marker) in enumerate(zip(pairs, ('#60666E', '#0B5CAD'), ('o', 's'))):
            row = next(r for r in rows if r['metric'] == metric)
            mean, lo, hi = [factor*float(row[k]) for k in ('difference', 'ci_low', 'ci_high')]
            ax.plot([lo, hi], [i, i], color=color, lw=1.3)
            ax.plot([lo, hi], [i, i], '|', color=color, ms=6)
            ax.plot(mean, i, marker=marker, color=color, ms=4.5)
        ax.axvline(0, color='0.35', lw=.8, ls='--')
        ax.set_yticks((0, 1), ('Exploration\n(n=20)', 'Confirmation\n(n=40)'))
        ax.set_ylim(1.55, -.55)
        ax.set_xlabel(title)
        ax.grid(axis='x', color='#D7DEE8', lw=.5)
    save_figure(fig, 'revision_paired')
    assert all(digest(Path(p)) == h for p, h in hashes.items())
    report = dict(input_hashes=hashes, builder_sha256=digest(Path(__file__)),
        old_new_samples_pooled=False, means_recomputed=True, saved_intervals_unchanged=True,
        original_experiment_data_changed=False, confirmation_drops=40, exploratory_drops=20,
        confirmation_executions=120, exploratory_table_executions=120,
        energy_saving_percent=100*(1-float(new['Value4']['energy_j'])/float(new['WBbudgetPRS4']['energy_j'])),
        additional_controller_power_break_even_w=(float(new['WBbudgetPRS4']['energy_j'])-
            float(new['Value4']['energy_j']))/7,
        new_training_runs=0, new_simulations=0)
    (HERE/'revision_manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
