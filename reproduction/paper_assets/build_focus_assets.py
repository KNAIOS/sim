"""Extract the complete declared subset; never change simulation inputs/results."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent/'xl_mimo_sim/results/joint_power_20260922_standard_transfer'
SENS = HERE.parent/'xl_mimo_sim/results/joint_power_20260922_sensitivity'
METHODS = ('Value4', 'PRS4', 'budgetPRS4', 'WBbudgetPRS4', 'Full array')
NAMES = dict(zip(METHODS, ('BA-RL4', 'PRS4', 'B-PRS4', 'WB-B4', 'Full array')))
REFS = METHODS[1:-1]


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def focused(row):
    return row['state'] == 'NLOS' and float(row['load']) == .5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(name, header, rows, spec):
    text = [rf'\begin{{tabular}}{{{spec}}}', r'\toprule', header+r'\\', r'\midrule']
    text.extend(' & '.join(row)+r'\\' for row in rows)
    text.extend([r'\bottomrule', r'\end{tabular}'])
    (HERE/name).write_text('\n'.join(text)+'\n', encoding='utf-8')


def save_figure(fig, stem):
    for ext, dpi in (('pdf', None), ('svg', None), ('png', 300), ('tiff', 600)):
        fig.savefig(HERE/f'{stem}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def sensitivity():
    """Reformat existing auxiliary evidence without changing points or intervals."""
    rows = read(SENS/'paired.csv')
    settings = (1, 0, 2, 3, 4)
    labels = ('beta=0', 'Default', 'beta=1.3', 'eta=.8', 'eta=1')
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.8), sharex=True, constrained_layout=True)
    for scenario, color, marker, name in ((0, '#0B5CAD', 'o', 'Low'),
            (1, '#0F766E', 's', 'Nominal'), (2, '#B64040', '^', 'High')):
        for ax, metric, label in zip(axes, ('energy_j', 'outage'),
                ('(a) Energy change (J)', '(b) Outage change (pp)')):
            for position, setting in enumerate(settings):
                if setting == 0:
                    mean = lo = hi = 0.
                else:
                    row = next(r for r in rows if int(r['scenario']) == scenario
                        and int(r['setting']) == setting and r['method'] == 'Value4'
                        and r['comparison'] == 'default parameter' and r['metric'] == metric)
                    factor = 100 if metric == 'outage' else 1
                    mean, lo, hi = [factor*float(row[k]) for k in ('difference', 'ci_low', 'ci_high')]
                x = position+(scenario-1)*.12
                ax.plot([x, x], [lo, hi], color=color, lw=1.1)
                ax.plot(x, mean, marker=marker, color=color, ms=3.5,
                        label=name if position == 0 else None)
            ax.set_ylabel(label)
    for ax in axes:
        ax.axhline(0, color='0.4', lw=.8, ls='--')
        ax.grid(axis='y', color='#D7DEE8', lw=.5)
    axes[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.28),
                   columnspacing=.8, fontsize=7.5)
    axes[1].set_xticks(range(5), labels, rotation=20, fontsize=7.5)
    save_figure(fig, 'focus_sensitivity')


def main():
    paths = [ROOT/f'{name}.csv' for name in ('episodes', 'summary', 'paired')]
    paths.append(SENS/'paired.csv')
    hashes = {str(p): digest(p) for p in paths}
    summaries = {r['method']: r for r in read(paths[1]) if focused(r) and r['method'] in METHODS}
    episodes = [r for r in read(paths[0]) if focused(r) and r['method'] in METHODS]
    paired = [r for r in read(paths[2]) if focused(r) and r['reference'] in REFS]
    assert len(summaries) == 5 and len(episodes) == 120 and len(paired) == 6
    for method in METHODS:
        seeds = (11, 22) if method == 'Value4' else (0,)
        expected = {(i, seed) for i in range(20) for seed in seeds}
        rows = [r for r in episodes if r['method'] == method]
        assert {(int(r['trajectory']), int(r['seed'])) for r in rows} == expected
        for metric in ('energy_j', 'outage', 'mean_active', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j'):
            np.testing.assert_allclose(np.mean([float(r[metric]) for r in rows]),
                                       float(summaries[method][metric]), rtol=0, atol=1e-10)
    for row in paired:
        np.testing.assert_allclose(float(row['difference']),
            float(summaries['Value4'][row['metric']])-float(summaries[row['reference']][row['metric']]),
            rtol=0, atol=1e-10)
    for filename, rows in (('focus_summary.csv', [summaries[m] for m in METHODS]),
                           ('focus_episodes.csv', episodes), ('focus_paired.csv', paired)):
        write_csv(HERE/filename, rows)
    table('focus_table.tex', r'Method & Energy (J) & Outage (\%) & Active',
          [[NAMES[m], f"{float(summaries[m]['energy_j']):.2f}",
            f"{100*float(summaries[m]['outage']):.3f}",
            f"{float(summaries[m]['mean_active']):.2f}"] for m in METHODS], '@{}lrrr@{}')
    terms = ('circuit_j', 'pa_j', 'switching_j', 'pilot_j', 'energy_j')
    labels = ('Circuit/processing', 'Power amplifier', 'Switching', 'Pilots', 'Total')
    components = []
    for term, label in zip(terms, labels):
        a, b = float(summaries['Value4'][term]), float(summaries['WBbudgetPRS4'][term])
        components.append([label, f'{a:.3f}', f'{b:.3f}', f'{a-b:+.3f}'])
    table('focus_components.tex', r'Component & BA-RL4 & WB-B4 & Difference', components, '@{}lrrr@{}')
    for m in ('Value4', 'WBbudgetPRS4'):
        np.testing.assert_allclose(sum(float(summaries[m][k]) for k in terms[:-1]),
                                   float(summaries[m]['energy_j']), rtol=0, atol=1e-9)

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ('energy_j', 'outage'),
            ('(a) Energy difference (J)', '(b) Outage difference (pp)')):
        factor = 100 if metric == 'outage' else 1
        for i, ref in enumerate(REFS):
            row = next(r for r in paired if r['reference'] == ref and r['metric'] == metric)
            mean, lo, hi = [factor*float(row[k]) for k in ('difference', 'ci_low', 'ci_high')]
            ax.plot([lo, hi], [i, i], color='#0B5CAD', lw=1.2)
            ax.plot([lo, hi], [i, i], '|', color='#0B5CAD', ms=5)
            ax.plot(mean, i, marker=('o', 's', '^')[i], color='#0B5CAD', ms=4)
        ax.axvline(0, color='0.35', lw=.8, ls='--')
        ax.set_yticks(range(3), [NAMES[r] for r in REFS])
        ax.set_ylim(2.5, -.5)
        ax.set_xlabel(title)
        ax.grid(axis='x', color='#D7DEE8', lw=.5)
    save_figure(fig, 'focus_paired')
    sensitivity()
    before = (HERE/'archive/pre_focus_paper_joint.tex').read_text(encoding='utf-8')
    after = (HERE/'paper_joint.tex').read_text(encoding='utf-8')
    assert before.split(r'\section{Experimental Evaluation}')[0] == after.split(r'\section{Experimental Evaluation}')[0]
    assert before.split(r'\section{Conclusion}')[1] == after.split(r'\section{Conclusion}')[1]
    assert all(digest(Path(p)) == h for p, h in hashes.items())
    report = dict(input_hashes=hashes, source_sha256=digest(Path(__file__)),
        state='NLOS', load=.5, allowed_counts=[4, 16], independent_drops=20,
        learned_seeds=[11, 22], executions=120, saved_paired_intervals=6,
        means_recomputed_from_all_subset_episodes=True, original_data_changed=False,
        title_and_nonexperiment_sections_unchanged=True, scope_chosen_after_broad_evaluation=True,
        new_simulations=0, new_training=0)
    (HERE/'focus_manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
