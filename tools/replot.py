"""Replot the two current-paper figures from unchanged archived CSV data."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/replotted'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
sys.path.insert(0, str(ROOT/'reproduction/paper_assets'))

import build_focus_assets as focus
import build_revision_assets as revision


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep all original input paths; redirect only generated figure/table files.
    focus.HERE = revision.HERE = OUT
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8, 'axes.labelsize':8, 'xtick.labelsize':8, 'ytick.labelsize':8,
        'axes.linewidth':.8, 'axes.spines.top':False, 'axes.spines.right':False,
        'legend.frameon':False, 'svg.fonttype':'none', 'pdf.fonttype':42})
    focus.sensitivity()
    revision.main()
    print('Replotted files are in:', OUT)


if __name__ == '__main__':
    main()
