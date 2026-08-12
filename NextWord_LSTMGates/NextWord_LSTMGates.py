# Added by tools/build_quantlets.py. Everything below this block is the
# course generator's own source, so this folder draws the figure the
# slides show rather than a second implementation of it.
#
# The chdir makes every relative path in that source resolve against this
# folder instead of the course repository, which is what lets the code be
# copied across unchanged. The mkdir is needed because matplotlib will not
# create the directory it is asked to save into: without it every Quantlet
# dies on FileNotFoundError the first time anyone runs one.
import os as _os
import pathlib as _pathlib
_os.chdir(_pathlib.Path(__file__).resolve().parent)
_pathlib.Path('figures').mkdir(exist_ok=True)


"""
Generate charts for Hour 2: Smarter Prediction (RNNs to Transformers)
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch


import sys; sys.path.insert(0, 'figures_src')
from palette import (use_course_style, CYCLE, CLARET, TEAL, GOLD, INK,
                     SOFT, PAPER, RULE, sequential_cmap, diverging_cmap)
use_course_style()
sns.set_palette(CYCLE)


# Common settings
FONT_SIZE = 8
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.titlesize': FONT_SIZE + 2
})


def create_lstm_gates():
    """The cell state as a highway; gates are learned taps on it."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    highway_y = 3.4
    ax.annotate('', xy=(11.4, highway_y), xytext=(0.6, highway_y),
                arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=6, alpha=0.9),
                zorder=1)
    ax.text(0.6, highway_y + 0.55, 'C(t-1)', fontsize=FONT_SIZE, color=TEAL,
            fontweight='bold', ha='left')
    ax.text(11.4, highway_y + 0.55, 'C(t)', fontsize=FONT_SIZE, color=TEAL,
            fontweight='bold', ha='right')

    gates = [
        (3.0, CLARET, 'Forget gate', 'keep or drop'),
        (6.0, GOLD, 'Input gate', 'add new info'),
        (9.0, SOFT, 'Output gate', 'reveal to h(t)'),
    ]
    for x, color, name, action in gates:
        box = FancyBboxPatch((x - 0.9, highway_y - 0.5), 1.8, 1.0,
                              boxstyle='round,pad=0.06', facecolor=color,
                              edgecolor=INK, linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(x, highway_y, name, ha='center', va='center', fontsize=FONT_SIZE - 1,
                color=PAPER, fontweight='bold', zorder=4)
        ax.annotate('', xy=(x, highway_y - 0.55), xytext=(x, 1.0), zorder=2,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.4))
        ax.text(x, 0.65, action, ha='center', va='center', fontsize=FONT_SIZE - 2, color=color)

    ax.text(6.0, 0.1, 'Every gate reads h(t-1) and x(t) and outputs a value between 0 and 1.',
            ha='center', fontsize=FONT_SIZE - 2, style='italic', color=SOFT)
    ax.set_title('The LSTM Cell: Gates Are Learned Taps on the Cell-State Highway',
                fontsize=FONT_SIZE + 1, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/lstm_gates.pdf')
    plt.close()
    print("Created: lstm_gates.pdf")


if __name__ == "__main__":
    create_lstm_gates()
