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


def create_one_hot_vs_dense():
    """Contrast a one-hot vector against a dense embedding."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.4))

    n_show = 16
    hot_index = 6
    ax1.set_xlim(-0.5, n_show + 2.5)
    ax1.set_ylim(-0.3, 1.3)
    for i in range(n_show):
        is_hot = (i == hot_index)
        rect = plt.Rectangle((i, 0), 0.9, 1.0,
                              facecolor=CLARET if is_hot else PAPER,
                              edgecolor=INK if is_hot else RULE,
                              linewidth=1.4 if is_hot else 0.7)
        ax1.add_patch(rect)
        ax1.text(i + 0.45, 0.5, '1' if is_hot else '0', ha='center', va='center',
                  fontsize=FONT_SIZE - 2, color=PAPER if is_hot else SOFT)
    ax1.text(n_show + 1.0, 0.5, '...', ha='center', va='center', fontsize=FONT_SIZE,
              color=SOFT)
    ax1.axis('off')
    ax1.set_title('One-hot: a single 1 among thousands of zeros (dimension = vocabulary size)',
                  fontsize=FONT_SIZE, fontweight='bold')

    np.random.seed(11)
    values = np.random.uniform(-1, 1, size=8)
    n_dense = len(values)
    offset = (n_show - n_dense) / 2
    ax2.set_xlim(-0.5, n_show + 2.5)
    ax2.set_ylim(-0.6, 1.3)
    cmap = diverging_cmap()
    for i, v in enumerate(values):
        rect = plt.Rectangle((offset + i, 0), 0.9, 1.0, facecolor=cmap((v + 1) / 2),
                              edgecolor=INK, linewidth=0.9)
        ax2.add_patch(rect)
        ax2.text(offset + i + 0.45, -0.18, '{:.2f}'.format(v), ha='center', va='top',
                  fontsize=FONT_SIZE - 2, color=SOFT)
    ax2.axis('off')
    ax2.set_title('Dense embedding: every dimension carries a meaningful value (dimension = a few hundred)',
                  fontsize=FONT_SIZE, fontweight='bold')

    fig.text(0.5, 0.01, 'Illustrative dimensions and values, not to scale.', ha='center',
              fontsize=FONT_SIZE - 2, style='italic', color=SOFT)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig('figures/one_hot_vs_dense.pdf')
    plt.close()
    print("Created: one_hot_vs_dense.pdf")


if __name__ == "__main__":
    create_one_hot_vs_dense()
