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
Generate charts for Hour 4: Applications & Future
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches
from scipy.stats import norm


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


def create_sampling_strategies():
    """Chart: greedy, temperature, top-k and nucleus sampling reshape one distribution"""
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.6), sharey=True)

    words = ['mat', 'rug', 'floor', 'table', 'chair']
    base_probs = np.array([0.50, 0.22, 0.14, 0.09, 0.05])
    x = np.arange(len(words))
    width = 0.6

    greedy_probs = np.zeros_like(base_probs)
    greedy_probs[np.argmax(base_probs)] = base_probs.max()

    temperature = 1.5
    logits = np.log(base_probs)
    temp_probs = np.exp(logits / temperature)
    temp_probs = temp_probs / temp_probs.sum()

    k = 3
    topk_idx = np.argsort(base_probs)[::-1][:k]
    topk_probs = np.zeros_like(base_probs)
    topk_probs[topk_idx] = base_probs[topk_idx] / base_probs[topk_idx].sum()

    p_threshold = 0.9
    order = np.argsort(base_probs)[::-1]
    cum = np.cumsum(base_probs[order])
    cutoff = int(np.searchsorted(cum, p_threshold) + 1)
    keep_idx = order[:cutoff]
    nucleus_probs = np.zeros_like(base_probs)
    nucleus_probs[keep_idx] = base_probs[keep_idx] / base_probs[keep_idx].sum()

    panels = [
        (axes[0], greedy_probs, CLARET, '(a) Greedy'),
        (axes[1], temp_probs, GOLD, '(b) Temperature, T=1.5'),
        (axes[2], topk_probs, TEAL, '(c) Top-k, k=3'),
        (axes[3], nucleus_probs, INK, '(d) Nucleus, p=0.9'),
    ]

    for ax, active, color, title in panels:
        ax.bar(x, base_probs, width, facecolor='none', edgecolor=RULE, linewidth=1.2, zorder=1)
        ax.bar(x, active, width * 0.55, color=color, edgecolor=INK, linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(words, rotation=40, ha='right', fontsize=FONT_SIZE-2)
        ax.set_title(title, fontsize=FONT_SIZE-1, fontweight='bold')
        ax.set_ylim(0, 0.62)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('Probability', fontsize=FONT_SIZE-1)

    fig.suptitle('Sampling Strategies Reshape the Same Distribution: "the cat sat on the ___"',
                 fontsize=FONT_SIZE+1, fontweight='bold')
    fig.text(0.5, 0.02,
             'Outline: original distribution. Filled: surviving probability. '
             'Indicative toy vocabulary, not measured model output.',
             ha='center', fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    plt.savefig('figures/sampling_strategies.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: sampling_strategies.pdf")


if __name__ == "__main__":
    create_sampling_strategies()
