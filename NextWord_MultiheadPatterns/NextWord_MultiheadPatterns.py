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


def create_multihead_patterns():
    """Different heads specialise: local, syntax, semantic, broad context."""
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n = len(words)

    def normalise(mat):
        return mat / mat.sum(axis=1, keepdims=True)

    idx = np.arange(n)
    head_local = normalise(np.exp(-np.abs(idx[:, None] - idx[None, :])))

    syntax = np.full((n, n), 0.04)
    for i, j in [(0, 1), (2, 1), (3, 2), (5, 3), (4, 5)]:
        syntax[i, j] = 0.55
    head_syntax = normalise(syntax)

    semantic = np.full((n, n), 0.04)
    for i, j in [(1, 5), (5, 1), (0, 4), (4, 0)]:
        semantic[i, j] = 0.5
    head_semantic = normalise(semantic)

    np.random.seed(3)
    broad = np.full((n, n), 1.0) + np.random.uniform(0, 0.15, size=(n, n))
    broad[:, 0] += 0.4
    head_broad = normalise(broad)

    heads = [
        ('Head 1: local context', head_local),
        ('Head 2: syntax', head_syntax),
        ('Head 3: semantic', head_semantic),
        ('Head 4: broad context', head_broad),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 3.5))
    cmap = sequential_cmap(TEAL)
    for ax, (title, mat) in zip(axes.flat, heads):
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max(), aspect='equal')
        ax.grid(False)  # gridlines must not cross heatmap cells
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(words, rotation=45, ha='right', fontsize=FONT_SIZE - 3)
        ax.set_yticklabels(words, fontsize=FONT_SIZE - 3)
        ax.set_title(title, fontsize=FONT_SIZE - 1, fontweight='bold')

    fig.text(0.5, 0.01, 'Same sentence, four heads: darker means stronger attention weight.',
              ha='center', fontsize=FONT_SIZE - 2, style='italic', color=SOFT)

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig('figures/multihead_patterns.pdf')
    plt.close()
    print("Created: multihead_patterns.pdf")


if __name__ == "__main__":
    create_multihead_patterns()
