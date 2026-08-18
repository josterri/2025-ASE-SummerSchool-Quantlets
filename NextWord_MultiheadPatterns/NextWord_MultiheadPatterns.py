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
                     SOFT, PAPER, RULE, sequential_cmap, diverging_cmap,
                     figpath, room_size, base_font_size, room_pt, room_pick, ROOM)
use_course_style()
sns.set_palette(CYCLE)


# Common settings
FONT_SIZE = base_font_size(8)
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
    # Four 6x6 grids on a room slide give each tick label 22pt of width and each
    # label is 28pt wide, so the room draws a four-word sentence in two panels.
    # The dependency pairs are picked with the sentence: they are indices into
    # it, and a pair naming word 5 of a four-word sentence is an IndexError, not
    # a layout defect.
    words = room_pick(['The', 'cat', 'sat', 'on', 'the', 'mat'],
                      ['The', 'cat', 'sat', 'mat'])
    n = len(words)

    def normalise(mat):
        return mat / mat.sum(axis=1, keepdims=True)

    idx = np.arange(n)
    head_local = normalise(np.exp(-np.abs(idx[:, None] - idx[None, :])))

    syntax = np.full((n, n), 0.04)
    for i, j in room_pick([(0, 1), (2, 1), (3, 2), (5, 3), (4, 5)],
                          [(0, 1), (2, 1), (3, 2)]):
        syntax[i, j] = 0.55
    head_syntax = normalise(syntax)

    semantic = np.full((n, n), 0.04)
    for i, j in room_pick([(1, 5), (5, 1), (0, 4), (4, 0)],
                          [(1, 3), (3, 1)]):
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
    room_heads = [
        ('Head 1: local', head_local),
        ('Head 2: syntax', head_syntax),
    ]

    # 2.25in: this figure owes three keys, so its frame carries two caption
    # lines and the budget is 128.1pt rather than 180.2pt.
    fig, axes = plt.subplots(*room_pick((2, 2), (1, 2)),
                             figsize=room_size(*room_pick((7.4, 3.5), (5.79, 2.31))))
    cmap = sequential_cmap(TEAL)
    for position, (ax, (title, mat)) in enumerate(zip(axes.flat, room_pick(heads, room_heads))):
        # 'auto' fills the slot; 'equal' would leave two square islands with
        # dead space either side and tick labels 22pt apart.
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max(), aspect=room_pick('equal', 'auto'))
        ax.grid(False)  # gridlines must not cross heatmap cells
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(words, rotation=room_pick(45, 0),
                           ha=room_pick('right', 'center'), fontsize=FONT_SIZE - 3)
        # Row labels on the left panel only in the room: the second copy would
        # sit in the 30pt gap between the two panels.
        ax.set_yticklabels(room_pick(words, words if position == 0 else [''] * n),
                           fontsize=FONT_SIZE - 3)
        ax.set_title(title, fontsize=FONT_SIZE - 1, fontweight='bold')
        # The titles are stripped by the room pass, so the same words are set as
        # an axis label, which the pass keeps.
        ax.set_xlabel(room_pick('', title), fontsize=FONT_SIZE - 3)

    fig.text(0.5, 0.01, room_pick('Same sentence, four heads: darker means stronger attention weight.',
                                  'Same sentence, two heads: darker means stronger attention weight.'),
              ha='center', fontsize=FONT_SIZE - 2, style='italic', color=SOFT)

    # As in create_one_hot_vs_dense: the strip is for the caption, the room pass
    # takes the caption to the slide, so the room reclaims the strip.
    plt.tight_layout(rect=room_pick((0, 0.06, 1, 1), (0, 0.0, 1, 1)))
    plt.savefig(figpath('multihead_patterns'))
    plt.close()
    print("Created: multihead_patterns.pdf")


if __name__ == "__main__":
    create_multihead_patterns()
