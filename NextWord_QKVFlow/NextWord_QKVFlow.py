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


def create_qkv_flow():
    """One token, three learned projections, scaled dot-product attention."""
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.set_xlim(0, 13.3)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(cx, cy, w, h, text, face, textcolor=INK):
        patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.06', facecolor=face,
                                edgecolor=INK, linewidth=1.2, zorder=3)
        ax.add_patch(patch)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=FONT_SIZE - 1,
                color=textcolor, fontweight='bold', zorder=4)

    def arrow(x0, y0, x1, y1, color=INK):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.4))

    box(1.1, 3, 1.7, 1.1, 'Token\nembedding x', PAPER)

    box(4.3, 5, 1.5, 0.9, r'$W_Q$', CLARET, textcolor=PAPER)
    box(4.3, 3, 1.5, 0.9, r'$W_K$', GOLD, textcolor=PAPER)
    box(4.3, 1, 1.5, 0.9, r'$W_V$', TEAL, textcolor=PAPER)
    arrow(1.95, 3.35, 3.55, 4.8, color=CLARET)
    arrow(1.95, 3.0, 3.55, 3.0, color=GOLD)
    arrow(1.95, 2.65, 3.55, 1.2, color=TEAL)

    box(7.2, 5, 1.5, 0.9, 'Query q', CLARET, textcolor=PAPER)
    box(7.2, 3, 1.5, 0.9, 'Key k', GOLD, textcolor=PAPER)
    box(7.2, 1, 1.5, 0.9, 'Value v', TEAL, textcolor=PAPER)
    arrow(5.05, 5, 6.45, 5, color=CLARET)
    arrow(5.05, 3, 6.45, 3, color=GOLD)
    arrow(5.05, 1, 6.45, 1, color=TEAL)

    box(10.1, 4, 2.0, 1.3, 'Scaled dot-product\n+ softmax', PAPER)
    arrow(7.95, 4.7, 9.1, 4.3, color=CLARET)
    arrow(7.95, 3.3, 9.1, 3.7, color=GOLD)

    box(12.2, 2.2, 1.5, 1.1, 'Output', SOFT, textcolor=PAPER)
    arrow(11.0, 3.6, 11.5, 2.5, color=INK)
    arrow(7.95, 1.15, 11.55, 1.95, color=TEAL)

    ax.set_title('One Input Token Becomes Query, Key and Value',
                fontsize=FONT_SIZE + 1, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/qkv_flow.pdf')
    plt.close()
    print("Created: qkv_flow.pdf")


if __name__ == "__main__":
    create_qkv_flow()
