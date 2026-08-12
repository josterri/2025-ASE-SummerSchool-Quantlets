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
Generate charts for Hour 3: The Modern Era (BERT, GPT, Scaling)
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


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


def create_pretrain_finetune_flow():
    """Chart 17: Pre-train once, adapt many times"""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    corpus_box = FancyBboxPatch((3.2, 4.9), 3.6, 0.7, boxstyle="round,pad=0.05",
                                 facecolor=PAPER, edgecolor=SOFT, linewidth=1.2)
    ax.add_patch(corpus_box)
    ax.text(5, 5.25, 'Generic text corpus', ha='center', va='center',
            fontsize=FONT_SIZE-1, color=INK)

    ax.annotate('', xy=(5, 4.55), xytext=(5, 4.9),
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.3))

    pretrain_box = FancyBboxPatch((2.7, 3.55), 4.6, 1.0, boxstyle="round,pad=0.05",
                                   facecolor=CLARET, edgecolor=INK, linewidth=1.4)
    ax.add_patch(pretrain_box)
    ax.text(5, 4.05, 'Pre-training:\none big, expensive run', ha='center', va='center',
            fontsize=FONT_SIZE, color=PAPER, fontweight='bold')

    ax.annotate('', xy=(5, 3.35), xytext=(5, 3.55),
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.3))

    base_box = FancyBboxPatch((3.5, 2.7), 3.0, 0.65, boxstyle="round,pad=0.05",
                               facecolor=TEAL, edgecolor=INK, linewidth=1.4)
    ax.add_patch(base_box)
    ax.text(5, 3.025, 'Base model', ha='center', va='center',
            fontsize=FONT_SIZE, color=PAPER, fontweight='bold')

    adaptations = ['Chat\nassistant', 'Code\ncompletion', 'Summarization', 'Sentiment\nclassifier']
    xs = np.linspace(1.3, 8.7, len(adaptations))
    for x, name in zip(xs, adaptations):
        ax.annotate('', xy=(x, 1.65), xytext=(5, 2.7),
                    arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.0, alpha=0.8))
        box = FancyBboxPatch((x-0.75, 1.0), 1.5, 0.65, boxstyle="round,pad=0.04",
                              facecolor=PAPER, edgecolor=GOLD, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, 1.325, name, ha='center', va='center', fontsize=FONT_SIZE-2, color=INK)

    ax.text(5, 0.4, 'Each branch: a cheap adaptation that reuses the pre-trained base',
            ha='center', fontsize=FONT_SIZE-1, style='italic', color=SOFT)

    ax.set_title('Pre-train Once, Adapt Many Times',
                fontsize=FONT_SIZE+1, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/pretrain_finetune_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: pretrain_finetune_flow.pdf")


if __name__ == "__main__":
    create_pretrain_finetune_flow()
