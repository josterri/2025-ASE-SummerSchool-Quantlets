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


def create_bert_vs_gpt():
    """Chart 14: BERT bidirectional mask vs GPT causal mask"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.3))

    tokens = ['The', 'cat', 'sat', 'on', 'the', '___']
    n = len(tokens)

    bidirectional_mask = np.ones((n, n))
    causal_mask = np.tril(np.ones((n, n)))

    panels = [
        (ax1, bidirectional_mask, sequential_cmap(TEAL), 'BERT: bidirectional',
         'sees every token, both directions'),
        (ax2, causal_mask, sequential_cmap(CLARET), 'GPT: causal',
         'sees only earlier tokens'),
    ]

    for ax, mask, cmap, title, xlabel in panels:
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        ax.grid(False)  # gridlines must not cross heatmap cells
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=FONT_SIZE-2)
        ax.set_yticklabels(tokens, fontsize=FONT_SIZE-2)
        ax.set_title(title, fontsize=FONT_SIZE, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE-2)
        for k in range(n + 1):
            ax.axhline(y=k-0.5, color=RULE, linewidth=0.5, alpha=0.6)
            ax.axvline(x=k-0.5, color=RULE, linewidth=0.5, alpha=0.6)

    ax1.set_ylabel('Token attending', fontsize=FONT_SIZE-1)

    fig.suptitle('Same Sentence, Two Masks: Full Context vs Past Only',
                 fontsize=FONT_SIZE+1, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/bert_vs_gpt.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: bert_vs_gpt.pdf")


if __name__ == "__main__":
    create_bert_vs_gpt()
