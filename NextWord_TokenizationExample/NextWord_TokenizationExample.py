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


def create_tokenization_example():
    """Chart 16: Subword tokenization example"""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    def draw_tokens(y, caption, tokens, colors):
        ax.text(0.2, y + 0.55, caption, ha='left', va='center',
                fontsize=FONT_SIZE-1, color=SOFT)
        x = 0.2
        for tok, color in zip(tokens, colors):
            width = 0.4 + 0.22 * len(tok)
            box = FancyBboxPatch((x, y - 0.3), width, 0.6, boxstyle="round,pad=0.04",
                                  facecolor=color, edgecolor=INK, linewidth=0.8)
            ax.add_patch(box)
            ax.text(x + width/2, y, tok, ha='center', va='center',
                    fontsize=FONT_SIZE-1, color=PAPER, fontweight='bold')
            x += width + 0.15

    draw_tokens(2.75, '"unbelievable": a common word yields few, familiar pieces',
                ['un', 'believ', 'able'], CYCLE[:3])
    draw_tokens(1.0, '"zyloquorix": a rare or new word fragments into more pieces',
                ['z', 'yl', 'o', 'quor', 'ix'], CYCLE)

    ax.set_title('Subword Tokenization: Rare Words Split into More Pieces',
                fontsize=FONT_SIZE+1, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/tokenization_example.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: tokenization_example.pdf")


if __name__ == "__main__":
    create_tokenization_example()
