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
Generate charts for Hour 1: The Task (N-grams and Early Methods)
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


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


import json as _json
from pathlib import Path as _Path


def _desks():
    return _json.loads((_Path("figures_src") / "toy_desk_numbers.json").read_text())


def create_markov_context_window():
    """How much of the sentence each generation is allowed to see."""
    d = _desks()
    sentence = ["The", "cat", "sat", "on", "the"]
    rows = [
        ("Generation 1 unigram", 0, d["desk1_unigram"]["p_mat"] * 100),
        ("Generation 2 bigram", 1, d["desk2_bigram"]["p_mat"] * 100),
        ("Generation 3 4-gram", 3, d["desk3_fourgram"]["p_mat"] * 100),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    for r, (label, visible, pmat) in enumerate(rows):
        y = len(rows) - r
        for i, w in enumerate(sentence):
            seen = i >= len(sentence) - visible
            ax.add_patch(plt.Rectangle(
                (i, y - 0.30), 0.92, 0.60,
                facecolor=PAPER if seen else "none",
                edgecolor=INK if seen else RULE,
                linewidth=1.1 if seen else 0.7))
            ax.text(i + 0.46, y, w, ha="center", va="center",
                    fontsize=FONT_SIZE, color=INK if seen else RULE)
        ax.add_patch(plt.Rectangle((len(sentence), y - 0.30), 0.92, 0.60,
                                   facecolor="none", edgecolor=CLARET,
                                   linewidth=1.4))
        ax.text(len(sentence) + 0.46, y, "____", ha="center", va="center",
                fontsize=FONT_SIZE, color=CLARET)
        ax.text(-0.2, y, label, ha="right", va="center", fontsize=FONT_SIZE)
        ax.text(len(sentence) + 1.15, y, "P(mat) = {:.1f}%".format(pmat),
                ha="left", va="center", fontsize=FONT_SIZE,
                fontweight="bold", color=CLARET)
    ax.set_xlim(-4.4, len(sentence) + 3.4)
    ax.set_ylim(0.35, len(rows) + 0.75)
    ax.axis("off")
    ax.set_title("Admit more context and the blank sharpens", pad=6)
    plt.tight_layout()
    plt.savefig("figures/markov_context_window.pdf")
    plt.close()
    print("Created: markov_context_window.pdf")


if __name__ == "__main__":
    create_markov_context_window()
