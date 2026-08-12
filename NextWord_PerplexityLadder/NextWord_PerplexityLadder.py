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


def create_perplexity_ladder():
    """Perplexity is 2 to the power of the entropy on the generation card."""
    d = _desks()
    labels = ["Generation 1\nunigram", "Generation 2\nbigram", "Generation 3\n4-gram"]
    bits = [d["desk1_unigram"]["entropy_bits"],
            d["desk2_bigram"]["entropy_bits"],
            d["desk3_fourgram"]["entropy_bits"]]
    ppl = [2 ** b for b in bits]

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bars = ax.bar(labels, ppl, color=[TEAL, TEAL, CLARET],
                  edgecolor=INK, linewidth=0.6)
    for bar, pp, b in zip(bars, ppl, bits):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                "{:.1f}".format(pp), ha="center", va="bottom",
                fontsize=FONT_SIZE, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                "H = {} bits".format(b), ha="center", va="center",
                fontsize=FONT_SIZE - 1, color=PAPER)
    ax.set_ylabel("Perplexity, that is 2 to the power of H")
    ax.set_title("Effective number of choices facing each generation")
    ax.set_ylim(0, max(ppl) * 1.25)
    ax.text(0.98, 0.95, "computed from the toy corpus entropies",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_SIZE - 2, style="italic", color=SOFT)
    plt.tight_layout()
    plt.savefig("figures/perplexity_ladder.pdf")
    plt.close()
    print("Created: perplexity_ladder.pdf")


if __name__ == "__main__":
    create_perplexity_ladder()
