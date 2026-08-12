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


def create_toy_corpus_distribution():
    """Generation 3: what the 4-gram model actually predicts for the blank."""
    data = _desks()
    d = data["desk3_fourgram"]
    words = [t["word"] for t in d["top"]]
    probs = [t["p"] * 100 for t in d["top"]]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = [CLARET if w == "mat" else TEAL for w in words]
    bars = ax.bar(words, probs, color=colors, edgecolor=INK, linewidth=0.6)
    for bar, pr in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                "{:.1f}%".format(pr), ha="center", va="bottom",
                fontsize=FONT_SIZE, fontweight="bold")
    ax.set_ylabel("P(word | sat on the)")
    ax.set_ylim(0, max(probs) * 1.25)
    ax.set_title("Generation 3: the blank predicted by counting 4-grams")
    note = "computed on the toy corpus: {} tokens, {} sentences".format(
        data["corpus"]["tokens"], data["corpus"]["sentences"])
    ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_SIZE - 2, style="italic", color=SOFT)
    plt.tight_layout()
    plt.savefig("figures/toy_corpus_distribution.pdf")
    plt.close()
    print("Created: toy_corpus_distribution.pdf")


if __name__ == "__main__":
    create_toy_corpus_distribution()
