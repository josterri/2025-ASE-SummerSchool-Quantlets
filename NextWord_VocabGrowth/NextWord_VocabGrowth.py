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


def create_vocab_growth():
    """Possible n-grams explode; the corpus cannot keep up."""
    d = _desks()
    V = d["corpus"]["vocabulary"]
    tokens = d["corpus"]["tokens"]
    orders = [1, 2, 3, 4]
    possible = [V ** n for n in orders]
    observed = [min(tokens - n + 1, V ** n) for n in orders]

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.plot(orders, possible, marker="o", color=CLARET,
            label="possible n-grams (V = {})".format(V))
    ax.plot(orders, observed, marker="s", color=TEAL,
            label="at most observed ({} tokens)".format(tokens))
    ax.fill_between(orders, observed, possible, color=CLARET, alpha=0.08)
    ax.set_yscale("log")
    ax.set_xticks(orders)
    ax.set_xlabel("n-gram order")
    ax.set_ylabel("count, log scale")
    ax.set_title("The counting model runs out of data")
    ax.legend(loc="upper left")
    ax.annotate("every n-gram in here\nis never seen even once",
                xy=(2.75, possible[2] / 300), fontsize=FONT_SIZE - 1,
                color=CLARET)
    plt.tight_layout()
    plt.savefig("figures/vocab_growth.pdf")
    plt.close()
    print("Created: vocab_growth.pdf")


if __name__ == "__main__":
    create_vocab_growth()
