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
                     SOFT, PAPER, RULE, sequential_cmap, diverging_cmap,
                     figpath, room_size, base_font_size, room_pt, room_pick, ROOM,
                     ROOM_SLOT_IN, ROOM_SLOT_CAPTIONED_IN)
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


import json as _json
from pathlib import Path as _Path


def _desks():
    return _json.loads((_Path("figures_src") / "toy_desk_numbers.json").read_text())


def create_markov_context_window():
    """How much of the sentence each generation is allowed to see."""
    d = _desks()
    sentence = ["The", "cat", "sat", "on", "the"]
    # Room edition: short row labels. The full names are 20 characters, which at
    # 19pt is 190pt of a 417pt slide claimed by the margin alone, and the six word
    # cells then have 33pt each for text that inks 35.
    rows = [
        (room_pick("Generation 1 unigram", "Gen 1"), 0, d["desk1_unigram"]["p_mat"] * 100),
        (room_pick("Generation 2 bigram", "Gen 2"), 1, d["desk2_bigram"]["p_mat"] * 100),
        (room_pick("Generation 3 4-gram", "Gen 3"), 3, d["desk3_fourgram"]["p_mat"] * 100),
    ]
    fig, ax = plt.subplots(figsize=room_size(*room_pick((8.0, 2.9), ROOM_SLOT_IN)))
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
        if ROOM:
            # The blank drawn as a rule rather than as four underscores. An
            # underscore is a glyph that sits ON the baseline's floor, so
            # va="center" centres the font's box and leaves the ink at the
            # bottom of it: at 19pt the four of them printed along the lower
            # edge of the claret rectangle and out past its right side, which
            # reads as a stray line rather than as a blank to be filled. A
            # segment is placed by its own coordinates and cannot drift with the
            # font.
            ax.plot([len(sentence) + 0.16, len(sentence) + 0.76], [y, y],
                    color=CLARET, linewidth=2.2, solid_capstyle="butt")
        else:
            ax.text(len(sentence) + 0.46, y, "____", ha="center", va="center",
                    fontsize=FONT_SIZE, color=CLARET)
        ax.text(-0.2, y, label, ha="right", va="center", fontsize=FONT_SIZE)
        ax.text(len(sentence) + 1.15, y,
                room_pick("P(mat) = {:.1f}%", "{:.1f}%").format(pmat),
                ha="left", va="center", fontsize=FONT_SIZE,
                fontweight="bold", color=CLARET)
    if ROOM:
        # "P(mat) = " printed three times is 210pt of a 417pt slide spent saying
        # the same thing twice more, and it is the reason the word cells had no
        # width left. It becomes a column heading instead: nothing is lost, and
        # each row carries only its own number.
        ax.text(len(sentence) + 1.15, len(rows) + 0.9, "P(mat)",
                ha="left", va="center", fontsize=FONT_SIZE, color=CLARET)
    # Room edition: narrower margins, which is how a box in DATA units is given
    # room for type measured in POINTS. The axes keeps the width tight_layout
    # gives it whatever the limits are, so cutting the span from 9.6 units to
    # 7.9 buys every word cell about a fifth more width without touching the
    # cell or the type. At 2.6 and -2.0 a cell was 35pt holding 33pt of "The".
    # The row labels and the percentages are drawn outside these limits on
    # purpose; the saved page grows to include them, and growing is free while
    # the type is 19pt against a 16pt floor.
    ax.set_xlim(room_pick(-4.4, -1.25), len(sentence) + room_pick(3.4, 1.65))
    ax.set_ylim(0.35, len(rows) + room_pick(0.75, 1.15))
    ax.axis("off")
    ax.set_title("Admit more context and the blank sharpens", pad=6)
    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath("markov_context_window"))
    plt.close()
    print("Created: markov_context_window.pdf")


if __name__ == "__main__":
    create_markov_context_window()
