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


def create_toy_corpus_distribution():
    """Generation 3: what the 4-gram model actually predicts for the blank."""
    data = _desks()
    d = data["desk3_fourgram"]
    words = [t["word"] for t in d["top"]]
    probs = [t["p"] * 100 for t in d["top"]]

    # Room edition: the full slot. At 4.9 inches the four value labels sat 8pt
    # apart and read as "15.4%15.4%"; the extra 0.9 inch buys 24pt of gap.
    # ROOM_SLOT_CAPTIONED_IN, not ROOM_SLOT_IN: the title goes in the frame
    # heading and the toy-corpus provenance line is the caption, so this saved
    # at 138.2pt against the 128.1pt two caption lines leave.
    fig, ax = plt.subplots(
        figsize=room_size(*room_pick((6.4, 3.2), ROOM_SLOT_CAPTIONED_IN)))
    colors = [CLARET if w == "mat" else TEAL for w in words]
    bars = ax.bar(words, probs, color=colors, edgecolor=INK, linewidth=0.6)
    for bar, pr in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                "{:.1f}%".format(pr), ha="center", va="bottom",
                fontsize=FONT_SIZE, fontweight="bold")
    # Rotated, so the character count is a height: 20 characters is 215pt in a
    # 176pt slot and the page grew to 190pt. Broken in two rather than cut, since
    # every word here is the condition being conditioned on.
    ax.set_ylabel(room_pick("P(word | sat on the)", "P(word |\nsat on the)"))
    ax.set_ylim(0, max(probs) * 1.25)
    ax.set_title("Generation 3: the blank predicted by counting 4-grams")
    note = "computed on the toy corpus: {} tokens, {} sentences".format(
        data["corpus"]["tokens"], data["corpus"]["sentences"])
    ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_SIZE - 2, style="italic", color=SOFT)
    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath("toy_corpus_distribution"))
    plt.close()
    print("Created: toy_corpus_distribution.pdf")


if __name__ == "__main__":
    create_toy_corpus_distribution()
