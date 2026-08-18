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


def create_perplexity_ladder():
    """Perplexity is 2 to the power of the entropy on the generation card."""
    d = _desks()
    bits = [d["desk1_unigram"]["entropy_bits"],
            d["desk2_bigram"]["entropy_bits"],
            d["desk3_fourgram"]["entropy_bits"]]
    ppl = [2 ** b for b in bits]
    # Room edition: the entropy moves from inside the bar to the tick label. It
    # cannot stay inside, because the third bar is 2.9 units tall and its
    # mid-height sits 3pt under its own value label; and it cannot simply go,
    # because H is the number the perplexity is two to the power of.
    labels = room_pick(
        ["Generation 1\nunigram", "Generation 2\nbigram", "Generation 3\n4-gram"],
        ["Gen {}\nH={}".format(i + 1, b) for i, b in enumerate(bits)])

    # ROOM_SLOT_CAPTIONED_IN, not ROOM_SLOT_IN: this saved at 166.3pt against a
    # 153.1pt budget. The frame heading takes the title and the caption carries
    # the toy-corpus provenance line the room edition drops below, so one
    # caption line is reserved and the chart is drawn to what is left.
    fig, ax = plt.subplots(
        figsize=room_size(*room_pick((6.0, 3.2), ROOM_SLOT_CAPTIONED_IN)))
    bars = ax.bar(labels, ppl, color=[TEAL, TEAL, CLARET],
                  edgecolor=INK, linewidth=0.6)
    for bar, pp, b in zip(bars, ppl, bits):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                "{:.1f}".format(pp), ha="center", va="bottom",
                fontsize=FONT_SIZE, fontweight="bold")
        if not ROOM:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                    "H = {} bits".format(b), ha="center", va="center",
                    fontsize=FONT_SIZE - 1, color=PAPER)
    # Rotated, so the character count is a height. Broken in two rather than cut,
    # because "2 to the H" is the whole relation between this axis and the H on
    # the tick labels below.
    ax.set_ylabel(room_pick("Perplexity, that is 2 to the power of H",
                            "Perplexity\n= 2 to the H"))
    if ROOM:
        # The unit left the tick label, where "H=5.45 bits" three times ran the
        # three labels into each other, and became the axis it belongs on.
        ax.set_xlabel("H in bits")
    ax.set_title("Effective number of choices facing each generation")
    ax.set_ylim(0, max(ppl) * 1.25)
    if not ROOM:
        # 38 characters, just under the room pass's 40-character axes threshold,
        # so it is not carried automatically. It was the widest run in the room
        # rendering at 0.87 of the page. Dropped here; the sentence goes on the
        # room slide.
        ax.text(0.98, 0.95, "computed from the toy corpus entropies",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_SIZE - 2, style="italic", color=SOFT)
    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath("perplexity_ladder"))
    plt.close()
    print("Created: perplexity_ladder.pdf")


if __name__ == "__main__":
    create_perplexity_ladder()
