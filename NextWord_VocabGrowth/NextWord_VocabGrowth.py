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


def create_vocab_growth():
    """Possible n-grams explode; the corpus cannot keep up."""
    d = _desks()
    V = d["corpus"]["vocabulary"]
    tokens = d["corpus"]["tokens"]
    orders = [1, 2, 3, 4]
    possible = [V ** n for n in orders]
    observed = [min(tokens - n + 1, V ** n) for n in orders]

    # ROOM_SLOT_CAPTIONED_IN, not ROOM_SLOT_IN: this chart saved at 170.2pt of a
    # 202.3pt band, which leaves 0.8 of a line once the badge is placed, and the
    # caption naming the two curves is the one thing it cannot do without. The
    # pilot frame overflowed by 68pt and the caption went off the page, so the
    # built slide was two unlabelled lines.
    fig, ax = plt.subplots(
        figsize=room_size(*room_pick((6.2, 3.3), ROOM_SLOT_CAPTIONED_IN)))
    # Room edition: NO legend, and shortening the entries was not enough.
    # Round 1 cut them to "possible, V=123" and "at most 342 seen", and the
    # opaque frame added centrally then stopped the claret line running through
    # the words. What it could not stop is the size: this axes is 88pt tall, and
    # two entries at 16pt with one line of padding stand about 50pt whatever
    # they say. The box covered the claret curve from n=1 to n=3.5, which is the
    # widening gap the whole chart is about, so the key was sitting on the
    # finding. Both names go on the room slide.
    possible_kw = ({} if ROOM else
                   {"label": "possible n-grams (V = {})".format(V)})
    observed_kw = ({} if ROOM else
                   {"label": "at most observed ({} tokens)".format(tokens)})
    ax.plot(orders, possible, marker="o", color=CLARET, **possible_kw)
    ax.plot(orders, observed, marker="s", color=TEAL, **observed_kw)
    ax.fill_between(orders, observed, possible, color=CLARET, alpha=0.08)
    ax.set_yscale("log")
    ax.set_xticks(orders)
    ax.set_xlabel("n-gram order")
    ax.set_ylabel("count, log scale")
    ax.set_title("The counting model runs out of data")
    if not ROOM:
        ax.legend(loc="upper left")
    # Left exactly as it was, and that is a decision rather than an omission.
    # Three room placements were tried and all three put this label across a
    # curve: text over a LINE, which check_room_charts.py cannot see, because it
    # compares text against text. The room axes is about 90pt tall and 250pt
    # wide; the label is 20pt by 170pt set on one line and 40pt by 100pt on two,
    # and neither shape has a clear region on a chart whose whole subject is two
    # series diverging across the plot. Each attempt was found by rendering the
    # frame and looking at it, which is the only thing that can see this.
    #
    # So the strip pass carries it to the slide, as it already did, and
    # figures_src/room_carried_spoken.json classifies it spoken: the caption
    # names both series, the widening gap is the picture, and this sentence says
    # what the picture already shows.
    ax.annotate("every n-gram in here\nis never seen even once",
                xy=(2.75, possible[2] / 300), fontsize=FONT_SIZE - 1,
                color=CLARET)
    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath("vocab_growth"))
    plt.close()
    print("Created: vocab_growth.pdf")


if __name__ == "__main__":
    create_vocab_growth()
