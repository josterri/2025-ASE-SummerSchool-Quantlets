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
Generate charts for Hour 2: Smarter Prediction (RNNs to Transformers)
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch


import sys; sys.path.insert(0, 'figures_src')
from palette import (use_course_style, CYCLE, CLARET, TEAL, GOLD, INK,
                     SOFT, PAPER, RULE, sequential_cmap, diverging_cmap,
                     figpath, room_size, base_font_size, room_pt, room_pick, ROOM)
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


def create_qkv_flow():
    """One token, three learned projections, scaled dot-product attention."""
    # 2.06in for the same reason as create_lstm_gates: the frame under this one
    # carries a caption line, so the budget is 153.1pt and not 180.2pt.
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7.4, 3.4), (5.79, 2.06))))
    if ROOM:
        # See create_lstm_gates: an axis('off') diagram loses the slide to
        # tight_layout unless its axes is placed by hand.
        ax.remove()
        ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.3)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(cx, cy, w, h, text, face, textcolor=INK):
        patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.06', facecolor=face,
                                edgecolor=INK, linewidth=room_pick(1.2, 2.0), zorder=3)
        ax.add_patch(patch)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=FONT_SIZE - 1,
                color=textcolor, fontweight='bold', zorder=4)

    def arrow(x0, y0, x1, y1, color=INK):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=room_pick(1.4, 2.4)))

    # The room re-lays the same five stages rather than shrinking them, and the
    # column widths below are MEASURED: tools/probe_room_text_extent.py reads
    # each label out of the built PDF, because a box is sized in data units and
    # the type in it is sized in points, so the two only meet after layout. At
    # the full slot width one unit is 31.35pt, and the four labels that decide
    # the layout are 'Token' 1.87, 'W_Q' 1.41, 'Query' 1.94 and 'Dot-product'
    # 3.91 units. Every box below is its own label plus about 0.45 units, the
    # four gaps are 0.65 units each so an arrow is 20pt of visible shaft rather
    # than a head stuck to a box, and the row sums to exactly 13.3.
    #
    # "embedding" is the one word that had to go, and it went because it does
    # not fit: 3.60 units of ink in a 3.0 unit box, printing past both edges.
    # Keeping it would have cost the whole 'Dot-product' box, and the dot
    # product is the mechanism this frame exists to show. It goes to the slide.
    box(room_pick(1.1, 1.275), 3, room_pick(1.7, 2.35), room_pick(1.1, 1.4),
        room_pick('Token\nembedding x', 'Token'), PAPER)

    wx = room_pick(4.3, 4.025)
    ww, wh = room_pick(1.5, 1.85), room_pick(0.9, 1.4)
    box(wx, 5, ww, wh, r'$W_Q$', CLARET, textcolor=PAPER)
    box(wx, 3, ww, wh, r'$W_K$', GOLD, textcolor=PAPER)
    box(wx, 1, ww, wh, r'$W_V$', TEAL, textcolor=PAPER)
    arrow(room_pick(1.95, 2.45), room_pick(3.35, 3.4), room_pick(3.55, 3.1), room_pick(4.8, 4.7), color=CLARET)
    arrow(room_pick(1.95, 2.45), 3.0, room_pick(3.55, 3.1), 3.0, color=GOLD)
    arrow(room_pick(1.95, 2.45), room_pick(2.65, 2.6), room_pick(3.55, 3.1), room_pick(1.2, 1.3), color=TEAL)

    qx = room_pick(7.2, 6.775)
    qw, qh = room_pick(1.5, 2.35), room_pick(0.9, 1.4)
    box(qx, 5, qw, qh, room_pick('Query q', 'Query'), CLARET, textcolor=PAPER)
    box(qx, 3, qw, qh, room_pick('Key k', 'Key'), GOLD, textcolor=PAPER)
    box(qx, 1, qw, qh, room_pick('Value v', 'Value'), TEAL, textcolor=PAPER)
    arrow(room_pick(5.05, 4.95), 5, room_pick(6.45, 5.6), 5, color=CLARET)
    arrow(room_pick(5.05, 4.95), 3, room_pick(6.45, 5.6), 3, color=GOLD)
    arrow(room_pick(5.05, 4.95), 1, room_pick(6.45, 5.6), 1, color=TEAL)

    # 4.35 units wide and lower than the standard box, because "Dot-product" is
    # 123pt of ink at 18pt and the standard 2.0-unit box holds 63pt of it. The
    # words are kept whole; the box is what changes size.
    box(room_pick(10.1, 10.775), room_pick(4, 3.6), room_pick(2.0, 4.35), room_pick(1.3, 2.1),
        room_pick('Scaled dot-product\n+ softmax', 'Dot-product\n+ softmax'), PAPER)
    arrow(room_pick(7.95, 7.95), room_pick(4.7, 4.8), room_pick(9.1, 8.6), room_pick(4.3, 4.3), color=CLARET)
    arrow(room_pick(7.95, 7.95), room_pick(3.3, 3.2), room_pick(9.1, 8.6), room_pick(3.7, 3.3), color=GOLD)

    box(room_pick(12.2, 10.775), room_pick(2.2, 1.0), room_pick(1.5, 2.75), room_pick(1.1, 1.4),
        'Output', SOFT, textcolor=PAPER)
    arrow(room_pick(11.0, 10.775), room_pick(3.6, 2.55), room_pick(11.5, 10.775),
          room_pick(2.5, 1.7), color=INK)
    arrow(room_pick(7.95, 7.95), room_pick(1.15, 1.0), room_pick(11.55, 9.4), room_pick(1.95, 1.0), color=TEAL)

    ax.set_title('One Input Token Becomes Query, Key and Value',
                fontsize=FONT_SIZE + 1, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figpath('qkv_flow'))
    plt.close()
    print("Created: qkv_flow.pdf")


if __name__ == "__main__":
    create_qkv_flow()
