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


def create_one_hot_vs_dense():
    """Contrast a one-hot vector against a dense embedding."""
    # 1.97in: three keys, so three caption lines, so a 103.1pt budget. The two
    # panel titles and the illustrative marker are all keys and none of them
    # fits a 33-character heading, which is what buys the height back elsewhere.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=room_size(*room_pick((7.2, 3.4), (5.79, 1.97))))

    n_show = 16
    hot_index = 6
    ax1.set_xlim(-0.5, n_show + 2.5)
    # Both panels are cropped to their cells in the room, and both to the SAME
    # range: see the note on ax2 below, where unequal ranges drew the two rows of
    # cells at different heights.
    ax1.set_ylim(*room_pick((-0.3, 1.3), (-0.1, 1.4)))
    for i in range(n_show):
        is_hot = (i == hot_index)
        rect = plt.Rectangle((i, 0), 0.9, 1.0,
                              facecolor=CLARET if is_hot else PAPER,
                              edgecolor=INK if is_hot else RULE,
                              linewidth=1.4 if is_hot else 0.7)
        ax1.add_patch(rect)
        ax1.text(i + 0.45, 0.5, '1' if is_hot else '0', ha='center', va='center',
                  fontsize=FONT_SIZE - 2, color=PAPER if is_hot else SOFT)
    ax1.text(n_show + 1.0, 0.5, '...', ha='center', va='center', fontsize=FONT_SIZE,
              color=SOFT)
    ax1.axis('off')
    ax1.set_title('One-hot: a single 1 among thousands of zeros (dimension = vocabulary size)',
                  fontsize=FONT_SIZE, fontweight='bold')
    if ROOM:
        # Both titles are stripped by the room pass and printed on the slide,
        # which would leave two anonymous rows of cells. These two words are the
        # minimum that keeps the contrast readable in the figure itself.
        # 1.27, not 1.1. The cells run from 0 to 1 and this panel is 45pt tall
        # once the figure is cut to the 103.1pt a three-line caption allows, so
        # a label centred at 1.1 is 4pt above the cell tops and 19pt of type
        # deep: tools/check_room_charts.py measured it printing into four of the
        # zeros by 9.6 by 3.8pt. 1.27 clears them and is still inside the 1.4
        # ceiling, so nothing else moves.
        ax1.text(-0.4, 1.27, 'One-hot', ha='left', va='center',
                 fontsize=FONT_SIZE, color=INK, fontweight='bold')

    np.random.seed(11)
    values = np.random.uniform(-1, 1, size=8)
    if ROOM:
        # The room reads this row by colour alone, because the numbers under it
        # do not fit (see below), and half the ramp is the wrong colour for that
        # job: diverging_cmap runs teal through PAPER to claret, so a value near
        # zero is drawn in exactly the PAPER of a one-hot ZERO cell one row
        # above. The row that exists to say "every dimension carries a value"
        # read as half empty. Magnitudes are pushed out of the middle of the
        # ramp; the signs and the order are the ones the seed produced, and the
        # figure's own caption already says the values are illustrative.
        values = np.sign(values) * (0.45 + 0.55 * np.abs(values))
    n_dense = len(values)
    offset = (n_show - n_dense) / 2
    ax2.set_xlim(-0.5, n_show + 2.5)
    # No numbers under the cells in the room, so no band reserved for them, and
    # the room range is then the same 1.5 as the one-hot panel above. That is
    # the point rather than a side effect: both panels get the same height from
    # tight_layout, so an axes spanning 1.9 units drew its cells 16 per cent
    # SHORTER than the cells they are being compared with.
    ax2.set_ylim(*room_pick((-0.6, 1.3), (-0.1, 1.4)))
    cmap = diverging_cmap()
    for i, v in enumerate(values):
        rect = plt.Rectangle((offset + i, 0), 0.9, 1.0, facecolor=cmap((v + 1) / 2),
                              edgecolor=INK, linewidth=0.9)
        ax2.add_patch(rect)
        value_label = ax2.text(offset + i + 0.45, -0.18, '{:.2f}'.format(v), ha='center', va='top',
                  fontsize=FONT_SIZE - 2, color=SOFT)
        if ROOM:
            # Eight five-character numbers sit 22pt apart and are 45pt wide
            # each, so they print straight through one another. The colour ramp
            # is what carries "every dimension holds a value" on a small screen.
            value_label.remove()
    ax2.axis('off')
    ax2.set_title('Dense embedding: every dimension carries a meaningful value (dimension = a few hundred)',
                  fontsize=FONT_SIZE, fontweight='bold')
    if ROOM:
        ax2.text(-0.4, 1.27, 'Dense', ha='left', va='center',
                 fontsize=FONT_SIZE, color=INK, fontweight='bold')

    fig.text(0.5, 0.01, 'Illustrative dimensions and values, not to scale.', ha='center',
              fontsize=FONT_SIZE - 2, style='italic', color=SOFT)

    # The reserved strip at the bottom is for the caption above. The room pass
    # blanks that caption and moves it to the slide, so in the room the strip
    # holds nothing and is 9pt of a 176pt figure spent on white. The default
    # inter-panel gap goes the same way: both panels have axis('off'), so there
    # are no tick labels for tight_layout to be reserving 1.08 em against.
    layout_kw = {"h_pad": 0.4} if ROOM else {}
    plt.tight_layout(rect=room_pick((0, 0.05, 1, 1), (0, 0.0, 1, 1)), **layout_kw)
    plt.savefig(figpath('one_hot_vs_dense'))
    plt.close()
    print("Created: one_hot_vs_dense.pdf")


if __name__ == "__main__":
    create_one_hot_vs_dense()
