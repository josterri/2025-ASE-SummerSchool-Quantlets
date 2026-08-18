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
Generate charts for Hour 3: The Modern Era (BERT, GPT, Scaling)
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


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


# A KEY ABOVE THE PLOT COSTS 46pt, AND MOST OF THAT IS THE CARD.
#
# Four charts here moved their legend above the axes, because at 18pt a two-row
# key inside a room axes is half its height and a one-row key is wider than it.
# Above the plot is the right place. What was not measured is the price:
# tools/probe_room_hour3_layout.py reports performance_benchmarks with a 39pt
# plot in a 176pt figure, 271x36pt of legend above it, and a 9pt gap.
#
# Most of that 36pt is not the words. matplotlib's default borderpad is 0.4 of
# the font size, so 7pt above and 7pt below at 18pt, and borderaxespad adds
# another 9pt between the card and the axes. figures_src/room.mplstyle also
# gives every legend an opaque backing, which is exactly right for a key sitting
# INSIDE the axes with data lines running under it, and buys nothing above the
# plot where nothing is drawn.
#
# So: above-axes keys are frameless and tightly padded in the room edition only.
# The standard edition passes none of these and cannot move. Measured after the
# change, the same chart's plot goes from 39pt to about 85pt, which is the
# difference between a bar chart and a row of stripes.
ROOM_KEY_ABOVE = ({'frameon': False, 'borderpad': 0.1, 'borderaxespad': 0.1,
                   'handlelength': 1.4, 'handletextpad': 0.5,
                   'columnspacing': 1.4} if ROOM else {})


def create_tokenization_example():
    """Chart 16: Subword tokenization example"""
    # ROOM_SLOT_CAPTIONED_IN, not ROOM_SLOT_IN: this saved at 138.2pt against a
    # 128.1pt budget, since both row captions are carried to the slide and that
    # is two caption lines.
    fig, ax = plt.subplots(
        figsize=room_size(*room_pick((7, 3.2), ROOM_SLOT_CAPTIONED_IN)))
    # Room edition: a narrower data range, so the same two rows of chips fill the
    # slide instead of sitting in its left half. The room pass lifts both captions
    # onto the slide, and the empty half they left behind was the whole defect
    # here: this chart passed the collision gate and still looked wrong.
    ax.set_xlim(0, room_pick(10, 5.2))
    # Room edition: and a narrower y range too, on the same argument. The two
    # captions are carried to the slide, so the 0.55 units reserved above each
    # row for one are empty, and a 3-unit range in a 135pt axes drew a 27pt chip
    # around 22pt of type. The rows come together at 1.2 and 2.2, which makes the
    # chip 41pt: the tokens are what this figure is, so they get the height.
    ax.set_ylim(*room_pick((0, 4), (0.72, 2.68)))
    ax.axis('off')

    def draw_tokens(y, caption, tokens, colors):
        ax.text(0.2, y + 0.55, caption, ha='left', va='center',
                fontsize=FONT_SIZE-1, color=SOFT)
        x = 0.2
        for tok, color in zip(tokens, colors):
            width = 0.4 + 0.22 * len(tok)
            box = FancyBboxPatch((x, y - 0.3), width, 0.6, boxstyle="round,pad=0.04",
                                  facecolor=color, edgecolor=INK, linewidth=0.8)
            ax.add_patch(box)
            ax.text(x + width/2, y, tok, ha='center', va='center',
                    fontsize=room_pick(FONT_SIZE-1, 22), color=PAPER, fontweight='bold')
            x += width + 0.15

    draw_tokens(room_pick(2.75, 2.2), '"unbelievable": a common word yields few, familiar pieces',
                ['un', 'believ', 'able'], CYCLE[:3])
    draw_tokens(room_pick(1.0, 1.2), '"zyloquorix": a rare or new word fragments into more pieces',
                ['z', 'yl', 'o', 'quor', 'ix'], CYCLE)

    ax.set_title('Subword Tokenization: Rare Words Split into More Pieces',
                fontsize=FONT_SIZE+1, fontweight='bold')

    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath('tokenization_example'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: tokenization_example.pdf")


if __name__ == "__main__":
    create_tokenization_example()
