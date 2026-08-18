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


def create_pretrain_finetune_flow():
    """Chart 17: Pre-train once, adapt many times"""
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7, 3.4), ROOM_SLOT_IN)))
    ax.set_xlim(0, 10)
    # Room edition: three levels in 2.45 inches rather than four. A level costs
    # about 30pt of box plus 9pt of arrow at 19pt type, and four of them plus the
    # caption do not fit; every box below is therefore taller and wider in the
    # room branch, on a shorter ladder.
    ax.set_ylim(0, room_pick(6, 4.75))
    ax.axis('off')

    if not ROOM:
        # The level the room edition gives up. "Generic text corpus" is the one
        # box the box below it already implies, and the sentence goes on the
        # room slide.
        corpus_box = FancyBboxPatch((3.2, 4.9), 3.6, 0.7, boxstyle="round,pad=0.05",
                                     facecolor=PAPER, edgecolor=SOFT, linewidth=1.2)
        ax.add_patch(corpus_box)
        ax.text(5, 5.25, 'Generic text corpus', ha='center', va='center',
                fontsize=FONT_SIZE-1, color=INK)

        ax.annotate('', xy=(5, 4.55), xytext=(5, 4.9),
                    arrowprops=dict(arrowstyle='->', color=INK, lw=1.3))

    pre_x, pre_y, pre_w, pre_h = room_pick((2.7, 3.55, 4.6, 1.0),
                                           (1.6, 3.1, 6.8, 1.35))
    pretrain_box = FancyBboxPatch((pre_x, pre_y), pre_w, pre_h, boxstyle="round,pad=0.05",
                                   facecolor=CLARET, edgecolor=INK, linewidth=1.4)
    ax.add_patch(pretrain_box)
    # Room edition: the same six words, broken so the longest line is 18
    # characters rather than 22. At 19pt "one big, expensive run" is 0.66 of the
    # page on its own, over the 0.60 a single run may occupy; nothing is dropped,
    # the line just breaks in a different place.
    ax.text(5, pre_y + pre_h / 2,
            room_pick('Pre-training:\none big, expensive run',
                      'One big, expensive\npre-training run'),
            ha='center', va='center',
            fontsize=FONT_SIZE, color=PAPER, fontweight='bold')

    ax.annotate('', xy=(5, pre_y - room_pick(0.2, 0.25)), xytext=(5, pre_y),
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.3))

    base_x, base_y, base_w, base_h = room_pick((3.5, 2.7, 3.0, 0.65),
                                               (3.3, 2.0, 3.4, 0.85))
    base_box = FancyBboxPatch((base_x, base_y), base_w, base_h, boxstyle="round,pad=0.05",
                               facecolor=TEAL, edgecolor=INK, linewidth=1.4)
    ax.add_patch(base_box)
    ax.text(5, base_y + base_h / 2, 'Base model', ha='center', va='center',
            fontsize=FONT_SIZE, color=PAPER, fontweight='bold')

    # Room edition: one word per branch. Four two-line labels at 17pt overprint
    # each other; the four branches themselves are the point and all four stay.
    adaptations = room_pick(['Chat\nassistant', 'Code\ncompletion', 'Summarization',
                             'Sentiment\nclassifier'],
                            ['Chat', 'Code', 'Summary', 'Sentiment'])
    xs = np.linspace(room_pick(1.3, 1.25), room_pick(8.7, 8.75), len(adaptations))
    # Room edition: the branch box is 2.4 units rather than 2.3, and its label is
    # set at the 16pt floor rather than 17. Measured: "Sentiment" is 89pt at 17pt
    # and the box was 85pt, so the word was drawn 2pt past its own border at each
    # end. Nothing catches that. It is text over a PATCH, so the collision rule
    # sees one span and no neighbour, and the widest-run rule sees 0.21 of the
    # page. Widening alone is not enough at four boxes across ten units, so the
    # two changes are made together: 84pt of word in an 89pt box.
    ad_top, ad_y, ad_half, ad_h = room_pick((1.65, 1.0, 0.75, 0.65),
                                            (1.35, 0.5, 1.20, 0.85))
    for x, name in zip(xs, adaptations):
        ax.annotate('', xy=(x, ad_top), xytext=(5, base_y),
                    arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.0, alpha=0.8))
        box = FancyBboxPatch((x - ad_half, ad_y), ad_half * 2, ad_h, boxstyle="round,pad=0.04",
                              facecolor=PAPER, edgecolor=GOLD, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, ad_y + ad_h / 2, name, ha='center', va='center',
                fontsize=room_pick(FONT_SIZE-2, FONT_SIZE-3), color=INK)

    ax.text(5, room_pick(0.4, 0.12), 'Each branch: a cheap adaptation that reuses the pre-trained base',
            ha='center', fontsize=FONT_SIZE-1, style='italic', color=SOFT)

    ax.set_title('Pre-train Once, Adapt Many Times',
                fontsize=FONT_SIZE+1, fontweight='bold')

    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath('pretrain_finetune_flow'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: pretrain_finetune_flow.pdf")


if __name__ == "__main__":
    create_pretrain_finetune_flow()
