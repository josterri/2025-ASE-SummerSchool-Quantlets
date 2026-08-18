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


def create_bert_vs_gpt():
    """Chart 14: BERT bidirectional mask vs GPT causal mask"""
    # ROOM_SLOT_CAPTIONED_IN, not ROOM_SLOT_IN: this saved at 165.6pt against a
    # 153.1pt budget. The two panel labels are the caption, because the room
    # pass took them off the panels themselves.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=room_size(*room_pick((7, 3.3), ROOM_SLOT_CAPTIONED_IN)))

    # Room edition: three tokens rather than six. Two panels across is affordable
    # only because these two share their tick labels, and the binding constraint
    # turned out to be the ROW labels, not the columns: a panel gets about 90pt
    # of height once the tick labels, the axis label and tight_layout's padding
    # are paid for, and four rows in 90pt stand 17pt apart carrying 17pt type.
    # Three rows still show a full block against a staircase, which is the whole
    # comparison.
    tokens = room_pick(['The', 'cat', 'sat', 'on', 'the', '___'],
                       ['The', 'cat', '___'])
    n = len(tokens)

    bidirectional_mask = np.ones((n, n))
    causal_mask = np.tril(np.ones((n, n)))

    # The room pass lifts every axes title onto the slide, so in the room edition
    # the x label is the only thing left naming a panel. It therefore carries the
    # model's name as well as what that model may look at.
    panels = [
        (ax1, bidirectional_mask, sequential_cmap(TEAL), 'BERT: bidirectional',
         room_pick('sees every token, both directions', 'BERT: all tokens')),
        (ax2, causal_mask, sequential_cmap(CLARET), 'GPT: causal',
         room_pick('sees only earlier tokens', 'GPT: earlier only')),
    ]

    for ax, mask, cmap, title, xlabel in panels:
        # Square cells hold each panel down to the height of the slot, which
        # leaves four tick labels 30pt apart carrying 31pt of ink. The mask is a
        # pattern, not a measurement, so stretching the cells costs nothing and
        # buys the labels their gap.
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, aspect=room_pick('equal', 'auto'))
        ax.grid(False)  # gridlines must not cross heatmap cells
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        # Never tilted in the room edition: a 45 degree tick label is how a chart
        # crams in more categories than fit, and it is the one thing the room gate
        # refuses outright.
        ax.set_xticklabels(tokens, rotation=room_pick(45, 0),
                           ha=room_pick('right', 'center'), fontsize=FONT_SIZE-2)
        # And only the left panel is labelled down the side: the two panels carry
        # the same tokens in the same order, which is what makes two panels
        # affordable here at all.
        ax.set_yticklabels(room_pick(tokens, tokens if ax is ax1 else [''] * n),
                           fontsize=FONT_SIZE-2)
        ax.set_title(title, fontsize=FONT_SIZE, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE-2)
        for k in range(n + 1):
            ax.axhline(y=k-0.5, color=RULE, linewidth=0.5, alpha=0.6)
            ax.axvline(x=k-0.5, color=RULE, linewidth=0.5, alpha=0.6)

    ax1.set_ylabel('Token attending', fontsize=FONT_SIZE-1)

    heading = 'Same Sentence, Two Masks: Full Context vs Past Only'
    if ROOM:
        # The room pass blanks a suptitle, it does not remove it, and
        # tight_layout still reserves the band a suptitle would occupy: measured,
        # 21pt of a 176pt figure, which is a fifth of these two panels. Every
        # other chart in this generator titles its AXES, which the same pass
        # blanks before tight_layout runs, so they pay nothing.
        #
        # A figure-level text is carried by the same rule at the same threshold,
        # 50 characters against a minimum of 25, so the room slide is required to
        # print exactly what it was required to print before. It is not in
        # tight_layout's model at all, so the band comes back to the panels,
        # which go from 68pt to 89pt tall. That is the difference between a 17pt
        # row label in a 23pt row and one in a 30pt row.
        fig.text(0.5, 0.99, heading, ha='center', va='top',
                 fontsize=FONT_SIZE+1, fontweight='bold')
    else:
        fig.suptitle(heading, fontsize=FONT_SIZE+1, fontweight='bold')

    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath('bert_vs_gpt'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: bert_vs_gpt.pdf")


if __name__ == "__main__":
    create_bert_vs_gpt()
