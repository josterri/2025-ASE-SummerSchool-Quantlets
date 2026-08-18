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


def create_inference_compute_scaling():
    """Chart 13: Inference-time compute scaling (indicative)"""
    # ROOM_SLOT_IN, and the way back to it is worth recording because the
    # measurement that argued for the smaller slot was correct and the
    # conclusion drawn from it was not.
    #
    # At ROOM_SLOT_CAPTIONED_IN this saved at 128.6pt against the 128.1pt a
    # two-line caption leaves, and no shorter canvas moved it: 1.95, 1.90 and
    # 1.75 inches all save 128.57pt, because below about 1.79 the content is
    # bigger than the canvas and the tight box grows back to it. The floor is
    # the key above the axes plus the axis label and the ticks, all fixed point
    # sizes. That is a real floor and it was measured rather than reasoned.
    #
    # What it missed is that the frame could go the OTHER way. The chart was
    # being squeezed to fit a caption the frame did not need: one of the two
    # lines was a gloss the frame title already gives, and only the
    # indicative-levels marker is a carried key. Cutting the caption to one line
    # raises the budget from 128.1 to 153.1, and at the full slot this saves
    # 145.8pt and fits with room to spare.
    #
    # It matters because of what the squeeze was costing. Rendered at 128.6pt
    # the y ticks 50 and 75 were almost touching and the whole 35 to 95 range
    # sat in about 20pt of plot, which is a chart whose only content is a rise
    # drawn too small to read. Shrinking to fit is not free, and the first
    # question is whether the thing being fitted around has to be there.
    #
    # And the answer, second time round, was that it did. Cutting the caption to
    # one line dropped two words the chart itself gave up, "time" and "without"
    # from its stripped title, and check_room_carried_text.py said so within the
    # hour. So the caption goes back to two lines and the height comes from
    # somewhere that was never teaching: the legend row above the axes. See the
    # comment on the labels below.
    #
    # 1.88in, a bespoke canvas rather than the captioned slot. With the legend
    # row gone the saved box is no longer floored by fixed furniture the way it
    # was: it tracks the canvas again, 1.95in giving 132.7pt and 1.88 giving
    # 126.8, so the old "no shorter canvas moves it" is simply no longer true.
    # That sentence was measured and correct, and it stopped being correct the
    # moment the thing doing the flooring was removed. A measurement is about a
    # configuration, not about a chart.
    fig, ax = plt.subplots(
        figsize=room_size(*room_pick((7, 3.3), (5.79, 1.88))))

    # Thinking tokens spent at answer time, log scale
    thinking_tokens = np.logspace(0, 4, 200)  # 1 to 10,000 tokens
    log_tokens = np.log10(thinking_tokens)

    # Rising, saturating accuracy as more compute is spent thinking
    accuracy = 44 + 44 / (1 + np.exp(-1.3 * (log_tokens - 1.9)))

    # Fixed model, no extra thinking time: flat reference
    fixed_accuracy = np.full_like(thinking_tokens, 46)

    # Room edition: two-word legend entries. Either of these at full length is
    # 34 characters, which is 82 per cent of the slide drawn inside the axes.
    ax.plot(thinking_tokens, accuracy, color=CLARET, linewidth=2.2,
            label=room_pick('More thinking tokens at answer time', 'more thinking'))
    ax.plot(thinking_tokens, fixed_accuracy, color=SOFT, linewidth=1.6, linestyle='--',
            label=room_pick('Fixed model, no extra thinking time', 'fixed model'))

    ax.set_xscale('log')
    ax.set_xlabel(room_pick('Compute spent at answer time (thinking tokens, log scale)',
                            'Thinking tokens'), fontsize=FONT_SIZE)
    # Rotated, so 22 characters is 240pt of height in a 176pt slot: the page grew
    # to 242pt to fit it. The carried title still says benchmark accuracy.
    # 'Accuracy %', not 'Accuracy (%)', and the two characters are the whole
    # difference between fitting and not. Dropping the legend row gave the axes
    # its height back and immediately handed the ceiling to this label instead:
    # rotated, it is measured against the axes HEIGHT, so at 15pt twelve
    # characters is about 100pt of ink and the saved box went to 140.2pt of a
    # 128.1pt budget with the canvas unchanged. That is the same trap the y
    # label on compute_requirements fell into, one chart over.
    ax.set_ylabel(room_pick('Benchmark accuracy (%)', 'Accuracy %'), fontsize=FONT_SIZE)
    ax.set_title('Inference-Time Compute: Accuracy Climbs Without Retraining',
                fontsize=FONT_SIZE+1, fontweight='bold')
    ax.set_xlim(1, 10000)
    ax.set_ylim(35, 95)
    # Room edition: the key goes ABOVE the plot.
    #
    # It was moved to the upper left inside the axes when the room legends were
    # transparent, so the claret curve ran through the words. Giving every room
    # legend an opaque card fixed the words and made the picture worse: the card
    # is 193x63pt in an 86pt axes, so it now HID the rise from 1 to 500 thinking
    # tokens, which is the only part of this chart anyone looks at. Neither
    # version collides with anything, because a curve is graphics and no
    # geometric check reads ink that is not a glyph. Both were found by looking.
    #
    # Third placement, and the first that is not a box at all. Above the axes is
    # a whole row of 15pt type plus its padding on a canvas of about 140pt, and
    # the axes gets what is left: 35 to 95 per cent ended up in roughly 20pt of
    # plot, with the y ticks 50 and 75 almost touching. Two series that each run
    # clear of the other for most of the width do not need a box, and a direct
    # label costs nothing but the ink it is printed in.
    #
    # It also costs no carried text, which is why this and not "delete the
    # legend". A legend REMOVED turns its labels into strings the chart gave up
    # and the frame is then required to print them. Labelled in place they are
    # still on the chart.
    if ROOM:
        ax.text(250, 82, 'more thinking', color=CLARET,
                fontsize=FONT_SIZE-1, fontweight='bold', ha='left', va='bottom')
        ax.text(700, 49, 'fixed model', color=SOFT,
                fontsize=FONT_SIZE-1, fontweight='bold', ha='left', va='bottom')
    else:
        ax.legend(loc='lower right', fontsize=FONT_SIZE-1)
    # Minor gridlines go in the room edition: a log x axis draws eight per
    # decade, which at this size is a hatch across the data rather than a
    # reference.
    ax.grid(True, alpha=0.3, which=room_pick('both', 'major'))

    since = ax.annotate('Since 2024: capability is bought\nat answer time, not only at training time',
               xy=(800, 79), xytext=(2.5, 68),
               arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.3),
               fontsize=FONT_SIZE-1, color=GOLD, fontweight='bold')
    if ROOM:
        # The text is long enough for the room pass to lift it onto the slide,
        # which is what should happen to it. The arrow is not text and would stay
        # behind, pointing from an empty patch of axes at nothing.
        since.arrow_patch.set_visible(False)

    ax.text(0.02, 0.04, 'Indicative levels, not measured benchmark scores',
            transform=ax.transAxes, fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath('inference_compute_scaling'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: inference_compute_scaling.pdf")


if __name__ == "__main__":
    create_inference_compute_scaling()
