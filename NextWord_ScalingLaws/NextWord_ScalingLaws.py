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


def create_scaling_laws():
    """Chart 10: Scaling laws visualization"""
    # Shorter than ROOM_SLOT_CAPTIONED_IN, and the 0.09 inch is measured rather
    # than chosen. Three keys survive here, the stylised marker this course
    # requires on the surface a student reads, which two points are GPT-2 and
    # GPT-3, and the part of the title a 33 character heading cannot hold, so
    # the caption runs to three lines and the figure gets 103.1pt. At the
    # captioned slot it saved 107.6pt, 4.5 over.
    fig, ax = plt.subplots(
        figsize=room_size(*room_pick((8, 5), (ROOM_SLOT_CAPTIONED_IN[0], 2.29))))

    # Generate synthetic but realistic scaling law data
    # Runs past 1T so the curve and band actually cover GPT-4 at 1.7e12, which
    # previously sat beyond the end of the line meant to explain it.
    model_sizes = np.logspace(7, 12.8, 60)  # 10M to about 6T parameters
    
    # Loss follows power law: L = (N_c/N)^alpha
    alpha = 0.076
    N_c = 8.8e13
    loss = (N_c / model_sizes) ** alpha + 1.69  # Add irreducible loss
    
    # Plot main scaling law
    ax.loglog(model_sizes, loss, linestyle='-', color=TEAL, linewidth=2,
              label=room_pick('Empirical scaling law', 'scaling law'))
    if ROOM:
        # Loss runs 2.8 to 4.9, so a log y axis has no decade boundary in range
        # and matplotlib labels it with minor ticks only. The room pass silences
        # minor tick labels to keep mathtext exponents off the axis, and the two
        # together leave a y axis carrying no numbers at all. Linear here says the
        # same thing and can be read.
        ax.set_yscale('linear')
    
    # Place each model ON the law, with a small deterministic wobble so the
    # points read as measurements scattered around the trend. The previous
    # version hand-typed losses (GPT-1 at 3.4 down to GPT-4 at 2.0) that were
    # derived independently of the curve being plotted, so the "law" line
    # passed through none of its own data, missing GPT-1 by about 1.1.
    actual_models = [
        (117e6, 'GPT-1', 0.05),
        (1.5e9, 'GPT-2', -0.04),
        (175e9, 'GPT-3', 0.04),
        (1.7e12, 'GPT-4', -0.03),
    ]

    # Room edition: all four points are drawn, the two ENDS are named.
    #
    # GPT-3 and GPT-4 are one decade apart on a six-decade axis, which is 48pt,
    # and each label is 52pt wide, so two labels at the same height overlap by
    # construction. check_room_charts.py does not see it: after its cross-axis
    # shrink the pair overlaps by 0.2pt, under the 0.8pt it needs. A reader sees
    # "GPT-3GPT-4" and that is what the render showed. Putting GPT-3's label on
    # the other side of its point was tried on paper and lands it on the curve,
    # because above-LEFT of a falling curve is under the curve.
    #
    # Naming the ends is what this chart needs, and the caption carried to the
    # room slide already says the points are indicative rather than measured.
    room_labelled = ('GPT-1', 'GPT-4')
    for size, name, wobble in actual_models:
        loss_val = (N_c / size) ** alpha + 1.69 + wobble
        ax.scatter(size, loss_val, s=100, c=CLARET, zorder=5)
        if not ROOM or name in room_labelled:
            ax.annotate(name, xy=(size, loss_val), xytext=(6, 6),
                       textcoords='offset points', fontsize=FONT_SIZE-1)
    
    # Add compute-optimal frontier (Chinchilla)
    ax.fill_between(model_sizes, loss * 0.95, loss * 1.05,
                    alpha=0.2, color=GOLD,
                    label=room_pick('Compute-optimal region', 'compute-optimal'))
    
    ax.set_xlabel(room_pick('Model Size (Parameters)', 'Parameters'), fontsize=FONT_SIZE)
    ax.set_ylabel('Test Loss', fontsize=FONT_SIZE)
    ax.set_title('Scaling Laws: Predictable Improvement with Size', 
                fontsize=FONT_SIZE+1, fontweight='bold')
    
    # Custom x-axis labels
    ax.set_xticks(room_pick([1e8, 1e9, 1e10, 1e11, 1e12], [1e8, 1e10, 1e12]))
    ax.set_xticklabels(room_pick(['100M', '1B', '10B', '100B', '1T'],
                                 ['100M', '10B', '1T']))

    # Room edition: the free corner is the one under the curve. Upper right is
    # where the loss curve and the GPT-2 and GPT-3 labels are, and a legend has
    # no opaque backing here, so both entries printed straight through them.
    # Minor gridlines go too: a log x axis draws eight per decade, which at this
    # size is a hatch across the data rather than a reference.
    # Room edition: the key goes above the plot, in one row. There is no free
    # corner inside a 90pt axes: two stacked entries are 44pt tall, half its
    # height, so upper right printed through the curve and lower left printed
    # through GPT-2. Above it, where the title used to be, nothing is drawn.
    # Minor gridlines go too: a log x axis draws eight per decade, which at this
    # size is a hatch across the data rather than a reference.
    ax.legend(loc=room_pick('upper right', 'lower center'),
              bbox_to_anchor=room_pick(None, (0.5, 1.01)),
              ncols=room_pick(1, 2), fontsize=FONT_SIZE-1, **ROOM_KEY_ABOVE)
    ax.grid(True, alpha=0.3, which=room_pick('both', 'major'))
    # Room edition: a decade and a half of clear air on the right, so GPT-4's
    # label is drawn INSIDE the axes. It used to end 3pt past the right edge,
    # which does not overflow: it grows the saved page and shrinks the plot.
    ax.set_xlim(*room_pick((5e7, 5e12), (4e7, 5e13)))
    # Bracket the law across the whole x range. The old cap of 4 clipped the
    # curve for anything below about 1B, where the loss exceeds it.
    # Room edition: 5.4 rather than 4.9, for the same reason in the other
    # direction. GPT-1 is the highest point on the curve and its label is drawn
    # above it, 13pt beyond the top of the axes.
    ax.set_ylim(2.8, room_pick(4.9, 5.4))

    # Add annotation about diminishing returns
    if not ROOM:
        # 35 characters, under the room pass's 40-character axes threshold, so it
        # would print across the curve rather than being carried. It goes on the
        # room slide instead.
        ax.annotate('Diminishing returns,\nbut predictable', xy=(1e11, 3.36), xytext=(2.5e8, 3.05),
                   arrowprops=dict(arrowstyle='->', color=GOLD, lw=1),
                   fontsize=FONT_SIZE-1, color=GOLD)
    ax.text(0.98, 0.03, 'Curve is the Kaplan form; points are indicative, not measured',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONT_SIZE-2, style='italic', color=SOFT)
    
    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath('scaling_laws'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: scaling_laws.pdf")


if __name__ == "__main__":
    create_scaling_laws()
