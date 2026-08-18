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
Generate charts for Hour 4: Applications & Future
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches
from scipy.stats import norm


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


def create_cost_quality_latency():
    """Chart: cost vs quality trade-off across model classes, bubble size is latency (indicative)"""
    # 1.87in: the carried title and the indicative-positions marker are both
    # keys, the marker is a stylised label this course requires on the surface a
    # student reads, and neither fits a heading, so the frame carries two
    # caption lines and the budget is 128.1pt.
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7.5, 4.0), (5.79, 1.76))))
    if ROOM:
        # This one has real axes, so it cannot take the full figure the way the
        # diagrams do, but it has the same disease: its point labels are drawn
        # outside the data area and tight_layout shrinks the axes to fit them,
        # which left the plot 244pt wide inside a 417pt figure and pushed two
        # labels into each other. The margins below are the ones the axis
        # labels and tick labels actually need at 16 to 17pt.
        ax.remove()
        # 0.36 of the height under the axes, not 0.28, and the 0.08 is the whole
        # reason this figure can reach its budget. Shrinking the canvas alone
        # stopped working at about 140pt: the x tick labels and "Cost per query
        # ($)" are drawn in POINTS below a rect given as a FRACTION, so the
        # shorter the canvas the further they hang past it, and the saved box
        # stopped following the canvas down. Reserving the band they need means
        # nothing is drawn outside the figure and the saved box is the canvas.
        #
        # 0.37 on the left is the other half of the same fix, and it pays for
        # the y label being turned flat below. Reserving that width is what
        # stops "Quality (%)" being drawn over the tick numbers, and the number
        # is MEASURED rather than chosen: flat, "Quality (%)" inks 106.8pt at
        # 19pt, the widest y tick label "100" inks 34.3pt, and labelpad is 8, so
        # the band the label needs is 149.1pt of a 416.9pt canvas. At 0.30 that
        # band was 125.1pt and the label hung 24pt off the left edge, which
        # bbox_inches='tight' duly included: the saved box came out 449.8pt,
        # \roomfig scaled it 0.927 to fit 416.83pt, and that took the 17pt point
        # labels down to 15.76pt, under the floor this whole edition exists to
        # hold. The comment above already claimed "nothing is drawn outside the
        # figure and the saved box is the canvas", which was true of the bottom
        # band it was written about and false of the left one.
        # tools/probe_figure_scale.py --check is what caught it, because it
        # predicts the page size from the figure instead of trusting the canvas.
        ax = fig.add_axes([0.37, 0.36, 0.60, 0.58])

    # Four two-line point labels on one small axes overprint each other and the
    # y tick labels. One word each in the room; the "indicative" qualifier is
    # not lost, because the provenance line below is carried to the slide.
    categories = room_pick(['Small local\nmodel', 'Mid-size\nhosted', 'Frontier\nmodel', 'Frontier +\nreasoning'],
                           ['Small', 'Mid-size', 'Frontier', 'Reasoning'])
    cost = np.array([0.001, 0.01, 0.05, 0.30])    # $ per query, indicative
    quality = np.array([58, 74, 87, 93])           # accuracy %, indicative
    latency = np.array([0.4, 1.2, 2.5, 11.0])      # seconds, indicative
    colors = [SOFT, TEAL, GOLD, CLARET]

    ax.plot(cost, quality, linestyle='--', color=RULE, linewidth=1.6, zorder=1)

    for c, q, l, color in zip(cost, quality, latency, colors):
        ax.scatter(c, q, s=l * 45, color=color, alpha=0.75, edgecolors=INK,
                   linewidth=1.2, zorder=3)

    # The two dearest models are only 78pt apart on a room slide and both their
    # labels are about 85pt wide, so one goes above its bubble and the other
    # below. Same reason the ceiling is raised below: an above-label on the 87
    # per cent point needs somewhere to be.
    # Mid-size is anchored right of its own bubble and Frontier left of its
    # own, because the two points are 58pt apart on a room slide and the two
    # labels are about 75 and 68pt wide: centred, they printed through each
    # other by 7.7 by 4.5pt, measured by tools/check_room_charts.py after the
    # figure was cut to its two-caption-line budget.
    offsets = room_pick([(-18, 14), (0, 20), (0, 20), (24, 16)],
                        [(0, 18), (-6, 18), (10, 16), (0, -28)])
    label_ha = room_pick(['center', 'center', 'center', 'left'],
                         ['center', 'right', 'left', 'center'])
    for c, q, label, (dx, dy), ha in zip(cost, quality, categories, offsets, label_ha):
        ax.annotate(label, xy=(c, q), xytext=(dx, dy), textcoords='offset points',
                    ha=ha, fontsize=FONT_SIZE-2)

    ax.set_xscale('log')
    ax.set_xlabel(room_pick('Cost per query (log scale, $, indicative)', 'Cost per query ($)'), fontsize=FONT_SIZE)
    # Turned FLAT in the room, the same move create_sampling_strategies makes,
    # and here it is what lets the figure reach its budget at all. Rotated, this
    # label is measured against the axes HEIGHT: "Quality (%)" at 19pt is about
    # 110pt of ink against an axes 74pt tall, so it hung 18pt past the canvas
    # top and bottom and the saved box stopped following the canvas down at
    # 143pt however short the canvas was made. Flat, it is measured against the
    # left margin instead, which is width this chart has and height it does not.
    # The string is untouched; only its rotation and anchor move.
    ylabel_kw = ({"rotation": 0, "ha": "right", "va": "center", "labelpad": 8}
                 if ROOM else {})
    ax.set_ylabel(room_pick('Answer quality (%, indicative)', 'Quality (%)'),
                  fontsize=FONT_SIZE, **ylabel_kw)
    ax.set_title('Cost, Quality and Latency: The Curve Bends',
                fontsize=FONT_SIZE+1, fontweight='bold')
    ax.set_xlim(0.0004, 1.0)
    # The room floor sits ON a labelled tick. This axis does not start at zero
    # and should not: the marks are points, not bars, so their position carries
    # no length to exaggerate, and zeroing it would squash four indicative
    # points into the top third for nothing. What is not allowed is a truncation
    # a reader cannot see, and a floor of 40 with its lowest label at 50 is
    # exactly that: the axis simply began somewhere, unlabelled. At 50 the
    # bottom of the axis says where it starts.
    ax.set_ylim(room_pick(40, 50), room_pick(100, 112))
    # Minor gridlines on a log axis are nine lines per decade, which at room
    # line weights reads as hatching behind the data.
    ax.grid(True, alpha=0.3, which=room_pick('both', 'major'))

    # The key moves to the empty lower right in the room. It sat at the lower
    # left, which was clear while the axis started at 40 and is not once the
    # axis starts at 50: the trend line rises out of that corner and ran through
    # the words. Data bottom left to top right leaves exactly one corner free.
    # Stays in the lower RIGHT and drops to the axis floor. It overlapped the
    # Reasoning label by 3.3pt once the figure was cut to its two-caption-line
    # budget, and the obvious repair, moving it to the lower left, was rendered
    # and is worse: there it prints straight through the Small bubble and the
    # trend line, which is text over GRAPHICS and the one thing no checker
    # here can see. So the key goes down and the Reasoning label comes up.
    ax.text(room_pick(0.0006, 0.006), room_pick(44, 50.6),
            room_pick('Bubble size = latency (indicative)', 'Bubble = latency'),
            fontsize=FONT_SIZE-1, style='italic', color=SOFT)
    ax.text(0.98, 0.04, 'Indicative positions: illustrate the trade-off shape, not measured benchmarks',
            transform=ax.transAxes, ha='right', fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    gains_note = ax.annotate('Quality gains cost more\nand take longer to arrive',
               xy=(0.09, 89.5), xytext=(0.02, 50),
               arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.3),
               fontsize=FONT_SIZE-1, color=GOLD, fontweight='bold')
    if ROOM:
        # Removed whole, not left to the room pass. The pass blanks the text of
        # a long annotation and records it, but the ARROW stays, and an arrow
        # pointing from nothing to a bubble is worse than no annotation.
        # Reported by hand instead, since a removed artist is not recorded.
        gains_note.remove()

    plt.tight_layout()
    plt.savefig(figpath('cost_quality_latency'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: cost_quality_latency.pdf")


if __name__ == "__main__":
    create_cost_quality_latency()
