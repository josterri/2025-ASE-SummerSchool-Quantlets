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


def create_agentic_risk_surface():
    """Chart: exposed surface widens as autonomy grows, structural, not quantitative"""
    # 2.07in, which is the budget for a frame carrying ONE caption line, and it
    # is NOT the number this figure's ledger entry asks for. The declared key
    # naming the four autonomy levels is 127 characters, which needs three
    # caption lines and a 103.1pt figure, and at 103.1pt the staircase is
    # destroyed: the label cards are 1.4 data units of a 5.4 unit axis, which is
    # 27pt for two lines of 17pt type. Rendered and looked at. Reported to the
    # lead instead of drawn that small, because the room chart already labels
    # all four stairs and captions each one, so that sentence is the chart read
    # back out loud rather than a key to it.
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7.5, 3.4), (5.79, 2.02))))
    if ROOM:
        # See create_multimodal_fusion.
        ax.remove()
        ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, room_pick(4.9, 5.4))
    ax.axis('off')

    # The four steps sit 99pt apart on a room slide. "3. Writing and acting on
    # external systems" is 21 characters on its widest line, about 200pt at
    # 17pt, so it runs through both its neighbours. Every label is cut to two
    # short lines; the staircase itself is what carries the argument.
    stage_names = room_pick([
        '1. Read-only\nanswering',
        '2. Tool use',
        '3. Writing and acting\non external systems',
        '4. Multi-step\nautonomous execution',
    ], [
        # The numeral goes in the room, and it buys the margin the label boxes
        # did not have. "3. Write" inks 76.5pt inside a 79.3pt white card, so it
        # touched the border on the right; the card cannot grow, because it must
        # stay inside its own stair or the curved arrow between two stairs runs
        # through it. Dropping "N. " takes ten points off every label and the
        # ordering is not lost: the stairs rise left to right with an arrow
        # between each pair.
        'Read\nonly',
        'Tool\nuse',
        'Write\naccess',
        'Multi\nstep',
    ])
    captions = room_pick([
        'no side\neffects',
        'reads external\ndata',
        'can change\nexternal state',
        'chains actions\nwithout review',
    ], [
        'no side\neffects',
        'reads\ndata',
        'changes\nstate',
        'chains,\nno review',
    ])
    xs = [0.4, 2.6, 4.8, 7.0]
    widths = [1.8, 1.8, 1.8, 1.8]
    heights = room_pick([1.3, 2.1, 2.9, 3.7], [1.7, 2.3, 2.9, 3.5])
    base_y = room_pick(0.9, 1.5)
    # The ramp starts past PAPER in the room. sequential_cmap runs white through
    # PAPER to the base colour, so 0.3 is cream: the first two stairs of a chart
    # whose whole argument is that the surface WIDENS were, projected, two
    # rectangles the same colour as the slide behind them.
    shades = sequential_cmap(CLARET)(np.linspace(room_pick(0.3, 0.62), 1.0, 4))

    for x, w, h, name, caption, shade in zip(xs, widths, heights, stage_names, captions, shades):
        backdrop = Rectangle((x, base_y), w, h, facecolor=shade, edgecolor=RULE, linewidth=0.8, zorder=1)
        ax.add_patch(backdrop)

        # Taller in the room than it was. Two lines at 17pt ink 37pt and the card
        # was 37.6pt, so the text filled it top to bottom as well as side to
        # side. The width stays inside the stair for the reason above.
        label_w, label_h = room_pick(1.6, 1.75), room_pick(0.8, 1.4)
        label_x = x + (w - label_w) / 2
        label_y = base_y + h - label_h - 0.15
        label = FancyBboxPatch((label_x, label_y), label_w, label_h, boxstyle="round,pad=0.04",
                                facecolor=PAPER, edgecolor=INK, linewidth=1.1, zorder=2)
        ax.add_patch(label)
        ax.text(x + w/2, label_y + label_h/2, name, ha='center', va='center',
                fontsize=FONT_SIZE-2, color=INK, fontweight='bold', zorder=3)

        ax.text(x + w/2, base_y - 0.15, caption, ha='center', va='top',
                fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    for i in range(3):
        x1 = xs[i] + widths[i]
        y1 = base_y + heights[i] / 2
        x2 = xs[i + 1]
        y2 = base_y + heights[i + 1] / 2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=INK, lw=1.5,
                                     connectionstyle='arc3,rad=-0.2'))

    ax.set_title('Agentic Risk: The Exposed Surface Widens with Autonomy',
                fontsize=FONT_SIZE+1, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figpath('agentic_risk_surface'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: agentic_risk_surface.pdf")


if __name__ == "__main__":
    create_agentic_risk_surface()
