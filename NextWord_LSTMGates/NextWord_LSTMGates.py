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


def create_lstm_gates():
    """The cell state as a highway; gates are learned taps on it."""
    # 2.06in, not the 2.45in slot: the slot reserves ONE caption line and this
    # frame needs one for the gate key that does not fit a 33-character heading.
    # tools/probe_room_fig_height.py measures the difference and names the target,
    # 153.1pt of saved box, which is 2.06in plus the 3pt this diagram draws
    # outside its own axes.
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7.2, 3.3), (5.79, 2.06))))
    if ROOM:
        # tight_layout sizes an axes so that everything hanging outside it still
        # fits the figure. On a diagram whose labels are drawn at 18pt and stick
        # past the data area, that shrinks the axes rather than the text, and
        # the whole diagram is then drawn small on a large slide. An axes placed
        # by hand has no subplot geometry for tight_layout to solve.
        ax.remove()
        ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    # The room crops the empty band under the gates rather than drawing it. The
    # action labels and the arrow tails move up with it, so the diagram itself
    # is unchanged and only the white around it goes: 5 units of height became
    # 3.8, which is 22 per cent more point per data unit for every box in it.
    ax.set_ylim(*room_pick((0, 5), (1.15, 4.95)))
    ax.axis('off')

    highway_y = 3.4
    ax.annotate('', xy=(11.4, highway_y), xytext=(0.6, highway_y),
                arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=6, alpha=0.9),
                zorder=1)
    # Higher in the room, because the gate boxes are wider there and "C(t-1)" is
    # 61pt of ink starting at the left end of the highway, which now runs over
    # the corner of the first box.
    cap_dy = room_pick(0.55, 0.78)
    ax.text(0.6, highway_y + cap_dy, 'C(t-1)', fontsize=FONT_SIZE, color=TEAL,
            fontweight='bold', ha='left')
    ax.text(11.4, highway_y + cap_dy, 'C(t)', fontsize=FONT_SIZE, color=TEAL,
            fontweight='bold', ha='right')

    # Three gate boxes sit 104pt apart on a room slide and each name is drawn at
    # 18pt, so "Forget gate" is wider than the gap to its neighbour. The word
    # "gate" is the one the frame title already carries.
    gates = room_pick([
        (3.0, CLARET, 'Forget gate', 'keep or drop'),
        (6.0, GOLD, 'Input gate', 'add new info'),
        (9.0, SOFT, 'Output gate', 'reveal to h(t)'),
    ], [
        (3.0, CLARET, 'Forget', 'keep/drop'),
        (6.0, GOLD, 'Input', 'add new'),
        (9.0, SOFT, 'Output', 'reveal'),
    ])
    # Shortening the names was not enough on its own: a 1.8 unit box is 63pt
    # wide at the room slot and "Output" is 71pt of ink at 18pt, so the label
    # printed straight through the box drawn to hold it. 2.4 units is 83pt.
    gate_half, gate_w = room_pick(0.9, 1.2), room_pick(1.8, 2.4)
    for x, color, name, action in gates:
        box = FancyBboxPatch((x - gate_half, highway_y - 0.5), gate_w, 1.0,
                              boxstyle='round,pad=0.06', facecolor=color,
                              edgecolor=INK, linewidth=room_pick(1.2, 2.0), zorder=3)
        ax.add_patch(box)
        ax.text(x, highway_y, name, ha='center', va='center', fontsize=FONT_SIZE - 1,
                color=PAPER, fontweight='bold', zorder=4)
        ax.annotate('', xy=(x, highway_y - 0.55), xytext=(x, room_pick(1.0, 2.05)), zorder=2,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=room_pick(1.4, 2.6)))
        ax.text(x, room_pick(0.65, 1.7), action, ha='center', va='center',
                fontsize=FONT_SIZE - 2, color=color)

    ax.text(6.0, 0.1, 'Every gate reads h(t-1) and x(t) and outputs a value between 0 and 1.',
            ha='center', fontsize=FONT_SIZE - 2, style='italic', color=SOFT)
    ax.set_title('The LSTM Cell: Gates Are Learned Taps on the Cell-State Highway',
                fontsize=FONT_SIZE + 1, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figpath('lstm_gates'))
    plt.close()
    print("Created: lstm_gates.pdf")


if __name__ == "__main__":
    create_lstm_gates()
