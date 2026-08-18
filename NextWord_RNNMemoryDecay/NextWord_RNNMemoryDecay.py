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


def create_rnn_memory_decay():
    """Chart 5: RNN memory decay visualization"""
    # The room figure fills the measured slide slot instead of keeping the 2:1
    # aspect, which would leave it 65pt narrower than the slide for no gain.
    fig, ax = plt.subplots(figsize=room_size(*room_pick((8, 4), (5.79, 2.45))))

    # Word positions
    positions = np.arange(1, 21)

    # Memory strength (exponential decay)
    vanilla_rnn = 100 * np.exp(-0.3 * positions)
    lstm = 100 * np.exp(-0.1 * positions)
    gru = 100 * np.exp(-0.15 * positions)

    # Plot lines
    ax.plot(positions, vanilla_rnn, 'o-', label=room_pick('Vanilla RNN', 'RNN'),
            linewidth=room_pick(2, 3), markersize=room_pick(4, 7))
    ax.plot(positions, lstm, 's-', label='LSTM',
            linewidth=room_pick(2, 3), markersize=room_pick(4, 7))
    if not ROOM:
        # GRU is standard-only. It sits between the other two and makes the same
        # point LSTM already makes, a gated cell holds memory far longer, so in
        # the room it costs a third legend entry and a third curve in the band
        # where the other two are closest. Two entries is the room budget and
        # the contrast the frame teaches is RNN against LSTM.
        ax.plot(positions, gru, '^-', label='GRU',
                linewidth=room_pick(2, 3), markersize=room_pick(4, 7))

        # The threshold rule and both free-floating notes are standard-only too.
        # The rule is drawn in CLARET, which at room scale is the same claret as
        # the RNN curve, and once its label has gone to the slide an unlabelled
        # dashed line in a series colour reads as a fourth series rather than as
        # a reference. Both strings are short, so neither is caught by the room
        # pass's recorder: they are reported by hand and belong on the slide.
        ax.axhline(y=10, color=CLARET, linestyle='--', alpha=0.5, linewidth=1)
        ax.text(19, 12, 'Effective threshold', fontsize=FONT_SIZE-1, color=CLARET, ha='right')
        ax.annotate('Information lost!', xy=(10, vanilla_rnn[9]), xytext=(12, 20),
                    arrowprops=dict(arrowstyle='->', color=CLARET, lw=1),
                    fontsize=FONT_SIZE-1, color=CLARET)

    ax.set_xlabel('Word Position', fontsize=FONT_SIZE)
    # A rotated y-label is measured against the axes HEIGHT, and the room slot
    # is 176pt tall: "Memory Strength (%)" is 204pt of ink there, taller than
    # the whole figure, and the axes it labels came out 47pt high. Measured with
    # tools/probe_room_text_extent.py, not guessed.
    ax.set_ylabel(room_pick('Memory Strength (%)', 'Memory %'), fontsize=FONT_SIZE)
    ax.set_title('RNN Memory Decay: Information Fades with Distance',
                fontsize=FONT_SIZE+1, fontweight='bold')
    # Inside the axes in the room, not above it: above it, the legend row cost
    # as much height as the plot itself. room.mplstyle now gives the legend an
    # opaque frame so no data line runs through the words.
    #
    # It took two looks to place, and neither gate saw either miss, because a
    # legend sitting on a curve is text over GRAPHICS. Side by side it was 240pt
    # of a 330pt axes and reached back to x=7, where LSTM is still at 49 per
    # cent. Stacked it was narrower and TALLER, and swallowed the LSTM tail from
    # x=12 to the right edge instead. The frame that stops a line crossing a word
    # is exactly what turns that into a line you cannot see at all.
    #
    # So the room makes room, rather than hunting for a corner this data does not
    # have: the axis runs to 125 and the legend sits in the empty band above the
    # curves. The ticks are pinned to three values so the headroom does not
    # arrive as a fourth number nobody reads.
    #
    # 140 rather than 125, and the difference was measured off the rendered page
    # rather than reasoned about: the legend box is 36pt tall in a 92pt axes, so
    # it takes the top 39 per cent whatever the axis says, and at 125 its floor
    # landed at 64 while LSTM at the legend's left edge is still 67. Three units.
    # The third look is the one that found it.
    ax.legend(loc='upper right', fontsize=FONT_SIZE-1, ncol=room_pick(1, 2))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, room_pick(105, 140))
    if ROOM:
        ax.set_yticks([0, 50, 100])

    plt.tight_layout()
    plt.savefig(figpath('rnn_memory_decay'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: rnn_memory_decay.pdf")


if __name__ == "__main__":
    create_rnn_memory_decay()
