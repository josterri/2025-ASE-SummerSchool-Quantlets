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


def create_positional_encoding():
    """Proper heatmap of the sinusoidal positional encoding, replacing the
    hand-drawn tikz grid."""
    n_positions = 51
    d_model = 64
    pos = np.arange(n_positions)[:, None]
    k = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (k // 2)) / np.float64(d_model))
    angles = pos * angle_rates
    pe = np.zeros_like(angles)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    # Dimensions above roughly 24 have wavelengths far longer than the window,
    # so they render as near-constant stripes that read as noise. Compute all
    # 64 so the wavelengths are correct, then plot the band that is legible.
    shown = 24
    # 1.95in, and this one has a floor that a smaller number goes straight
    # through. The frame carries a two-line caption, so the budget is 128.1pt.
    # 1.52in reaches the 103.1pt a THREE-line caption would need and the figure
    # is destroyed at it: the ROTATED y label is about 100pt of ink measured
    # against the axes HEIGHT, so at that size the label is taller than the
    # heatmap, the 0 and 20 ticks touch, the colorbar numbers overprint, and the
    # oscillation this chart exists to show is a 20pt band of stripes. Rendered
    # and looked at, which is the only thing that could have said so.
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7.2, 3.3), (5.79, 1.95))))
    im = ax.imshow(pe[:, :shown].T, cmap=diverging_cmap(), aspect='auto',
                    vmin=-1, vmax=1, origin='lower')
    ax.grid(False)  # gridlines must not cross heatmap cells

    # Three long labels, and between them they were most of this figure. The
    # room page came out 348 by 215pt and the heatmap held 40 per cent of it:
    # "Position in sequence" is 195pt across, and "Embedding dimension" is 212pt
    # of ROTATED ink, which is what set the page height and pushed a 2.45in
    # figure to 3.06in. All three are measured, in
    # tools/probe_room_text_extent.py, not estimated from character counts.
    ax.set_xlabel(room_pick('Position in sequence', 'Position'), fontsize=FONT_SIZE)
    ax.set_ylabel(room_pick('Embedding dimension', 'Dimension'), fontsize=FONT_SIZE)
    ax.set_title('Each dimension oscillates at its own rate, so every position gets a unique column',
                fontsize=FONT_SIZE + 1, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(room_pick('Encoding value', ''), fontsize=FONT_SIZE)

    ax.text(0.5, -0.32, 'First {} of {} dimensions. Higher ones vary more '
            'slowly still, carrying coarse position.'.format(shown, d_model),
            transform=ax.transAxes, ha='center', va='top',
            fontsize=FONT_SIZE - 1, color=SOFT, style='italic')

    plt.tight_layout()
    plt.savefig(figpath('positional_encoding'))
    plt.close()
    print("Created: positional_encoding.pdf")


if __name__ == "__main__":
    create_positional_encoding()
