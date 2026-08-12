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
                     SOFT, PAPER, RULE, sequential_cmap, diverging_cmap)
use_course_style()
sns.set_palette(CYCLE)


# Common settings
FONT_SIZE = 8
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
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    im = ax.imshow(pe[:, :shown].T, cmap=diverging_cmap(), aspect='auto',
                    vmin=-1, vmax=1, origin='lower')
    ax.grid(False)  # gridlines must not cross heatmap cells

    ax.set_xlabel('Position in sequence', fontsize=FONT_SIZE)
    ax.set_ylabel('Embedding dimension', fontsize=FONT_SIZE)
    ax.set_title('Each dimension oscillates at its own rate, so every position gets a unique column',
                fontsize=FONT_SIZE + 1, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Encoding value', fontsize=FONT_SIZE)

    ax.text(0.5, -0.32, 'First {} of {} dimensions. Higher ones vary more '
            'slowly still, carrying coarse position.'.format(shown, d_model),
            transform=ax.transAxes, ha='center', va='top',
            fontsize=FONT_SIZE - 1, color=SOFT, style='italic')

    plt.tight_layout()
    plt.savefig('figures/positional_encoding.pdf')
    plt.close()
    print("Created: positional_encoding.pdf")


if __name__ == "__main__":
    create_positional_encoding()
