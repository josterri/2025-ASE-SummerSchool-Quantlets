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


def create_agentic_risk_surface():
    """Chart: exposed surface widens as autonomy grows, structural, not quantitative"""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 4.9)
    ax.axis('off')

    stage_names = [
        '1. Read-only\nanswering',
        '2. Tool use',
        '3. Writing and acting\non external systems',
        '4. Multi-step\nautonomous execution',
    ]
    captions = [
        'no side\neffects',
        'reads external\ndata',
        'can change\nexternal state',
        'chains actions\nwithout review',
    ]
    xs = [0.4, 2.6, 4.8, 7.0]
    widths = [1.8, 1.8, 1.8, 1.8]
    heights = [1.3, 2.1, 2.9, 3.7]
    base_y = 0.9
    shades = sequential_cmap(CLARET)(np.linspace(0.3, 1.0, 4))

    for x, w, h, name, caption, shade in zip(xs, widths, heights, stage_names, captions, shades):
        backdrop = Rectangle((x, base_y), w, h, facecolor=shade, edgecolor=RULE, linewidth=0.8, zorder=1)
        ax.add_patch(backdrop)

        label_w, label_h = 1.6, 0.8
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
    plt.savefig('figures/agentic_risk_surface.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: agentic_risk_surface.pdf")


if __name__ == "__main__":
    create_agentic_risk_surface()
