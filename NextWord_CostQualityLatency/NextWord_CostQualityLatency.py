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


def create_cost_quality_latency():
    """Chart: cost vs quality trade-off across model classes, bubble size is latency (indicative)"""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    categories = ['Small local\nmodel', 'Mid-size\nhosted', 'Frontier\nmodel', 'Frontier +\nreasoning']
    cost = np.array([0.001, 0.01, 0.05, 0.30])    # $ per query, indicative
    quality = np.array([58, 74, 87, 93])           # accuracy %, indicative
    latency = np.array([0.4, 1.2, 2.5, 11.0])      # seconds, indicative
    colors = [SOFT, TEAL, GOLD, CLARET]

    ax.plot(cost, quality, linestyle='--', color=RULE, linewidth=1.6, zorder=1)

    for c, q, l, color in zip(cost, quality, latency, colors):
        ax.scatter(c, q, s=l * 45, color=color, alpha=0.75, edgecolors=INK,
                   linewidth=1.2, zorder=3)

    offsets = [(-18, 14), (0, 20), (0, 20), (24, 16)]
    label_ha = ['center', 'center', 'center', 'left']
    for c, q, label, (dx, dy), ha in zip(cost, quality, categories, offsets, label_ha):
        ax.annotate(label, xy=(c, q), xytext=(dx, dy), textcoords='offset points',
                    ha=ha, fontsize=FONT_SIZE-2)

    ax.set_xscale('log')
    ax.set_xlabel('Cost per query (log scale, $, indicative)', fontsize=FONT_SIZE)
    ax.set_ylabel('Answer quality (%, indicative)', fontsize=FONT_SIZE)
    ax.set_title('Cost, Quality and Latency: The Curve Bends',
                fontsize=FONT_SIZE+1, fontweight='bold')
    ax.set_xlim(0.0004, 1.0)
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3, which='both')

    ax.text(0.0006, 44, 'Bubble size = latency (indicative)',
            fontsize=FONT_SIZE-1, style='italic', color=SOFT)
    ax.text(0.98, 0.04, 'Indicative positions: illustrate the trade-off shape, not measured benchmarks',
            transform=ax.transAxes, ha='right', fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    ax.annotate('Quality gains cost more\nand take longer to arrive',
               xy=(0.09, 89.5), xytext=(0.02, 50),
               arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.3),
               fontsize=FONT_SIZE-1, color=GOLD, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/cost_quality_latency.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: cost_quality_latency.pdf")


if __name__ == "__main__":
    create_cost_quality_latency()
