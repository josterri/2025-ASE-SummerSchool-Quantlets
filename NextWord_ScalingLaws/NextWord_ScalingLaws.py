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


def create_scaling_laws():
    """Chart 10: Scaling laws visualization"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Generate synthetic but realistic scaling law data
    # Runs past 1T so the curve and band actually cover GPT-4 at 1.7e12, which
    # previously sat beyond the end of the line meant to explain it.
    model_sizes = np.logspace(7, 12.8, 60)  # 10M to about 6T parameters
    
    # Loss follows power law: L = (N_c/N)^alpha
    alpha = 0.076
    N_c = 8.8e13
    loss = (N_c / model_sizes) ** alpha + 1.69  # Add irreducible loss
    
    # Plot main scaling law
    ax.loglog(model_sizes, loss, linestyle='-', color=TEAL, linewidth=2, label='Empirical scaling law')
    
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

    for size, name, wobble in actual_models:
        loss_val = (N_c / size) ** alpha + 1.69 + wobble
        ax.scatter(size, loss_val, s=100, c=CLARET, zorder=5)
        ax.annotate(name, xy=(size, loss_val), xytext=(6, 6),
                   textcoords='offset points', fontsize=FONT_SIZE-1)
    
    # Add compute-optimal frontier (Chinchilla)
    ax.fill_between(model_sizes, loss * 0.95, loss * 1.05, 
                    alpha=0.2, color=GOLD, label='Compute-optimal region')
    
    ax.set_xlabel('Model Size (Parameters)', fontsize=FONT_SIZE)
    ax.set_ylabel('Test Loss', fontsize=FONT_SIZE)
    ax.set_title('Scaling Laws: Predictable Improvement with Size', 
                fontsize=FONT_SIZE+1, fontweight='bold')
    
    # Custom x-axis labels
    ax.set_xticks([1e8, 1e9, 1e10, 1e11, 1e12])
    ax.set_xticklabels(['100M', '1B', '10B', '100B', '1T'])
    
    ax.legend(loc='upper right', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(5e7, 5e12)
    # Bracket the law across the whole x range. The old cap of 4 clipped the
    # curve for anything below about 1B, where the loss exceeds it.
    ax.set_ylim(2.8, 4.9)

    # Add annotation about diminishing returns
    ax.annotate('Diminishing returns,\nbut predictable', xy=(1e11, 3.36), xytext=(2.5e8, 3.05),
               arrowprops=dict(arrowstyle='->', color=GOLD, lw=1),
               fontsize=FONT_SIZE-1, color=GOLD)
    ax.text(0.98, 0.03, 'Curve is the Kaplan form; points are indicative, not measured',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONT_SIZE-2, style='italic', color=SOFT)
    
    plt.tight_layout()
    plt.savefig('figures/scaling_laws.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: scaling_laws.pdf")


if __name__ == "__main__":
    create_scaling_laws()
