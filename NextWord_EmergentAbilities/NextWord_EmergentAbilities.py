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


def create_emergent_abilities():
    """Chart 15: Emergent abilities chart"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Model sizes (parameters)
    model_sizes = np.logspace(8, 12, 100)  # 100M to 1T
    
    # Different abilities emerge at different scales
    abilities = [
        ('Basic Grammar', 1e8, 0.5e9),
        ('Simple Q&A', 5e8, 2e9),
        ('Translation', 1e9, 5e9),
        ('Summarization', 5e9, 20e9),
        ('Arithmetic', 10e9, 50e9),
        ('Reasoning', 50e9, 200e9),
        ('Code Generation', 100e9, 500e9),
        ('Complex Reasoning', 500e9, 2e12)
    ]
    
    # Plot emergence curves. Start the ramp at 0.35, not 0: sequential_cmap is
    # built from white through PAPER to the base colour, so index 0 is pure
    # white and the first two curves were invisible against the background.
    colors = sequential_cmap()(np.linspace(0.35, 1, len(abilities)))
    
    for i, (name, start, end) in enumerate(abilities):
        # Create sigmoid curve for emergence
        x = np.log10(model_sizes)
        x_start = np.log10(start)
        x_end = np.log10(end)
        
        # Sigmoid function
        performance = 100 / (1 + np.exp(-5 * (x - (x_start + x_end)/2) / (x_end - x_start)))
        
        ax.semilogx(model_sizes, performance, linewidth=2, label=name, color=colors[i])
        
        # Mark emergence point (50% performance)
        emergence_point = 10**((x_start + x_end)/2)
        ax.scatter(emergence_point, 50, s=50, color=colors[i], zorder=5)
    
    ax.set_xlabel('Model Size (Parameters)', fontsize=FONT_SIZE)
    ax.set_ylabel('Task Performance (%)', fontsize=FONT_SIZE)
    ax.set_title('Emergent Abilities: Sudden Capability Jumps at Scale', 
                fontsize=FONT_SIZE+1, fontweight='bold')
    
    # Custom x-axis labels
    ax.set_xticks([1e8, 1e9, 1e10, 1e11, 1e12])
    ax.set_xticklabels(['100M', '1B', '10B', '100B', '1T'])
    
    ax.legend(loc='center left', fontsize=FONT_SIZE-2, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5e7, 5e12)
    ax.set_ylim(-5, 105)
    
    # Add annotation
    ax.annotate('Abilities emerge\nsuddenly at scale!', xy=(1e11, 50), xytext=(5e10, 25),
               arrowprops=dict(arrowstyle='->', color=CLARET, lw=1.5),
               fontsize=FONT_SIZE, color=CLARET, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/emergent_abilities.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: emergent_abilities.pdf")


if __name__ == "__main__":
    create_emergent_abilities()
