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
Generate additional charts: Information Theory, Compute Requirements, Loss Curves, Applications Tree
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
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


def create_compute_requirements():
    """Chart 18: Compute requirements graph"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Model data: name, year, FLOPs (in log scale)
    models = [
        ('N-gram', 1980, 1e6),
        ('Small NN', 1990, 1e9),
        ('Word2Vec', 2013, 1e12),
        ('LSTM', 2015, 1e14),
        ('Transformer-Base', 2017, 1e16),
        ('BERT', 2018, 1e18),
        ('GPT-2', 2019, 1e19),
        ('GPT-3', 2020, 1e23),
        ('PaLM', 2022, 2.5e24),
        ('GPT-4', 2023, 1e25)
    ]
    
    names = [m[0] for m in models]
    years = [m[1] for m in models]
    flops = [m[2] for m in models]
    
    # Create scatter plot with size based on compute
    sizes = [np.log10(f) * 20 for f in flops]
    # Ramp starts at 0.35: sequential_cmap runs white -> PAPER -> base, so the
    # first markers would otherwise be white on a white background.
    colors = sequential_cmap()(np.linspace(0.35, 1, len(models)))
    
    scatter = ax.scatter(years, flops, s=sizes, c=colors, alpha=0.6, 
                        edgecolors=INK, linewidth=1)
    
    # Add labels for significant models
    for name, year, flop in models:
        if name in ['N-gram', 'Word2Vec', 'BERT', 'GPT-3', 'GPT-4']:
            ax.annotate(name, xy=(year, flop), xytext=(0, 5),
                       textcoords='offset points', ha='center', fontsize=FONT_SIZE-1)
    
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=FONT_SIZE)
    ax.set_ylabel('Training Compute (FLOPs)', fontsize=FONT_SIZE)
    ax.set_title('Exponential Growth in Compute Requirements', 
                fontsize=FONT_SIZE+1, fontweight='bold')
    
    # Custom y-axis labels
    ax.set_yticks([1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24])
    ax.set_yticklabels(['1M', '1B', '1T', '1P', '1E', '1Z', '1Y'])
    
    # Add doubling time annotation
    ax.text(2015, 1e22, 'Doubling every\n3.4 months!', fontsize=FONT_SIZE,
           bbox=dict(boxstyle='round', facecolor=GOLD, alpha=0.5),
           ha='center', fontweight='bold')
    
    # Add hardware generations
    ax.axvspan(1980, 1995, alpha=0.1, color=SOFT, label='CPU Era')
    ax.axvspan(1995, 2010, alpha=0.1, color=TEAL, label='GPU Era')
    ax.axvspan(2010, 2024, alpha=0.1, color=GOLD, label='TPU/Cluster Era')
    
    ax.legend(loc='lower right', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1975, 2025)
    ax.set_ylim(1e5, 1e26)
    
    plt.tight_layout()
    plt.savefig('figures/compute_requirements.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: compute_requirements.pdf")


if __name__ == "__main__":
    create_compute_requirements()
