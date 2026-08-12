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


def create_finetuning_vs_prompting():
    """Chart 13: Fine-tuning vs prompting comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Accuracy comparison
    methods = ['Zero-shot\nPrompting', 'Few-shot\nPrompting', 'Fine-tuning\n(1K examples)', 
               'Fine-tuning\n(10K examples)']
    accuracy = [72, 85, 89, 92]
    time_required = [0.001, 0.01, 24, 72]  # Hours
    
    colors = sequential_cmap(CLARET)(np.linspace(0.35, 0.95, len(methods)))
    bars1 = ax1.bar(methods, accuracy, color=colors, alpha=0.7, edgecolor=INK, linewidth=1)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracy):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc}%', ha='center', va='bottom', fontsize=FONT_SIZE-1, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)
    ax1.set_title('Accuracy Comparison', fontsize=FONT_SIZE+1, fontweight='bold')
    ax1.set_ylim(65, 95)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Time/Cost comparison (log scale)
    bars2 = ax2.bar(methods, time_required, color=colors, alpha=0.7, edgecolor=INK, linewidth=1)
    ax2.set_yscale('log')
    ax2.set_ylabel('Time Required (hours, log scale)', fontsize=FONT_SIZE)
    ax2.set_title('Effort/Cost Comparison', fontsize=FONT_SIZE+1, fontweight='bold')
    
    # Add value labels
    for bar, time in zip(bars2, time_required):
        height = bar.get_height()
        if time < 1:
            label = f'{time*60:.0f}s' if time >= 0.01 else 'Instant'
        else:
            label = f'{time:.0f}h'
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                label, ha='center', va='bottom', fontsize=FONT_SIZE-1, fontweight='bold')
    
    # Add annotation
    # Reserve the bottom band FIRST. tight_layout does not account for fig.text,
    # so the axes previously expanded over the caption and it overprinted the
    # x tick labels of both panels.
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    fig.text(0.5, 0.025,
            'Prompting reaches 85% at once; fine-tuning reaches 92% after days. '
            'Levels indicative, not measured.',
            ha='center', fontsize=FONT_SIZE, color=INK, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=PAPER, edgecolor=GOLD))
    plt.savefig('figures/finetuning_vs_prompting.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: finetuning_vs_prompting.pdf")


if __name__ == "__main__":
    create_finetuning_vs_prompting()
