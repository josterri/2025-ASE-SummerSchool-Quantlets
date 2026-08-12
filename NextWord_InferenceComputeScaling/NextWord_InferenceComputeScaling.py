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


def create_inference_compute_scaling():
    """Chart 13: Inference-time compute scaling (indicative)"""
    fig, ax = plt.subplots(figsize=(7, 3.3))

    # Thinking tokens spent at answer time, log scale
    thinking_tokens = np.logspace(0, 4, 200)  # 1 to 10,000 tokens
    log_tokens = np.log10(thinking_tokens)

    # Rising, saturating accuracy as more compute is spent thinking
    accuracy = 44 + 44 / (1 + np.exp(-1.3 * (log_tokens - 1.9)))

    # Fixed model, no extra thinking time: flat reference
    fixed_accuracy = np.full_like(thinking_tokens, 46)

    ax.plot(thinking_tokens, accuracy, color=CLARET, linewidth=2.2,
            label='More thinking tokens at answer time')
    ax.plot(thinking_tokens, fixed_accuracy, color=SOFT, linewidth=1.6, linestyle='--',
            label='Fixed model, no extra thinking time')

    ax.set_xscale('log')
    ax.set_xlabel('Compute spent at answer time (thinking tokens, log scale)', fontsize=FONT_SIZE)
    ax.set_ylabel('Benchmark accuracy (%)', fontsize=FONT_SIZE)
    ax.set_title('Inference-Time Compute: Accuracy Climbs Without Retraining',
                fontsize=FONT_SIZE+1, fontweight='bold')
    ax.set_xlim(1, 10000)
    ax.set_ylim(35, 95)
    ax.legend(loc='lower right', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3, which='both')

    ax.annotate('Since 2024: capability is bought\nat answer time, not only at training time',
               xy=(800, 79), xytext=(2.5, 68),
               arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.3),
               fontsize=FONT_SIZE-1, color=GOLD, fontweight='bold')

    ax.text(0.02, 0.04, 'Indicative levels, not measured benchmark scores',
            transform=ax.transAxes, fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    plt.tight_layout()
    plt.savefig('figures/inference_compute_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: inference_compute_scaling.pdf")


if __name__ == "__main__":
    create_inference_compute_scaling()
