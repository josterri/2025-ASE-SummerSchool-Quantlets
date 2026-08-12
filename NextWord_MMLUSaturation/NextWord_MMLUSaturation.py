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


def create_mmlu_saturation():
    """Chart 15: Benchmark saturation and replacement (indicative)"""
    fig, ax = plt.subplots(figsize=(7, 3.3))

    years = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    mmlu = np.array([28, 35, 44, 68, 86, 88, 90, 91])

    ax.plot(years, mmlu, marker='o', color=CLARET, linewidth=2.2, markersize=4,
            label='MMLU accuracy')
    ax.axhline(y=90, color=SOFT, linestyle='--', linewidth=1.4,
               label='Human-expert ceiling')

    successors = [
        (2023.8, 38, 'GPQA Diamond'),
        (2025.0, 6, 'ARC-AGI-2'),
        (2025.3, 11, "Humanity's Last Exam"),
    ]
    for x, y, name in successors:
        ax.scatter(x, y, s=36, color=TEAL, zorder=5)
        ax.annotate(name, xy=(x, y), xytext=(3, 4), textcoords='offset points',
                    fontsize=FONT_SIZE-2, color=TEAL)

    ax.set_xlabel('Year', fontsize=FONT_SIZE)
    ax.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)
    ax.set_title('Benchmarks Saturate, Then Get Replaced',
                fontsize=FONT_SIZE+1, fontweight='bold')
    ax.set_xlim(2018.5, 2026.7)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower left', fontsize=FONT_SIZE-2)
    ax.grid(True, alpha=0.3)

    ax.text(2019.2, 72, 'Indicative levels, not exact reported scores',
            ha='left', fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    plt.tight_layout()
    plt.savefig('figures/mmlu_saturation.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: mmlu_saturation.pdf")


if __name__ == "__main__":
    create_mmlu_saturation()
