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


def create_rlhf_pipeline():
    """Chart: RLHF in three stages left to right, human preference data enters at stage (b) only"""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    def box(x, y, w, h, text, facecolor, textcolor=PAPER, fontsize=FONT_SIZE-2):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=facecolor, edgecolor=INK, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, fontweight='bold')

    stage_w, stage_h, stage_y = 2.5, 1.4, 2.5
    stage_x = [0.4, 3.75, 7.1]
    stage_colors = [TEAL, GOLD, CLARET]
    stage_texts = [
        '(a) Supervised\nfine-tuning on\ndemonstrations',
        '(b) Reward model on\nhuman preference\ncomparisons',
        '(c) Policy optimisation\nagainst the\nreward model',
    ]
    for x, text, color in zip(stage_x, stage_texts, stage_colors):
        box(x, stage_y, stage_w, stage_h, text, color)

    y_mid = stage_y + stage_h / 2
    arrow_labels = ['base\npolicy', 'reward\nsignal']
    for i in range(2):
        x1 = stage_x[i] + stage_w
        x2 = stage_x[i + 1]
        ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle='->', color=INK, lw=1.5))
        ax.text((x1 + x2) / 2, y_mid + 0.35, arrow_labels[i], ha='center', va='bottom',
                fontsize=FONT_SIZE-2, color=SOFT)

    # Human preference data enters only at stage (b)
    human_x, human_w, human_y, human_h = stage_x[1], stage_w, 4.7, 1.0
    box(human_x, human_y, human_w, human_h, 'Human preference\ncomparisons: "A or B?"', SOFT)
    ax.annotate('', xy=(human_x + human_w/2, stage_y + stage_h), xytext=(human_x + human_w/2, human_y),
                arrowprops=dict(arrowstyle='->', color=SOFT, lw=1.6))
    ax.text(human_x + human_w/2, human_y + human_h + 0.15, 'The only human data in this pipeline',
            ha='center', va='bottom', fontsize=FONT_SIZE-2, color=CLARET, fontweight='bold')

    captions = [
        'trained on:\n(prompt, ideal response) pairs',
        'trained on:\nhuman preference comparisons',
        "trained on: reward model's\nscores, no new human labels",
    ]
    for x, caption in zip(stage_x, captions):
        ax.text(x + stage_w/2, stage_y - 0.2, caption, ha='center', va='top',
                fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    ax.set_title('RLHF: Human Judgment Enters at One Stage',
                fontsize=FONT_SIZE+1, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/rlhf_pipeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: rlhf_pipeline.pdf")


if __name__ == "__main__":
    create_rlhf_pipeline()
