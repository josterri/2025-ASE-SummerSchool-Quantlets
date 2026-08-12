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


def create_applications_tree():
    """Chart 20: Applications tree diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xlim(0, 10)
    # Bottom starts at 1.5, just under the caption. bbox_inches='tight' crops to
    # the axes extent rather than to the drawn content, so a ylim of 0 left a
    # band of empty axis below the caption that survived the crop.
    ax.set_ylim(1.5, 10)
    ax.axis('off')

    # Root: Next-word prediction
    root_box = FancyBboxPatch((4, 8), 2, 0.8, boxstyle="round,pad=0.05",
                              facecolor=TEAL, edgecolor=INK, linewidth=2)
    ax.add_patch(root_box)
    ax.text(5, 8.4, 'Next-Word\nPrediction', ha='center', va='center', 
           color=PAPER, fontsize=FONT_SIZE+1, fontweight='bold')
    
    # Main branches. These are the three categories Hour 4 names in prose, in
    # the same order and wording. The chart previously branched Generation /
    # Understanding / Translation / Analysis, which the slide never mentions,
    # so the frame described a taxonomy its own diagram did not have.
    main_applications = [
        (2.0, 6, 'Generation'),
        (5.0, 6, 'Understanding'),
        (8.0, 6, 'Transformation')
    ]
    
    for x, y, label in main_applications:
        box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6, boxstyle="round,pad=0.03",
                             facecolor=PAPER, edgecolor=TEAL, linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=FONT_SIZE-1)
        # Connect to root
        ax.plot([5, x], [7.8, y+0.3], linestyle='-', color=INK, linewidth=1, alpha=0.5)
    
    # Specific applications
    # Leaves stack vertically under their parent. The previous layout fanned
    # them horizontally in fixed 0.7-wide boxes, which put "Reports" on top of
    # "Q&A" and let long labels such as "Classification" overflow their box.
    # Stacking also uses the dead band that used to sit under the tree.
    # Leaf labels are the examples the slide gives for each category, so a
    # student reading the frame sees the same words in the prose and the tree.
    leaf_groups = [
        (2.0, ['Drafting', 'Code completion', 'Summarisation', 'Translation']),
        (5.0, ['Classification', 'Extraction', 'Routing', 'Search']),
        (8.0, ['Code to code', 'Register shift', 'Text to JSON']),
    ]

    for parent_x, labels in leaf_groups:
        for i, label in enumerate(labels):
            y = 4.7 - i * 0.62
            half = max(0.36, 0.058 * len(label))
            box = FancyBboxPatch((parent_x - half, y - 0.2), 2 * half, 0.4,
                                 boxstyle="round,pad=0.02",
                                 facecolor=PAPER, edgecolor=GOLD, linewidth=0.5)
            ax.add_patch(box)
            ax.text(parent_x, y, label, ha='center', va='center', fontsize=FONT_SIZE-2)
        ax.plot([parent_x, parent_x], [5.7, 4.9], linestyle='-', color=INK,
                linewidth=0.5, alpha=0.3)
    
    # Add title and annotation
    ax.text(5, 9.5, 'How Next-Word Prediction Powers Everything', 
           ha='center', fontsize=FONT_SIZE+2, fontweight='bold')
    
    # Sits just under the deepest leaf. At y=0.5 it left a bare band roughly two
    # units tall, which on a slide is wasted column width.
    ax.text(5, 2.0, 'Every branch is the same next-word predictor, pointed at a different problem',
           ha='center', fontsize=FONT_SIZE, style='italic',
           bbox=dict(boxstyle='round', facecolor=GOLD, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('figures/applications_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: applications_tree.pdf")


if __name__ == "__main__":
    create_applications_tree()
