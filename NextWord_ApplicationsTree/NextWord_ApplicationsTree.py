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
                     SOFT, PAPER, RULE, sequential_cmap, diverging_cmap,
                     figpath, room_size, base_font_size, room_pt, room_pick, ROOM)
use_course_style()
sns.set_palette(CYCLE)


# Common settings
FONT_SIZE = base_font_size(8)
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
    # The full slot in the room, not the standard 10 by 6 fitted into it, which
    # comes back 4.08 inches wide and leaves 1.7 inches of slide unused. Those
    # inches are the difference between 88pt and 139pt per category box.
    fig, ax = plt.subplots(figsize=room_pick(room_size(10, 6), (5.79, 2.45)))

    ax.set_xlim(0, 10)
    # Bottom starts at 1.5, just under the caption. bbox_inches='tight' crops to
    # the axes extent rather than to the drawn content, so a ylim of 0 left a
    # band of empty axis below the caption that survived the crop.
    #
    # The room tree has no leaves, so it starts just under the category row.
    ax.set_ylim(room_pick(1.5, 4.6), 10)
    ax.axis('off')

    # Root: Next-word prediction
    # Box sizes are in DATA units and the type inside them is in POINTS, so the
    # two stop matching the moment the type triples. One axis unit is 41.7pt at
    # room scale, so this 2 unit box is 83pt holding 107pt of 20pt type: the
    # words ran outside the teal, and the words are PAPER, which is cream on a
    # white page. The room root was invisible rather than merely untidy.
    root_w, root_h = room_pick((2, 0.8), (3.3, 1.5))
    root_box = FancyBboxPatch((5 - root_w / 2, 8), root_w, root_h,
                              boxstyle="round,pad=0.05",
                              facecolor=TEAL, edgecolor=INK, linewidth=2)
    ax.add_patch(root_box)
    ax.text(5, 8 + root_h / 2, 'Next-Word\nPrediction', ha='center', va='center',
           color=PAPER, fontsize=FONT_SIZE+1, fontweight='bold')
    
    # Main branches. These are the three categories Hour 4 names in prose, in
    # the same order and wording. The chart previously branched Generation /
    # Understanding / Translation / Analysis, which the slide never mentions,
    # so the frame described a taxonomy its own diagram did not have.
    # Staggered onto two rows in the room, and the arithmetic says there was no
    # other option. "Generation", "Understanding" and "Transformation" are 37
    # characters, and a character at the 16pt floor is 9.7pt, so the three words
    # are 359pt of ink on an axes about 400pt wide. Forty points, shared between
    # two gaps and two margins, is not a gap. Spreading them further ran
    # "Transformation" off the right edge, which is worse: a text past the axes
    # makes the layout pass shrink the axes.
    #
    # Dropping the middle word one row gives each of them a whole row to itself
    # and costs nothing, because the slot has vertical room once the leaves are
    # gone. The words are unchanged, which is the point: they are the taxonomy
    # Hour 4 names in prose.
    main_applications = room_pick(
        [(2.0, 6, 'Generation'), (5.0, 6, 'Understanding'),
         (8.0, 6, 'Transformation')],
        [(2.2, 6.8, 'Generation'), (5.0, 5.4, 'Understanding'),
         (7.8, 6.8, 'Transformation')])
    
    for x, y, label in main_applications:
        # Same arithmetic as the root, and the same reason: at 16pt one character
        # is about 8.6pt, which is 0.206 of an axis unit, so a box wide enough for
        # its label is 0.103 units per character plus a little air. The leaf boxes
        # below have always done this; the category boxes were a fixed 1.2 units,
        # which is 50pt for a 112pt word.
        # No box round a category in the room, only the word.
        #
        # "Generation", "Understanding" and "Transformation" are 37 characters
        # together, which is 355pt of 16pt type on a 417pt page. That leaves 62pt
        # for two gaps and two margins, and it is enough for the WORDS to stand
        # clear of each other. It is not enough once each word is wrapped in a
        # box with padding on both sides: the boxes met, the words met, and the
        # reader merged two of them into 'UnderstandingTransformation'.
        #
        # The tree still reads as a tree, because the branch lines are what say
        # so. A rounded rectangle round a word is decoration in a diagram this
        # small, and it was decoration that cost the diagram its labels.
        if not ROOM:
            box = FancyBboxPatch((x - 0.6, y - 0.3), 1.2, 0.6,
                                 boxstyle="round,pad=0.03",
                                 facecolor=PAPER, edgecolor=TEAL, linewidth=1)
            ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=room_pick(FONT_SIZE - 1, FONT_SIZE - 3))
        # Connect to root
        ax.plot([5, x], [8 - 0.2, y+0.3], linestyle='-', color=INK, linewidth=1,
                alpha=0.5)
    
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

    # HAND-CARRIED for the room: the eleven leaf examples go on the slide, as a
    # line under each category name.
    #
    # The room tree keeps the root and the three categories and drops the leaves.
    # That choice is about which words are load-bearing. "Generation",
    # "Understanding" and "Transformation" are the taxonomy Hour 4 names in
    # prose, and this chart was once rebuilt precisely because it branched into
    # categories the slide never mentioned; shortening them to fit would put that
    # defect back. The leaves are examples, and an example reads as well in a
    # sentence as in a box.
    #
    # With the leaves gone the three category boxes have the whole 417pt to share,
    # which is 139pt each, and "Transformation" is 133pt of room type. Keeping the
    # leaves would have meant 88pt each.
    for parent_x, labels in (() if ROOM else leaf_groups):
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
    plt.savefig(figpath('applications_tree'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: applications_tree.pdf")


if __name__ == "__main__":
    create_applications_tree()
