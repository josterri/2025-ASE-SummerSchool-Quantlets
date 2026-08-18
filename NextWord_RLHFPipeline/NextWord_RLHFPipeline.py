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


def create_rlhf_pipeline():
    """Chart: RLHF in three stages left to right, human preference data enters at stage (b) only"""
    # 2.07in: the frame under this one carries a caption line, so the budget is
    # 153.1pt and not the 180.2pt an uncaptioned chart gets.
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7.2, 3.8), (5.79, 2.07))))
    if ROOM:
        # See create_multimodal_fusion.
        ax.remove()
        ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    # Raised floor and lowered ceiling in the room: the standard limits leave a
    # third of the figure empty once the title is stripped.
    ax.set_ylim(room_pick(0, 1.2), room_pick(6.2, 5.9))
    ax.axis('off')

    def box(x, y, w, h, text, facecolor, textcolor=PAPER, fontsize=FONT_SIZE-2):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=facecolor, edgecolor=INK, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, fontweight='bold')

    # Three stages of three lines each is nine runs of up to 21 characters,
    # which at 17pt is twice the 125pt a room column has. The room keeps the
    # stage letters and the one word that names each stage; what each is
    # trained on is still shown, in the caption row below.
    stage_w, stage_h, stage_y = room_pick(2.5, 2.9), room_pick(1.4, 1.5), 2.5
    stage_x = room_pick([0.4, 3.75, 7.1], [0.35, 3.55, 6.75])
    stage_colors = [TEAL, GOLD, CLARET]
    stage_texts = room_pick([
        '(a) Supervised\nfine-tuning on\ndemonstrations',
        '(b) Reward model on\nhuman preference\ncomparisons',
        '(c) Policy optimisation\nagainst the\nreward model',
    ], [
        '(a)\nFine-tune',
        '(b) Reward\nmodel',
        # "(c) Optimise" inks 117pt inside a 121pt box: 2pt of margin each side,
        # cream on claret, which reads as a word wedged against its own border.
        # A wider box is not available, since three boxes plus two arrows plus
        # two edges already spend the whole 417pt, so the word is the thing that
        # gives. "Policy tuning" is the same stage and 20pt narrower, and the
        # caption below it still says what it is tuned against.
        '(c) Policy\ntuning',
    ])
    for x, text, color in zip(stage_x, stage_texts, stage_colors):
        box(x, stage_y, stage_w, stage_h, text, color)

    y_mid = stage_y + stage_h / 2
    arrow_labels = ['base\npolicy', 'reward\nsignal']
    for i in range(2):
        x1 = stage_x[i] + stage_w
        x2 = stage_x[i + 1]
        ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle='->', color=INK, lw=1.5))
        arrow_note = ax.text((x1 + x2) / 2, y_mid + 0.35, arrow_labels[i], ha='center', va='bottom',
                fontsize=FONT_SIZE-2, color=SOFT)
        if ROOM:
            # The room columns leave a 20pt gap between boxes and these labels
            # are 50pt wide, so they print into the box text either side.
            arrow_note.remove()

    # Human preference data enters only at stage (b)
    # Wider than a stage box in the room, and centred on the same column. It is
    # the only box on its row, so nothing constrains its width, and at the stage
    # width "preferences" left 4pt of margin. The centre is computed from the
    # stage rather than picked, so the arrow below still lands on stage (b).
    human_w = room_pick(stage_w, 3.3)
    human_x = stage_x[1] - (human_w - stage_w) / 2
    human_y, human_h = room_pick(4.7, 4.3), room_pick(1.0, 1.35)
    box(human_x, human_y, human_w, human_h,
        room_pick('Human preference\ncomparisons: "A or B?"', 'Human\npreferences'), SOFT)
    ax.annotate('', xy=(human_x + human_w/2, stage_y + stage_h), xytext=(human_x + human_w/2, human_y),
                arrowprops=dict(arrowstyle='->', color=SOFT, lw=1.6))
    human_note = ax.text(human_x + human_w/2, human_y + human_h + 0.15,
            'The only human data in this pipeline',
            ha='center', va='bottom', fontsize=FONT_SIZE-2, color=CLARET, fontweight='bold')
    if ROOM:
        # The carried title of this figure is "RLHF: Human Judgment Enters at
        # One Stage", so on a room slide this line is the same sentence twice,
        # and the second copy prints into the box below it.
        human_note.remove()

    captions = room_pick([
        'trained on:\n(prompt, ideal response) pairs',
        'trained on:\nhuman preference comparisons',
        "trained on: reward model's\nscores, no new human labels",
    ], [
        'demos',
        'preferences',
        'reward scores',
    ])
    for x, caption in zip(stage_x, captions):
        ax.text(x + stage_w/2, stage_y - room_pick(0.2, 0.5), caption, ha='center', va='top',
                fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    ax.set_title('RLHF: Human Judgment Enters at One Stage',
                fontsize=FONT_SIZE+1, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figpath('rlhf_pipeline'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: rlhf_pipeline.pdf")


if __name__ == "__main__":
    create_rlhf_pipeline()
