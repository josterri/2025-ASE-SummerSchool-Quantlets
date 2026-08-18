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


def create_rag_pipeline():
    """Chart: RAG pipeline. The retrieval loop stays visually distinct from a plain LLM call"""
    # 1.73in: the carried title and the sentence naming the stage this room
    # edition dropped are both keys and both too long for a heading, so the
    # frame carries two caption lines and the budget is 128.1pt.
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7.5, 4.0), (5.79, 1.71))))
    if ROOM:
        # See create_multimodal_fusion: tight_layout shrinks an axis('off')
        # diagram to fit the labels that hang outside it. Placed by hand, the
        # axes keeps the full slide.
        ax.remove()
        ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10.2)
    # The room floor is raised because the caption band at the bottom is blanked
    # and carried to the slide, leaving a third of the figure empty.
    ax.set_ylim(room_pick(0, 1.5), room_pick(6.3, 6.4))
    ax.axis('off')

    def box(x, y, w, h, text, facecolor, textcolor=PAPER, fontsize=FONT_SIZE-1):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=facecolor, edgecolor=INK, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, fontweight='bold')

    def arrow(x1, y1, x2, y2, color=INK, lw=1.3):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    y_row, h_row = 4.4, room_pick(1.0, 1.1)
    # Six boxes across a room slide leaves each one 40pt of width, and every
    # label in the standard set is two or three times that at 18pt.
    #
    # The room kept FOUR of them and that was one too many. One x unit is 40.9pt
    # here, and four labels plus a "Documents" store need 391pt of box on a 417pt
    # slide before a single arrow is drawn: the Answer box ran off the right edge
    # with its border clipped, and "Documents" inked 114pt inside a 102pt box, so
    # the one word naming the store hung out of both ends of it.
    #
    # Three stages fit with 46pt arrows and 8pt of margin inside every box. The
    # stage that goes is "Prompt", and it is not lost: the italic line below is
    # over the carried-text threshold, so the room slide is already required to
    # print "the prompt combines the question with retrieved context". The two
    # that stay are the two the carried title is about, a question answered from
    # documents fetched at answer time.
    stages = room_pick([
        (0.2, 1.3, 'User\nquestion', SOFT),
        (1.75, 1.3, 'Embed\nquery', TEAL),
        (3.3, 1.9, 'Vector store:\ntop-k documents', TEAL),
        (5.45, 1.7, 'Assemble\nprompt', GOLD),
        (7.4, 0.9, 'LLM', CLARET),
        (8.55, 1.35, 'Grounded\nanswer', INK),
    ], [
        (0.30, 2.62, 'Question', SOFT),
        (4.05, 2.50, 'Retrieve', TEAL),
        (7.68, 2.22, 'Answer', INK),
    ])
    for x, w, text, color in stages:
        box(x, y_row, w, h_row, text, color)

    y_mid = y_row + h_row / 2
    for (x1, w1, _, _), (x2, _, _, _) in zip(stages[:-1], stages[1:]):
        arrow(x1 + w1, y_mid, x2, y_mid)

    # Document collection: retrieved fresh at answer time (the loop)
    # The room store is 3.18 units, WIDER than the Retrieve box above it, because
    # "Documents" is the widest label on this chart at 114pt and the box has to
    # hold it. Everything else is spaced so the dashed loop can be drawn around
    # the store and the stage together without cutting into either neighbour.
    doc_x, doc_w, doc_y, doc_h = room_pick(3.3, 3.71), room_pick(1.9, 3.18), room_pick(2.2, 1.9), room_pick(0.8, 1.1)
    box(doc_x, doc_y, doc_w, doc_h, room_pick('Document\ncollection', 'Documents'), TEAL)
    vs_x, vs_w = stages[room_pick(2, 1)][0], stages[room_pick(2, 1)][1]
    ax.annotate('', xy=(vs_x + vs_w/2, y_row), xytext=(doc_x + doc_w/2, doc_y + doc_h),
                arrowprops=dict(arrowstyle='<->', color=TEAL, lw=1.6))

    rect_x0, rect_x1 = room_pick(1.6, 3.55), room_pick(5.35, 7.05)
    rect_y0, rect_y1 = room_pick(2.05, 1.75), room_pick(5.55, 5.65)
    loop = Rectangle((rect_x0, rect_y0), rect_x1 - rect_x0, rect_y1 - rect_y0,
                      fill=False, edgecolor=TEAL, linewidth=1.4, linestyle='--')
    ax.add_patch(loop)
    ax.text((rect_x0 + rect_x1) / 2, rect_y1 + 0.15,
            room_pick('Retrieval: fetched fresh, every query', 'Fresh every query'),
            ha='center', va='bottom', fontsize=FONT_SIZE-2, color=TEAL, fontweight='bold')

    ax.text(5.1, 1.25,
            'The prompt combines the question with retrieved context.\n'
            'A plain LLM call skips retrieval and answers only from weights learned at training time.',
            ha='center', va='center', fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    ax.set_title('RAG: Documents Retrieved at Answer Time, Not Baked into Weights',
                fontsize=FONT_SIZE+1, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figpath('rag_pipeline'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: rag_pipeline.pdf")


if __name__ == "__main__":
    create_rag_pipeline()
