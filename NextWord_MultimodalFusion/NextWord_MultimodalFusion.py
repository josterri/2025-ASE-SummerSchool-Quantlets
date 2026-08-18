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


def create_multimodal_fusion():
    """Chart: separate encoders converge on one shared space, then a single decoder"""
    # 1.73in: two keys, the carried title and the shared-geometry sentence, so
    # a two-line caption and a 128.1pt budget.
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7.2, 3.8), (5.79, 1.73))))
    if ROOM:
        # tight_layout sizes an axes so that everything hanging outside it still
        # fits the figure. On a diagram whose labels are drawn at 18pt and stick
        # well past the data area, that shrinks the axes rather than the text:
        # measured here, it left the axes 44pt wide inside a 417pt figure, so
        # the whole diagram was drawn at a quarter scale and every box label
        # printed over its neighbour. An axes placed by hand has no subplot
        # geometry for tight_layout to solve, so it keeps the full slide.
        ax.remove()
        ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10.5)
    # The room band is tighter, because the italic caption at the bottom is
    # blanked and carried to the slide, and it is centred on y=3 so the two
    # encoders, the shared space and the decoder share one axis.
    ax.set_ylim(room_pick(0, 0.6), room_pick(6, 5.4))
    ax.axis('off')

    def box(x, y, w, h, text, facecolor, textcolor=PAPER, fontsize=FONT_SIZE-1):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=facecolor, edgecolor=INK, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, fontweight='bold')

    # Two-word box labels are 145pt at 18pt and the boxes are 95pt wide on a
    # room slide, so the room breaks each one over two lines and gives the boxes
    # the height to hold them.
    #
    # The room widths are arithmetic, not taste. The axes is the whole slide, so
    # one x unit is 39.7pt: "encoder" inks 82pt, "Shared" 72pt and "Decoder"
    # 84pt, and every box is sized to hold its own label with about 10pt of
    # margin on each side. The first room pass sized these by eye and "Decoder"
    # came out 84pt of cream text in a 72pt box, so the word hung out of both
    # ends of the box it names.
    img_x, img_y, enc_w, enc_h = room_pick(0.4, 0.25), room_pick(3.9, 3.25), room_pick(2.4, 2.57), room_pick(1.0, 1.7)
    txt_x, txt_y = room_pick(0.4, 0.25), room_pick(1.1, 1.05)
    box(img_x, img_y, enc_w, enc_h, room_pick('Image encoder', 'Image\nencoder'), TEAL)
    box(txt_x, txt_y, enc_w, enc_h, room_pick('Text encoder', 'Text\nencoder'), CLARET)

    space_cx, space_cy = room_pick(5.4, 5.17), 3.0
    space = mpatches.Ellipse((space_cx, space_cy), width=room_pick(2.2, 2.7), height=room_pick(2.6, 3.0),
                              facecolor=GOLD, edgecolor=INK, linewidth=1.4, zorder=2)
    ax.add_patch(space)
    ax.text(space_cx, space_cy, room_pick('Shared\nrepresentation\nspace', 'Shared\nspace'), ha='center', va='center',
            fontsize=FONT_SIZE-1, color=PAPER, fontweight='bold', zorder=3)

    ax.annotate('', xy=(space_cx - room_pick(0.9, 1.2), space_cy + 0.7), xytext=(img_x + enc_w, img_y + enc_h/2),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.6,
                                 connectionstyle='arc3,rad=-0.15'))
    ax.annotate('', xy=(space_cx - room_pick(0.9, 1.2), space_cy - 0.7), xytext=(txt_x + enc_w, txt_y + enc_h/2),
                arrowprops=dict(arrowstyle='->', color=CLARET, lw=1.6,
                                 connectionstyle='arc3,rad=0.15'))

    dec_x, dec_w, dec_y, dec_h = room_pick(7.4, 7.52), room_pick(2.2, 2.72), room_pick(2.5, 2.15), room_pick(1.0, 1.7)
    box(dec_x, dec_y, dec_w, dec_h, 'Decoder', INK)
    ax.annotate('', xy=(dec_x, dec_y + dec_h/2), xytext=(space_cx + room_pick(1.1, 1.4), space_cy),
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.6))

    out_arrow = ax.annotate('', xy=(9.85, dec_y + dec_h/2), xytext=(dec_x + dec_w, dec_y + dec_h/2),
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.3))
    out_text = ax.text(9.95, dec_y + dec_h/2, 'Output', ha='left', va='center',
            fontsize=FONT_SIZE-1, color=INK, fontweight='bold')
    if ROOM:
        # "Output" leaves the room edition, arrow and all, and is reported as
        # carried rather than redrawn. It was tried as a fifth box, and five
        # boxes cannot be done: the four elements the carried title names, two
        # encoders and one shared space and one decoder, already need 318pt of
        # box on a 417pt slide once every label has a real margin. A fifth box
        # buys a word that is not in the title at the price of every box that
        # is. An arrow with nothing at the end of it would be worse than either.
        out_arrow.remove()
        out_text.remove()

    ax.text(5.4, 0.5, 'Two modalities land in the same geometry before anything is generated.',
            ha='center', fontsize=FONT_SIZE-1, style='italic', color=SOFT)

    ax.set_title('Multimodal Fusion: Two Encoders, One Shared Space, One Decoder',
                fontsize=FONT_SIZE+1, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figpath('multimodal_fusion'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: multimodal_fusion.pdf")


if __name__ == "__main__":
    create_multimodal_fusion()
