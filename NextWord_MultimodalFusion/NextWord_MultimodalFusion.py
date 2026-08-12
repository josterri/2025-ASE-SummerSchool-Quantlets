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


def create_multimodal_fusion():
    """Chart: separate encoders converge on one shared space, then a single decoder"""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, text, facecolor, textcolor=PAPER, fontsize=FONT_SIZE-1):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=facecolor, edgecolor=INK, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, fontweight='bold')

    img_x, img_y, enc_w, enc_h = 0.4, 3.9, 2.4, 1.0
    txt_x, txt_y = 0.4, 1.1
    box(img_x, img_y, enc_w, enc_h, 'Image encoder', TEAL)
    box(txt_x, txt_y, enc_w, enc_h, 'Text encoder', CLARET)

    space_cx, space_cy = 5.4, 3.0
    space = mpatches.Ellipse((space_cx, space_cy), width=2.2, height=2.6,
                              facecolor=GOLD, edgecolor=INK, linewidth=1.4, zorder=2)
    ax.add_patch(space)
    ax.text(space_cx, space_cy, 'Shared\nrepresentation\nspace', ha='center', va='center',
            fontsize=FONT_SIZE-1, color=PAPER, fontweight='bold', zorder=3)

    ax.annotate('', xy=(space_cx - 0.9, space_cy + 0.7), xytext=(img_x + enc_w, img_y + enc_h/2),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.6,
                                 connectionstyle='arc3,rad=-0.15'))
    ax.annotate('', xy=(space_cx - 0.9, space_cy - 0.7), xytext=(txt_x + enc_w, txt_y + enc_h/2),
                arrowprops=dict(arrowstyle='->', color=CLARET, lw=1.6,
                                 connectionstyle='arc3,rad=0.15'))

    dec_x, dec_w, dec_y, dec_h = 7.4, 2.2, 2.5, 1.0
    box(dec_x, dec_y, dec_w, dec_h, 'Decoder', INK)
    ax.annotate('', xy=(dec_x, dec_y + dec_h/2), xytext=(space_cx + 1.1, space_cy),
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.6))

    ax.annotate('', xy=(9.85, dec_y + dec_h/2), xytext=(dec_x + dec_w, dec_y + dec_h/2),
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.3))
    ax.text(9.95, dec_y + dec_h/2, 'Output', ha='left', va='center',
            fontsize=FONT_SIZE-1, color=INK, fontweight='bold')

    ax.text(5.4, 0.5, 'Two modalities land in the same geometry before anything is generated.',
            ha='center', fontsize=FONT_SIZE-1, style='italic', color=SOFT)

    ax.set_title('Multimodal Fusion: Two Encoders, One Shared Space, One Decoder',
                fontsize=FONT_SIZE+1, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/multimodal_fusion.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: multimodal_fusion.pdf")


if __name__ == "__main__":
    create_multimodal_fusion()
