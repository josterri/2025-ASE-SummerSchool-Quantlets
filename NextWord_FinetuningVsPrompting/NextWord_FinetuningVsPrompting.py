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


def create_finetuning_vs_prompting():
    """Chart 13: Fine-tuning vs prompting comparison"""
    # 1.98in, not the 2.45in slot. The probe estimates one caption line here and
    # the frame needs two: the carried marker alone is 98 characters once its
    # numbers are in, "Accuracy Comparison" is the frame heading, and a
    # 33-character heading plus one 62-character line cannot hold both. Two
    # lines is a 128.1pt budget.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=room_size(*room_pick((10, 4), (5.79, 1.72))))

    # Accuracy comparison. Two panels of four two-line categories is 32 tick
    # label draws, and the room budget is five. The room keeps the accuracy
    # panel; the effort side is not lost, because the caption below already
    # states it in words and the room pass carries that caption to the slide.
    methods = room_pick(['Zero-shot\nPrompting', 'Few-shot\nPrompting', 'Fine-tuning\n(1K examples)',
                         'Fine-tuning\n(10K examples)'],
                        ['Zero\nshot', 'Few\nshot', 'FT\n1K', 'FT\n10K'])
    accuracy = [72, 85, 89, 92]
    time_required = [0.001, 0.01, 24, 72]  # Hours
    
    # sequential_cmap runs white to PAPER to the base colour, so the low end of
    # the ramp is cream on white. On paper that is a light bar; projected onto a
    # small screen the first two bars had no edge a viewer could find, and the
    # 0.7 alpha took what was left. The room starts the ramp past PAPER and
    # nearly drops the alpha.
    colors = sequential_cmap(CLARET)(np.linspace(room_pick(0.35, 0.62), 0.95, len(methods)))
    bars1 = ax1.bar(methods, accuracy, color=colors, alpha=room_pick(0.7, 0.95), edgecolor=INK, linewidth=1)

    # Add value labels
    for bar, acc in zip(bars1, accuracy):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + room_pick(0.5, 1.5),
                f'{acc}%', ha='center', va='bottom', fontsize=FONT_SIZE-1, fontweight='bold')

    ax1.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)
    ax1.set_title('Accuracy Comparison', fontsize=FONT_SIZE+1, fontweight='bold')
    # The room axis starts at zero. This is the one chart in the hour where the
    # length of the mark IS the quantity, and a floor of 65 drew 72% as a stub
    # beside 92% as a full bar: about a fifth of the height for about four fifths
    # of the value. The four value labels stay for the same reason they are worth
    # their width here, since from zero the bars are close in height and the
    # numbers are what separates them.
    ax1.set_ylim(room_pick(65, 0), room_pick(95, 108))
    if ROOM:
        # Two ticks, not three. The frame under this chart carries two caption
        # lines, so the figure is cut to about 130pt and the plot itself is
        # roughly 70pt of that: three labels of 19pt do not fit in 70pt and
        # tools/check_room_charts.py measured 0 printing into 50 and 50 into
        # 100. The four bar values are labelled on top of the bars anyway, so
        # the middle tick was the one carrying least.
        ax1.set_yticks([0, 100])
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

    if ROOM:
        # One panel in the room. Removing the axes is not enough on its own:
        # the survivor keeps the left cell of a 1x2 gridspec, so tight_layout
        # would leave it on the left half of the slide. Reassigning it to a 1x1
        # grid is what makes it fill the slot.
        ax2.remove()
        ax1.set_subplotspec(fig.add_gridspec(1, 1)[0])

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
    plt.savefig(figpath('finetuning_vs_prompting'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: finetuning_vs_prompting.pdf")


if __name__ == "__main__":
    create_finetuning_vs_prompting()
