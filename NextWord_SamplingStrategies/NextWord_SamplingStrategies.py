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


def create_sampling_strategies():
    """Chart: greedy, temperature, top-k and nucleus sampling reshape one distribution"""
    # Four panels side by side give each one 104pt and five word labels, which
    # is why the standard figure tilts them. Tilting is banned in a room chart,
    # so the room stacks two panels instead: the tick labels are then drawn once
    # across the full slide, upright, with 71pt each.
    # 2.10in: four keys, two panel names, the carried title and the outline key
    # that also marks the vocabulary indicative, so three caption lines and a
    # 103.1pt budget.
    fig, axes = plt.subplots(*room_pick((1, 4), (2, 1)),
                             figsize=room_size(*room_pick((10, 3.6), (5.79, 2.20))), sharey=True)

    words = ['mat', 'rug', 'floor', 'table', 'chair']
    base_probs = np.array([0.50, 0.22, 0.14, 0.09, 0.05])
    x = np.arange(len(words))
    width = 0.6

    greedy_probs = np.zeros_like(base_probs)
    greedy_probs[np.argmax(base_probs)] = base_probs.max()

    temperature = 1.5
    logits = np.log(base_probs)
    temp_probs = np.exp(logits / temperature)
    temp_probs = temp_probs / temp_probs.sum()

    k = 3
    topk_idx = np.argsort(base_probs)[::-1][:k]
    topk_probs = np.zeros_like(base_probs)
    topk_probs[topk_idx] = base_probs[topk_idx] / base_probs[topk_idx].sum()

    p_threshold = 0.9
    order = np.argsort(base_probs)[::-1]
    cum = np.cumsum(base_probs[order])
    cutoff = int(np.searchsorted(cum, p_threshold) + 1)
    keep_idx = order[:cutoff]
    nucleus_probs = np.zeros_like(base_probs)
    nucleus_probs[keep_idx] = base_probs[keep_idx] / base_probs[keep_idx].sum()

    # Indexed off a list of specifications rather than built twice, because
    # axes[2] does not exist in the room and room_pick evaluates both of its
    # arguments. The two kept are the extremes: no truncation, and truncation.
    strategies = [
        (greedy_probs, CLARET, '(a) Greedy'),
        (temp_probs, GOLD, '(b) Temperature, T=1.5'),
        (topk_probs, TEAL, '(c) Top-k, k=3'),
        (nucleus_probs, INK, '(d) Nucleus, p=0.9'),
    ]
    panels = list(zip(axes.flat, room_pick(strategies, [strategies[0], strategies[2]])))

    for position, (ax, (active, color, title)) in enumerate(panels):
        # A 1.2pt rule-coloured outline is a smudge on a 60pt room panel, and
        # the outline is what carries "this is the distribution before the
        # strategy touched it".
        ax.bar(x, base_probs, width, facecolor='none', edgecolor=room_pick(RULE, SOFT),
               linewidth=room_pick(1.2, 1.8), zorder=1)
        ax.bar(x, active, width * 0.55, color=color, edgecolor=INK, linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        # Word labels once, under the lower panel, in the room.
        ax.set_xticklabels(room_pick(words, words if position == len(panels) - 1 else [''] * len(words)),
                           rotation=room_pick(40, 0), ha=room_pick('right', 'center'), fontsize=FONT_SIZE-2)
        ax.set_title(title, fontsize=FONT_SIZE-1, fontweight='bold')
        ax.set_ylim(0, 0.62)
        if ROOM:
            # One labelled tick a panel. This figure owes four keys, so its
            # frame carries three caption lines and the figure is cut to
            # 103.1pt, which leaves each stacked panel about 38pt. The default
            # 0.0 and 0.5 then put the upper panel's 0.0 within 3pt of the
            # lower panel's 0.5, and tools/check_room_charts.py measured the
            # pair overlapping across 27.4pt of width. Two numbers from
            # different panels touching read as one, and the 0.0 is the one
            # carrying least: the bars stand on a drawn axis at zero.
            ax.set_yticks([0.5])
        ax.grid(True, alpha=0.3, axis='y')

    # The room pass strips the titles, so each panel is named by its y label
    # instead, and in the room that label is set HORIZONTALLY.
    #
    # It was rotated and anchored to the bottom of its own panel, which was an
    # answer to the right problem and made a worse one. A stacked panel is about
    # 55pt tall, "Greedy" set vertically at 18pt is 63pt, so a centred pair meet
    # at the panel boundary and read as one word: the gate cannot see that,
    # because they are two spans. Anchoring them low stopped them meeting and
    # ran "Greedy" off the top of the figure instead, which is where the band of
    # dead space above the first panel came from.
    #
    # Turned flat the constraint disappears, because a panel name is then
    # measured against the WIDTH of the margin and not the height of the panel.
    # It costs about 55pt on the left, which this chart has: five bars across a
    # full slide is 64pt each and the widest tick label is "floor" at 45pt.
    ylabel_kw = ({"rotation": 0, "ha": "right", "va": "center", "labelpad": 10}
                 if ROOM else {"y": 0.5, "ha": "center"})
    axes[0].set_ylabel(room_pick('Probability', 'Greedy'), fontsize=FONT_SIZE-1,
                       **ylabel_kw)
    if ROOM:
        axes[1].set_ylabel('Top-k', fontsize=FONT_SIZE-1, **ylabel_kw)

    fig.suptitle('Sampling Strategies Reshape the Same Distribution: "the cat sat on the ___"',
                 fontsize=FONT_SIZE+1, fontweight='bold')
    fig.text(0.5, 0.02,
             'Outline: original distribution. Filled: surviving probability. '
             'Indicative toy vocabulary, not measured model output.',
             ha='center', fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    plt.savefig(figpath('sampling_strategies'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: sampling_strategies.pdf")


if __name__ == "__main__":
    create_sampling_strategies()
