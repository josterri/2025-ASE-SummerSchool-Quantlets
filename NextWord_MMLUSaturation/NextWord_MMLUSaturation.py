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
                     SOFT, PAPER, RULE, sequential_cmap, diverging_cmap,
                     figpath, room_size, base_font_size, room_pt, room_pick, ROOM,
                     ROOM_SLOT_IN, ROOM_SLOT_CAPTIONED_IN)
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


# A KEY ABOVE THE PLOT COSTS 46pt, AND MOST OF THAT IS THE CARD.
#
# Four charts here moved their legend above the axes, because at 18pt a two-row
# key inside a room axes is half its height and a one-row key is wider than it.
# Above the plot is the right place. What was not measured is the price:
# tools/probe_room_hour3_layout.py reports performance_benchmarks with a 39pt
# plot in a 176pt figure, 271x36pt of legend above it, and a 9pt gap.
#
# Most of that 36pt is not the words. matplotlib's default borderpad is 0.4 of
# the font size, so 7pt above and 7pt below at 18pt, and borderaxespad adds
# another 9pt between the card and the axes. figures_src/room.mplstyle also
# gives every legend an opaque backing, which is exactly right for a key sitting
# INSIDE the axes with data lines running under it, and buys nothing above the
# plot where nothing is drawn.
#
# So: above-axes keys are frameless and tightly padded in the room edition only.
# The standard edition passes none of these and cannot move. Measured after the
# change, the same chart's plot goes from 39pt to about 85pt, which is the
# difference between a bar chart and a row of stripes.
ROOM_KEY_ABOVE = ({'frameon': False, 'borderpad': 0.1, 'borderaxespad': 0.1,
                   'handlelength': 1.4, 'handletextpad': 0.5,
                   'columnspacing': 1.4} if ROOM else {})


def create_mmlu_saturation():
    """Chart 15: Benchmark saturation and replacement (indicative)"""
    fig, ax = plt.subplots(figsize=room_size(*room_pick((7, 3.3), ROOM_SLOT_IN)))

    years = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    mmlu = np.array([28, 35, 44, 68, 86, 88, 90, 91])

    ax.plot(years, mmlu, marker='o', color=CLARET, linewidth=2.2, markersize=4,
            label=room_pick('MMLU accuracy', 'MMLU'))
    ax.axhline(y=90, color=SOFT, linestyle='--', linewidth=1.4,
               label=room_pick('Human-expert ceiling', 'expert ceiling'))

    # Room edition: two successor benchmarks rather than three. ARC-AGI-2 at
    # (2025.0, 6) and Humanity's Last Exam at (2025.3, 11) are 14pt apart and
    # print over each other; two points are enough to show that the replacements
    # start where MMLU finished.
    successors = room_pick(
        [(2023.8, 38, 'GPQA Diamond'),
         (2025.0, 6, 'ARC-AGI-2'),
         (2025.3, 11, "Humanity's Last Exam")],
        [(2023.8, 38, 'GPQA'),
         (2025.0, 6, 'ARC-AGI-2')])
    for x, y, name in successors:
        ax.scatter(x, y, s=36, color=TEAL, zorder=5)
        # Room edition: BOTH successors are labelled to the LEFT of their point.
        #
        # They separate vertically on their own, 38 against 6 accuracy points,
        # once the plot is tall enough; what does not resolve itself is the
        # right-hand edge. ARC-AGI-2 sits at 2025 on an axis that ends at 2026.7,
        # and its label is 88pt long, so drawn to the right it ended 32pt past
        # the axes. That is not an overflow, it is a wider saved page and a
        # shorter plot, and it is why this figure saved at 384pt where its
        # neighbours saved at 376.
        #
        # Lifting a label instead was tried and was worse: 16pt of lift printed
        # GPQA across the MMLU curve and the expert-ceiling line, and detached it
        # from its own dot. Nothing went red, because text over a LINE is text
        # over graphics and the collision rule sees only text over text.
        ax.annotate(name, xy=(x, y),
                    xytext=room_pick((3, 4), (-4, 4)),
                    textcoords='offset points',
                    ha=room_pick('left', 'right'),
                    fontsize=FONT_SIZE-2, color=TEAL)

    # Room edition: no x-axis label, on the same argument as the timeline. Three
    # four-digit ticks reading 2020, 2023, 2026 are years, and the 24pt this
    # costs was a fifth of the plot: the axes here was 42pt tall.
    ax.set_xlabel(room_pick('Year', ''), fontsize=FONT_SIZE)
    ax.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)
    ax.set_title('Benchmarks Saturate, Then Get Replaced',
                fontsize=FONT_SIZE+1, fontweight='bold')
    ax.set_xlim(2018.5, 2026.7)
    ax.set_ylim(0, 100)
    if ROOM:
        # Eight four-digit year ticks are 44pt of ink at 46pt of spacing.
        ax.set_xticks([2020, 2023, 2026])
    # Room edition: the legend moves out of the corner the successor benchmarks
    # live in. At 17pt two entries stand 44pt tall in a 120pt axes, which reaches
    # up to the GPQA point at y=38.
    # Room edition: the key goes above the plot, in one row. Inside the axes its
    # dashed handle sits at about y=42 and reads as a second horizontal dashed
    # line in the data, which is worse than crowded: it is wrong.
    ax.legend(loc=room_pick('lower left', 'lower center'),
              bbox_to_anchor=room_pick(None, (0.5, 1.01)),
              ncols=room_pick(1, 2), fontsize=FONT_SIZE-2, **ROOM_KEY_ABOVE)
    ax.grid(True, alpha=0.3)

    ax.text(2019.2, 72, 'Indicative levels, not exact reported scores',
            ha='left', fontsize=FONT_SIZE-2, style='italic', color=SOFT)

    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath('mmlu_saturation'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: mmlu_saturation.pdf")


if __name__ == "__main__":
    create_mmlu_saturation()
