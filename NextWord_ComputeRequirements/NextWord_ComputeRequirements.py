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


def create_compute_requirements():
    """Chart 18: Compute requirements graph"""
    # Room: an explicit canvas, not room_size(8, 5). Fitting an 8x5 into the
    # slot preserves aspect, which means the height binds and the result is
    # 3.92in wide on a 5.79in slide: a third of the width thrown away, and the
    # figure still 27pt taller than the frame that has to carry its caption.
    # Full width and a shorter canvas buys both. Height measured, not chosen:
    # tools/probe_room_fig_height.py names the target.
    fig, ax = plt.subplots(figsize=room_pick(room_size(8, 5), (5.79, 2.25)))
    
    # Model data: name, year, FLOPs (in log scale)
    models = [
        ('N-gram', 1980, 1e6),
        ('Small NN', 1990, 1e9),
        ('Word2Vec', 2013, 1e12),
        ('LSTM', 2015, 1e14),
        ('Transformer-Base', 2017, 1e16),
        ('BERT', 2018, 1e18),
        ('GPT-2', 2019, 1e19),
        ('GPT-3', 2020, 1e23),
        ('PaLM', 2022, 2.5e24),
        ('GPT-4', 2023, 1e25)
    ]
    
    names = [m[0] for m in models]
    years = [m[1] for m in models]
    flops = [m[2] for m in models]
    
    # Create scatter plot with size based on compute
    sizes = [np.log10(f) * 20 for f in flops]
    # Ramp starts at 0.35: sequential_cmap runs white -> PAPER -> base, so the
    # first markers would otherwise be white on a white background.
    colors = sequential_cmap()(np.linspace(0.35, 1, len(models)))
    
    scatter = ax.scatter(years, flops, s=sizes, c=colors, alpha=0.6, 
                        edgecolors=INK, linewidth=1)
    
    # Add labels for significant models
    for name, year, flop in models:
        # Three labels in the room, not five. GPT-3 at 2020 and GPT-4 at 2023
        # are two years and two orders of magnitude apart, which is about 20pt on
        # a room axes, so their labels sit on each other. BERT goes for the same
        # reason. The points stay; only the naming thins out.
        # Two labels in the room, not three. Three was already a cut from five,
        # and it was still one too many: "Word2Vec" is 8 characters of 19pt type
        # sitting in the middle of the rising cluster, so it printed straight
        # across the bubbles either side of it. The two that stay are the ENDS of
        # the arc, which is the whole argument of the chart: 1980 to 2023, a
        # million FLOPs to a yotta. Word2Vec goes on the slide.
        if name in room_pick(['N-gram', 'Word2Vec', 'BERT', 'GPT-3', 'GPT-4'],
                             ['N-gram', 'GPT-4']):
            # N-gram sits at 1980, the first year on the axis, so a centred label
            # reaches back over the y tick labels. In the room that is a
            # collision with "1M"; at 8pt it was a near miss nobody saw. It is
            # pushed right and left-aligned there instead.
            first = ROOM and name == 'N-gram'
            # GPT-4 has the mirror problem at the other end: 2023 with 2025 as
            # the last year on the axis, so a centred label runs off the plot.
            # Right-aligning it puts the word over the empty top-left of the
            # decade band, where the curve has not reached yet.
            last = ROOM and name == 'GPT-4'
            if first:
                # Beside the bubble and vertically centred on it, not above and
                # away. At a 14pt offset the word floated in the middle of the
                # empty first band, eight years of axis from the point it names,
                # and a label that far from its marker labels nothing.
                #
                # va is passed ONLY here, never as its own default on the shipped
                # path. 'baseline' is what annotate already uses, and passing it
                # anyway is the linespacing bet in another costume: a parameter
                # that changes nothing visible can still change the bytes, and
                # the bytes are what is guaranteed.
                offset, align = (10, 0), 'left'
                extra = {"va": "center"}
            elif last:
                offset, align, extra = (-6, 8), 'right', {}
            else:
                offset, align, extra = (0, 5), 'center', {}
            ax.annotate(name, xy=(year, flop), xytext=offset,
                       textcoords='offset points',
                       ha=align, fontsize=FONT_SIZE-1, **extra)
    
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=FONT_SIZE)
    # A y label is rotated, so its LENGTH is measured against the axes HEIGHT,
    # and the room axes is about 150pt tall. 'Training Compute (FLOPs)' is 228pt
    # of 19pt type, so it overflowed and tight_layout shrank the axes to make
    # room for it. 'Training FLOPs' is 133pt and fits.
    # 'FLOPs', not 'Training FLOPs'. A rotated label is measured against the
    # axes HEIGHT, and at 19pt the longer one is 137.9pt of ink, which pinned
    # the saved box at 140.7pt no matter what canvas it was given: two canvas
    # heights 0.17in apart produced byte-identical dimensions. The word
    # "Training" is not lost, it is in the caption, which says training compute
    # has been doubling every 3.4 months.
    ax.set_ylabel(room_pick('Training Compute (FLOPs)', 'FLOPs'),
                  fontsize=FONT_SIZE)
    ax.set_title('Exponential Growth in Compute Requirements', 
                fontsize=FONT_SIZE+1, fontweight='bold')
    
    # Custom y-axis labels
    # Seven decade labels over 150pt of room axes is 21pt apart for 20pt type,
    # so they print on each other. Four is the most that fits, and every one of
    # them still lands on a labelled decade.
    # Three, not four. The room axes is about 94pt tall once the x label and the
    # x tick labels have taken their share of a 176pt slot, and four 20pt labels
    # in 94pt are 23pt apart. Three are 47pt apart and still span the range.
    ax.set_yticks(room_pick([1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24],
                            [1e6, 1e15, 1e24]))
    if not ROOM:
        ax.set_yticklabels(['1M', '1B', '1T', '1P', '1E', '1Z', '1Y'])
    # No labels set in the room, deliberately. The room pass installs its own
    # formatter on every log axis, to keep mathtext exponents off a small screen,
    # and that formatter replaces whatever is set here. Setting them anyway
    # produced two sets of labels on one axis. The formatter's ladder now runs to
    # yotta, so it prints exactly these words by itself.
    
    # Add doubling time annotation
    # HAND-CARRIED for the room: "Doubling every 3.4 months" goes on the slide.
    #
    # Moving it was tried first and made things worse. A text drawn in DATA
    # coordinates inside an axes and wider than that axes does not overflow: the
    # layout pass shrinks the AXES to fit it, and this box is 143pt of 19pt type
    # plus a rounded frame in an axes about 330pt wide. The chart came back with
    # the plot squeezed to a sliver under a gold slab. It is the same mechanism
    # that flattened sparse_vs_dense, and between them they are most of what was
    # wrong with the 42.
    if not ROOM:
        ax.text(2015, 1e22, 'Doubling every\n3.4 months!', fontsize=FONT_SIZE,
               bbox=dict(boxstyle='round', facecolor=GOLD, alpha=0.5),
               ha='center', fontweight='bold')
    
    # Add hardware generations
    # One word per era in the room. The legend sits inside the axes and
    # 'TPU/Cluster Era' alone is 135pt of it. The bands are still there and still
    # in the same three colours; only the key gets shorter.
    # 10 per cent tint is right on paper, where a reader is a foot from the page
    # and the bands are a whisper behind the data. Projected small it is nothing
    # at all: all three read as white. That matters more in the room than on
    # paper, because the room has NO legend, so the slide names three bands a
    # student cannot see. A carried sentence pointing at invisible ink is worse
    # than no sentence. 22 per cent separates them and still sits behind the
    # bubbles.
    band_alpha = room_pick(0.1, 0.22)
    ax.axvspan(1980, 1995, alpha=band_alpha, color=SOFT,
               label=room_pick('CPU Era', 'CPU'))
    ax.axvspan(1995, 2010, alpha=band_alpha, color=TEAL,
               label=room_pick('GPU Era', 'GPU'))
    ax.axvspan(2010, 2024, alpha=band_alpha, color=GOLD,
               label=room_pick('TPU/Cluster Era', 'TPU'))
    
    # HAND-CARRIED for the room: the era key, "CPU Era, GPU Era, TPU and cluster
    # era", goes on the slide. Three entries of 20pt type is a 60pt legend box in
    # a 94pt axes, which is not a key beside a chart, it is a chart beside a key.
    # The three shaded bands stay, in the same three colours, and the slide names
    # them once.
    if not ROOM:
        ax.legend(loc='lower right', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1975, 2025)
    ax.set_ylim(1e5, 1e26)
    
    plt.tight_layout()
    plt.savefig(figpath('compute_requirements'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: compute_requirements.pdf")


if __name__ == "__main__":
    create_compute_requirements()
