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
Generate charts for Hour 2: Smarter Prediction (RNNs to Transformers)
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch


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


def create_attention_heatmap():
    """Chart 6: Attention heatmap visualization"""
    fig, ax = plt.subplots(figsize=room_size(*room_pick((8, 6), (5.79, 2.45))))

    # Example sentence. Eight words need eight tick labels on each axis, and at
    # 16pt that is roughly three times the width a room chart has. The room
    # sentence is a GRAMMATICAL subset of the same words, in the same order, so
    # that a reader can parse the axis; it keeps the causal-mask shape and the
    # "later words look back at the subject" pattern, which is what the frame
    # teaches. The boosted column is j == 1 ('cat') for i > 2, so in the room
    # the hot cell is the tail of the verb phrase looking back at its subject.
    # Every room word is four characters or fewer on purpose: four unrotated
    # tick labels share 5.79in with a colorbar, and 'sleeping' ran into 'was'.
    words = room_pick(['The', 'cat', 'that', 'was', 'sleeping', 'peacefully', 'woke', 'up'],
                      ['The', 'cat', 'woke', 'up'])
    n_words = len(words)
    
    # Create attention matrix (synthetic but realistic pattern)
    attention = np.zeros((n_words, n_words))
    
    # Fill with realistic attention patterns
    np.random.seed(42)
    for i in range(n_words):
        for j in range(i+1):  # Only attend to previous words (causal mask)
            if i == j:
                attention[i, j] = 0.3 + np.random.random() * 0.3  # Self-attention
            elif j == 1 and i > 2:  # 'cat' gets attention from later words
                attention[i, j] = 0.2 + np.random.random() * 0.3
            elif j == 4 and i > 5:  # 'sleeping' gets attention
                attention[i, j] = 0.15 + np.random.random() * 0.2
            else:
                attention[i, j] = np.random.random() * 0.15
    
    # Normalize rows to sum to 1
    for i in range(n_words):
        if attention[i, :i+1].sum() > 0:
            attention[i, :i+1] /= attention[i, :i+1].sum()
    
    # Create heatmap
    # 'auto' in the room, so the grid fills the slide slot instead of leaving a
    # square island with dead space either side of it.
    im = ax.imshow(attention, cmap=sequential_cmap(TEAL), aspect=room_pick('equal', 'auto'), vmin=0, vmax=0.6)
    ax.grid(False)  # gridlines must not cross heatmap cells

    # Set ticks and labels
    ax.set_xticks(np.arange(n_words))
    ax.set_yticks(np.arange(n_words))
    ax.set_xticklabels(words, rotation=room_pick(45, 0), ha=room_pick('right', 'center'))
    ax.set_yticklabels(words)
    
    # Add grid
    for i in range(n_words + 1):
        ax.axhline(y=i-0.5, color=SOFT, linewidth=room_pick(0.5, 1.0), alpha=0.3)
        ax.axvline(x=i-0.5, color=SOFT, linewidth=room_pick(0.5, 1.0), alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    # No colorbar label in the room. It is set rotated, so it costs 32pt of the
    # 381pt width and stands 88pt tall beside a heatmap that had only 220pt of
    # width left; the bar's own 0.0 and 0.5 ticks already say which end is which.
    cbar.set_label(room_pick('Attention Weight', ''), fontsize=FONT_SIZE)

    ax.set_xlabel(room_pick('Input Words (Attended To)', 'Attended to'), fontsize=FONT_SIZE)
    # "Attending from" is 143pt of rotated ink against an axes 105pt high, so it
    # overhangs the chart it labels top and bottom. One word does the same work
    # while the x-axis beside it still reads "Attended to".
    ax.set_ylabel(room_pick('Output Words (Attending From)', 'From'), fontsize=FONT_SIZE)
    ax.set_title('Self-Attention: Each Word Looks at Relevant Previous Words',
                fontsize=FONT_SIZE+1, fontweight='bold')

    # Add annotation
    subject_note = ax.annotate('Strong attention\nto subject "cat"', xy=(1, 6), xytext=(3.5, 6.5),
               arrowprops=dict(arrowstyle='->', color=CLARET, lw=1),
               fontsize=FONT_SIZE-1, color=CLARET)
    if ROOM:
        # Both its anchors sit outside the four-word grid, and an in-plot
        # annotation is banned in a room chart anyway.
        subject_note.remove()

    plt.tight_layout()
    plt.savefig(figpath('attention_heatmap'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: attention_heatmap.pdf")


if __name__ == "__main__":
    create_attention_heatmap()
