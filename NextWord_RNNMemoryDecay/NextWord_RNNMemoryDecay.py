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


def create_rnn_memory_decay():
    """Chart 5: RNN memory decay visualization"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Word positions
    positions = np.arange(1, 21)
    
    # Memory strength (exponential decay)
    vanilla_rnn = 100 * np.exp(-0.3 * positions)
    lstm = 100 * np.exp(-0.1 * positions)
    gru = 100 * np.exp(-0.15 * positions)
    
    # Plot lines
    ax.plot(positions, vanilla_rnn, 'o-', label='Vanilla RNN', linewidth=2, markersize=4)
    ax.plot(positions, lstm, 's-', label='LSTM', linewidth=2, markersize=4)
    ax.plot(positions, gru, '^-', label='GRU', linewidth=2, markersize=4)
    
    # Add critical threshold line
    ax.axhline(y=10, color=CLARET, linestyle='--', alpha=0.5, linewidth=1)
    ax.text(19, 12, 'Effective threshold', fontsize=FONT_SIZE-1, color=CLARET, ha='right')
    
    # Annotations
    ax.annotate('Information lost!', xy=(10, vanilla_rnn[9]), xytext=(12, 20),
               arrowprops=dict(arrowstyle='->', color=CLARET, lw=1),
               fontsize=FONT_SIZE-1, color=CLARET)
    
    ax.set_xlabel('Word Position', fontsize=FONT_SIZE)
    ax.set_ylabel('Memory Strength (%)', fontsize=FONT_SIZE)
    ax.set_title('RNN Memory Decay: Information Fades with Distance', 
                fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(loc='upper right', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('figures/rnn_memory_decay.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: rnn_memory_decay.pdf")


if __name__ == "__main__":
    create_rnn_memory_decay()
