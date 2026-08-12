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
Generate charts for Hour 1: The Task (N-grams and Early Methods)
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


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


def create_word_vector_visualization():
    """Chart 4: Word vector space visualization"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Deterministic ring layout. The previous version scattered each cluster
    # with np.random.normal(centre, 0.3) and then annotated every label AT its
    # point with no offset, so whenever two points landed close the words
    # overlapped: "sofa" on "cabinet", "run" on "jump", "sleep" on "eat". A ring
    # guarantees even spacing, and offsetting each label radially outward from
    # the cluster centre keeps the text clear of the markers and of each other.
    animals = ['cat', 'kitten', 'dog', 'puppy', 'mouse', 'hamster']
    furniture = ['table', 'chair', 'desk', 'sofa', 'bed', 'cabinet']
    verbs = ['run', 'walk', 'jump', 'sit', 'sleep', 'eat']

    animal_center = [2.0, 1.8]
    furniture_center = [-2.1, -1.0]
    verb_center = [0.2, -2.2]

    def _ring(center, count, radius=0.62):
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + np.pi / count
        points = np.column_stack([center[0] + radius * np.cos(angles),
                                  center[1] + radius * np.sin(angles)])
        return points, angles

    animal_points, animal_angles = _ring(animal_center, len(animals))
    furniture_points, furniture_angles = _ring(furniture_center, len(furniture))
    verb_points, verb_angles = _ring(verb_center, len(verbs))

    ax.scatter(animal_points[:, 0], animal_points[:, 1], c=TEAL, s=90, alpha=0.7, label='Animals')
    ax.scatter(furniture_points[:, 0], furniture_points[:, 1], c=GOLD, s=90, alpha=0.7, label='Furniture')
    ax.scatter(verb_points[:, 0], verb_points[:, 1], c=CLARET, s=90, alpha=0.7, label='Verbs')

    def _label(points, angles, words):
        for (x, y), angle, word in zip(points, angles, words):
            ax.annotate(word, (x, y),
                        xytext=(x + 0.26 * np.cos(angle), y + 0.24 * np.sin(angle)),
                        fontsize=FONT_SIZE - 1, color=INK,
                        ha='left' if np.cos(angle) >= 0 else 'right',
                        va='bottom' if np.sin(angle) >= 0 else 'top')

    _label(animal_points, animal_angles, animals)
    _label(furniture_points, furniture_angles, furniture)
    _label(verb_points, verb_angles, verbs)

    # cat and kitten are adjacent on the ring, so this reads as intended
    ax.plot([animal_points[0, 0], animal_points[1, 0]],
            [animal_points[0, 1], animal_points[1, 1]],
            linestyle='--', color=TEAL, alpha=0.6, linewidth=2)
    ax.annotate('Similar words\nare close together',
                xy=((animal_points[0, 0] + animal_points[1, 0]) / 2,
                    (animal_points[0, 1] + animal_points[1, 1]) / 2),
                xytext=(2.6, 3.2), fontsize=FONT_SIZE - 1, color=SOFT,
                arrowprops=dict(arrowstyle='->', color=TEAL, alpha=0.6))

    ax.set_xlabel('Dimension 1', fontsize=FONT_SIZE)
    ax.set_ylabel('Dimension 2', fontsize=FONT_SIZE)
    ax.set_title('Word Embeddings: Similar Words Cluster Together',
                fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(loc='lower right', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3.6, 4.3)
    ax.set_ylim(-3.4, 3.9)
    
    plt.tight_layout()
    plt.savefig('figures/word_vector_space.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: word_vector_space.pdf")


import json as _json
from pathlib import Path as _Path


if __name__ == "__main__":
    create_word_vector_visualization()
