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


def create_word_vector_visualization():
    """Chart 4: Word vector space visualization"""
    # ROOM_SLOT_CAPTIONED_IN, not ROOM_SLOT_IN: two caption lines are owed, the
    # cluster names and the closeness claim, and 141.6pt did not fit the 128.1pt
    # that leaves.
    fig, ax = plt.subplots(
        figsize=room_size(*room_pick((8, 5), ROOM_SLOT_CAPTIONED_IN)))

    # Deterministic ring layout. The previous version scattered each cluster
    # with np.random.normal(centre, 0.3) and then annotated every label AT its
    # point with no offset, so whenever two points landed close the words
    # overlapped: "sofa" on "cabinet", "run" on "jump", "sleep" on "eat". A ring
    # guarantees even spacing, and offsetting each label radially outward from
    # the cluster centre keeps the text clear of the markers and of each other.
    # Room edition: two words per cluster instead of six. Eighteen words is 67
    # pairwise text collisions at 18pt in this canvas, and the claim the chart
    # makes, that words of a kind land together, needs two points per cluster to
    # be visible, not six.
    animals = room_pick(['cat', 'kitten', 'dog', 'puppy', 'mouse', 'hamster'],
                        ['cat', 'kitten'])
    furniture = room_pick(['table', 'chair', 'desk', 'sofa', 'bed', 'cabinet'],
                          ['table', 'chair'])
    verbs = room_pick(['run', 'walk', 'jump', 'sit', 'sleep', 'eat'],
                      ['run', 'walk'])

    # Room edition: three corners rather than a diagonal. With six words the
    # original centres put "run" and "kitten" on almost the same line, 15pt
    # apart, so they read as one phrase; nothing collided and it still looked
    # wrong. Two clusters above and one below separates every pair by a cluster's
    # width.
    #
    # Round 1 moved them to 11pt apart, which is two word spaces at 18pt: the
    # intent was recorded and not reached, and the rendering still read "run
    # kitten". The lower word of a cluster is labelled to the LEFT of its point,
    # so the animals reach back towards the verbs by the width of "kitten"
    # before anything visible moves. Pushing the animals out and the verbs back
    # buys 36pt of clear air between the two runs, which is a gap and not a
    # space. "cat" then ends at 4.25 against a limit of 4.4, and "chair" at
    # -3.76 against -4.2, so both stay inside the axes.
    animal_center = room_pick([2.0, 1.8], [3.0, 1.4])
    furniture_center = room_pick([-2.1, -1.0], [-2.4, 1.4])
    verb_center = room_pick([0.2, -2.2], [-0.35, -1.9])

    # Room edition: a wider ring. Two words per cluster sit at opposite ends of
    # a diameter, and 0.62 puts them 22pt apart in a 110pt axes, which is one
    # line of 18pt type between two labels.
    def _ring(center, count, radius=room_pick(0.62, 0.85)):
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
    if not ROOM:
        # 32 characters, under the room pass's 40-character axes threshold, so it
        # would not be carried automatically and would simply print across the
        # points. Dropped here, and the sentence goes on the room slide.
        ax.annotate('Similar words\nare close together',
                    xy=((animal_points[0, 0] + animal_points[1, 0]) / 2,
                        (animal_points[0, 1] + animal_points[1, 1]) / 2),
                    xytext=(2.6, 3.2), fontsize=FONT_SIZE - 1, color=SOFT,
                    arrowprops=dict(arrowstyle='->', color=TEAL, alpha=0.6))

    ax.set_xlabel('Dimension 1', fontsize=FONT_SIZE)
    ax.set_ylabel('Dimension 2', fontsize=FONT_SIZE)
    ax.set_title('Word Embeddings: Similar Words Cluster Together',
                fontsize=FONT_SIZE+1, fontweight='bold')
    if not ROOM:
        # Three entries at 18pt stack 60pt tall and land across the data: in the
        # room rendering "Furniture" printed over both "kitten" and "table". The
        # words on the chart are the key here (cat and kitten, table and chair,
        # run and walk), so the legend goes and the naming goes on the slide.
        ax.legend(loc='lower right', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3)
    if ROOM:
        # The axes are two arbitrary embedding dimensions, so the numbers on them
        # carry nothing a reader can use, and at 18pt they collide with the words
        # that do. The labels stay; the scale goes.
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlim(*room_pick((-3.6, 4.3), (-4.2, 4.4)))
    # Room edition, lower floor: a label is a fixed number of POINTS tall, so
    # shortening the canvas to the captioned slot made it a taller share of the
    # y range. "walk" is labelled below its own point and landed across the axis
    # spine at -4.7. Found by rendering the figure and looking at it, which is
    # the only thing that can see a word sitting on a line.
    ax.set_ylim(*room_pick((-3.4, 3.9), (-5.6, 4.2)))


    if not ROOM:
        plt.tight_layout()  # see LAYOUT ORDER at the top of this file
    plt.savefig(figpath('word_vector_space'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: word_vector_space.pdf")


import json as _json
from pathlib import Path as _Path


if __name__ == "__main__":
    create_word_vector_visualization()
