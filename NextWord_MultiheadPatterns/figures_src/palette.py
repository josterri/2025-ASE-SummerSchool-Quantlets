"""Course palette, matching course.sty and the site. Import from figure scripts."""
from pathlib import Path

import matplotlib.pyplot as plt

CLARET = "#990F3D"
TEAL = "#0D7680"
GOLD = "#9C6A10"
INK = "#262A33"
SOFT = "#66605A"
PAPER = "#F7E6D3"
RULE = "#D9C5AC"

CYCLE = [CLARET, TEAL, GOLD, INK, SOFT]

STYLE = Path(__file__).with_name("course.mplstyle")


def use_course_style():
    plt.style.use(str(STYLE))


# ---------------------------------------------------------------- colormaps
# Figure scripts previously reached for matplotlib built-ins (Blues, RdYlBu_r,
# viridis), which silently defeat the course palette even when the style sheet
# is loaded. These two builders keep gradients on-palette.

def sequential_cmap(base=CLARET, name="course_seq"):
    """Light paper to a saturated palette colour. Use for magnitude heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, ["#FFFFFF", PAPER, base])


def diverging_cmap(low=TEAL, high=CLARET, name="course_div"):
    """Teal through paper to claret. Use when a midpoint is meaningful."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, [low, PAPER, high])


# Semantic aliases so figure scripts never spell a raw colour name. Anything
# that used to be 'red'/'blue'/'green' maps here instead.
GOOD = TEAL
BAD = CLARET
WARN = GOLD
NEUTRAL = SOFT
EMPHASIS = CLARET
MUTED = RULE
