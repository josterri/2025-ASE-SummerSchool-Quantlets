"""Course palette, matching course.sty and the site. Import from figure scripts."""
import os
import re
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
ROOM_STYLE = Path(__file__).with_name("room.mplstyle")

# ------------------------------------------------------------- room edition
# The room decks are 20pt for a very small screen, and their charts must carry
# text at 16pt ON THE PAGE. A room figure is therefore drawn at the size it is
# displayed at and included with no width option, so the scale is 1.0 by
# construction rather than by arithmetic that can be wrong.
#
# One environment variable rather than an argument, because every generator here
# is run four different ways (directly, by tools/build_figures.py, by
# provenance.py's clean rooms, and by an extracted Quantlet folder) and only one
# of those four passes arguments. Unset means standard, so all four keep drawing
# exactly what they draw today: that is the property tools/add_room_mode.py
# verifies by byte-comparing figures/ before and after it edits anything.
ROOM = os.environ.get("NWP_ROOM") == "1"

# Measured, not chosen: planning-logs/2026-08-17-room-slot.json, from
# tools/probe_room_slot.py compiling a 20pt frame and reading the lengths TeX
# computed. Width is \textwidth; height is the body band less one caption line.
ROOM_SLOT_IN = (5.79, 2.45)

# The slot for a chart that has to carry a CAPTION, which after 2026-08-17 is
# most of them. ROOM_SLOT_IN reserves one caption line and nothing else, and the
# pilot showed that is not what a chart frame costs:
#
#   body band                 202.3pt
#   Quantlet badge             -12.0pt   set small, under the figure
#   two lines of caption       -50.0pt   at the 25pt baseline
#   -------------------------------------
#   left for the figure        140.3pt   which is 1.95 in
#
# Two lines rather than one because the median figure owes two keys once the
# chart title has gone into the frame heading, and a key is not optional: the
# pilot frame lost a legend off the bottom of the page and shipped two
# unlabelled curves. See
# planning-logs/2026-08-17-pilot-the-chart-frame-does-not-fit.md.
#
# NOT applied globally, deliberately. Most room figures already save at about
# 138pt because tight cropping removes whitespace the canvas allowed for, and
# scaling those down as well would cost 20% of every chart to fix the 19 that
# are actually too tall. Use it where a figure measures over 138pt:
# tools/probe_room_fig_height.py names them.
ROOM_SLOT_CAPTIONED_IN = (5.79, 1.95)

ROOM_FLOOR_PT = 16.0


def use_course_style():
    """The course style, with the room size overlay on top when ROOM is set.

    A list, not a second style sheet. room.mplstyle carries sizes and line
    weights only, so the palette is spelled out in exactly one place for
    matplotlib and there is no second copy to drift.
    """
    styles = [str(STYLE)]
    if ROOM:
        styles.append(str(ROOM_STYLE))
    plt.style.use(styles)


def room_size(width, height):
    """The figsize to draw at: unchanged normally, fitted to the slot in room mode.

    Aspect is preserved and the result fits inside the measured slot, so a chart
    that was 8x5 becomes 3.9x2.45 rather than being stretched. A tall chart
    therefore comes out narrow, which is a real cost and shows up as collided
    labels; that is per-chart work, not something a scale factor can fix, and it
    is why every chart item in the room queue carries a look flag.
    """
    if not ROOM:
        return (width, height)
    scale = min(ROOM_SLOT_IN[0] / width, ROOM_SLOT_IN[1] / height)
    return (round(width * scale, 3), round(height * scale, 3))


def figpath(name):
    """Where a chart is written: figures/<name>.pdf, or figures/room/<name>.pdf.

    Takes the bare stem so the two paths cannot disagree about the name, which
    matters because check_quantlets.py derives a Quantlet's name from the figure
    filename in five places.

    NWP_FIG_OUT redirects the whole thing somewhere else, and exists for probes.
    A diagnostic that has to run the generators would otherwise write over
    `figures/`, and those files are hash-gated: `check_figure_toolchain.py`
    requires every one to carry the recorded producer and the pinned
    `/CreationDate`, neither of which a probe run sets. So a probe would leave a
    red gate behind it and the redness would be about the probe. Redirecting is
    cheaper than teaching every probe to restore what it clobbered.
    """
    stem = name
    for prefix in ("figures/room/", "figures/"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    override = os.environ.get("NWP_FIG_OUT")
    if override:
        directory = Path(override) / ("room" if ROOM else "standard")
    else:
        directory = Path("figures/room") if ROOM else Path("figures")
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / f"{stem}.pdf")


def base_font_size(default):
    """The generators' FONT_SIZE constant, room-aware.

    19, and the 3 is measured rather than guessed. The generators spell sizes as
    FONT_SIZE and FONT_SIZE minus an offset, and the largest offset anywhere in
    the six of them is 3, at generate_hour2_charts.py:500 where the multi-head
    tick labels are FONT_SIZE - 3. So 19 is the smallest base that puts every
    derived size at or above the 16pt floor, and the first attempt at 17 left
    those labels drawn at 14.0pt: under the floor, in a room profile whose whole
    purpose is the floor.
    """
    return 19 if ROOM else default


def room_pick(standard, room):
    """The room value while the room pass runs, the shipped value otherwise.

    Every hand repair to a room chart goes through this, and the reason is a
    guarantee rather than a preference. `tools/add_room_mode.py --verify-bytes`
    requires all 45 shipped figures to come out byte-identical after any room
    work, and a call that returns `standard` unchanged outside the room pass
    cannot break that no matter what is passed as `room`. An `if` written inline
    at 200 call sites has the same intent and none of the guarantee, because one
    of them will eventually be written the wrong way round.

    It is also the grep. `room_pick` in a generator is the complete list of
    places the two editions differ, which is the question anyone reading these
    generators in a year will actually have.

        ax.legend(labels=[room_pick("More thinking tokens at answer time",
                                    "more thinking")])
    """
    return room if ROOM else standard


# --------------------------------------------------- room: text that moves out
# A chart's own title is the widest thing in it. Measured over the 45 room
# renderings: 39 of the 42 that missed the floor had ONE text run occupying more
# than 80 per cent of the figure width, and in almost every case that run was the
# title, "RNN Memory Decay: Information Fades with Distance" and its siblings.
# The tight bounding box is the union of the artists, so one long sentence sets
# the figure's width on its own and every tick label is innocent.
#
# On a room slide the frame already carries a title, so an in-figure title is the
# same words twice, and the second copy is what forces the chart to be scaled
# down until nothing in it can be read. So in room mode it is removed.
#
# REMOVED, NOT DISCARDED. Every string taken out is written to
# figures_src/room_carried_text.json against the figure it came from, because
# some of them are load-bearing: "Curve is the Kaplan form; points are
# indicative, not measured" is the sentence that marks a stylised number as
# stylised, which this course requires on the surface a student reads. The room
# deck has to carry it, and a recorded list is what lets that be checked rather
# than remembered.
CARRIED = Path(__file__).with_name("room_carried_text.json")

# Two thresholds, because figure-level text and axes-level text are different
# things and one number for both is wrong in one direction or the other.
#
# A run sitting on the FIGURE is a caption by position: nothing else is drawn
# there. 25 characters is enough to separate it from a stray label.
#
# A run sitting inside an AXES may be a caption or may be the chart's content,
# and stripping content would be the checker writing the teaching. 40 characters
# is where the two separate on this tree, and the case that fixes the threshold
# is real: "Input Words (Attended To)" is 25 characters and is the y-axis label
# of the attention heatmap, while "Curve is the Kaplan form; points are
# indicative, not measured" is 60 and is a provenance caption that belongs on
# the slide. Anything in between is left alone deliberately.
CAPTION_MIN_CHARS = 25
AXES_CAPTION_MIN_CHARS = 40


def _record_carried(figure_stem, strings):
    """Record what this figure gave up, merging into the shared file.

    The write is atomic, and that is not fussiness. During the repair loop two
    generators run at once, in separate processes, both appending here. A plain
    write truncates first, so the other process can read an empty file, hit the
    ValueError branch, and write back a file holding only its own key: every
    other chart's carried strings gone, with no error anywhere. Those strings are
    sentences the room decks are required to print, so the damage would show up
    as a coverage check quietly guarding less.

    Writing to a neighbour and renaming makes a reader see either the old file or
    the new one. It does not serialise read-modify-write, so a concurrent pair
    can still lose one update; the whole-tree rebuild at the end of the loop is
    what settles that, and it rewrites this file from scratch.
    """
    import json
    import os

    if not strings:
        return
    data = {}
    if CARRIED.is_file():
        try:
            data = json.loads(CARRIED.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    data[figure_stem] = strings
    temp = CARRIED.with_name(CARRIED.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    # Retried, because on Windows os.replace RAISES when another process has the
    # target open, where on POSIX it would simply succeed. Two generators ran at
    # once during the repair loop and the build died four times on
    # PermissionError, and the way it died is the part that matters: a generator
    # aborted part way leaves figures/room/ half written, with the charts it did
    # not reach silently holding their PREVIOUS content. That looks exactly like
    # a repair that did not take. It was caught only because a gate reported
    # strings that had already been replaced.
    #
    # This is a cost of the atomic write, which was itself added to stop a
    # concurrent truncate losing every other chart's carried strings. Retrying
    # keeps that and pays for it.
    import time
    for attempt in range(12):
        try:
            os.replace(temp, CARRIED)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.05 * (attempt + 1))


# A log axis is formatted by matplotlib as 10 to the n, in mathtext, and a
# mathtext superscript is drawn at 0.663 of its base size. At a base of 19 that
# is 12.6pt, under the floor, and no font setting reaches it: the ratio is
# typography, not a parameter. Measured across the room set, that one ratio was
# the entire reason six charts missed the floor.
#
# So the exponents go. "1B" is not a workaround for "10 to the 9", it is the
# better label on a small screen, and it removes mathtext from the axis rather
# than arguing about how small a superscript is allowed to be.
def _plain_number(value, _pos=None, suffixes=True):
    if value == 0:
        return "0"
    size = abs(value)
    if not suffixes:
        # An axis whose LABEL already carries the magnitude, "Parameters
        # (millions)", must not have a second magnitude glued to its ticks. The
        # 1000 tick on such an axis became "1K", and the room chart then claimed
        # a thousand parameters where the standard chart said a thousand million.
        # It was legible, well spaced, green on every gate, and false. A
        # legibility pass is allowed to make a number bigger; it is never allowed
        # to make it a different number.
        return "{:g}".format(value)
    # The ladder runs to yotta because a compute axis does. It stopped at tera,
    # and above that the loop divided by 1e12 and appended "T", so 1e18 printed
    # as "1e+06T": scientific notation with a unit suffix glued on, on the one
    # axis in the tree that reaches 1e25. Every rung above tera was invented the
    # day compute_requirements was measured, which is the day anything first
    # asked for one.
    for divisor, suffix in ((1e24, "Y"), (1e21, "Z"), (1e18, "E"), (1e15, "P"),
                            (1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if size >= divisor:
            return "{:g}{}".format(value / divisor, suffix)
    if size >= 1:
        return "{:g}".format(value)
    return ("{:f}".format(value)).rstrip("0").rstrip(".") or "0"


AXIS_LABEL_WRAP = 30

# An axis label that already names a magnitude. When one of these is present the
# tick formatter must not add a second one.
MAGNITUDE_IN_LABEL = re.compile(r"\b(thousand|million|billion|trillion)s?\b", re.I)

# Only three call sites carry mathtext in this tree, all of them the Q, K and V
# weight matrices in qkv_flow. Converted rather than exempted, on the
# instructor's decision of 2026-08-17: a subscript cannot be drawn at the floor,
# and "W_Q" set at the full 19pt reads better across a room than a 12.6pt
# subscript does.
#
# Deliberately conservative. It converts one shape, a name with a single
# subscript or superscript, and leaves anything more complicated alone. A
# general mathtext-to-text rewriter would quietly mangle a real formula, and
# there is no need for one: an unconverted mathtext run stays under the floor
# and probe_figure_scale.py --check fails on it, so the floor check IS the
# detector for whatever this does not handle.
SIMPLE_MATH = re.compile(r"^\$([A-Za-z]+)([_^])\{?([A-Za-z0-9]+)\}?\$$")


def _room_demath(text):
    found = SIMPLE_MATH.match(text.strip())
    return "".join(found.groups()) if found else text


def _room_axis_text(ax):
    """Plain log labels, and long axis labels wrapped rather than shortened.

    Wrapped, not cut. A 56 character axis label is 500pt of ink on a 417pt
    slide and sets the whole figure's width on its own, but every word in it is
    the author's, so the line breaks and nothing is lost. Mathtext labels are
    left alone: a break inside $...$ is a syntax error, not a line break.
    """
    import textwrap

    from matplotlib.ticker import FuncFormatter, NullFormatter

    import functools

    for axis, scale, label in ((ax.xaxis, ax.get_xscale(), ax.get_xlabel()),
                               (ax.yaxis, ax.get_yscale(), ax.get_ylabel())):
        if scale != "log":
            continue
        # Suffixes only when the axis label does not already state a magnitude.
        # See _plain_number: "Parameters (millions)" plus a "1K" tick is a chart
        # that lies about its own data.
        suffixes = not MAGNITUDE_IN_LABEL.search(label or "")
        fmt = FuncFormatter(functools.partial(_plain_number, suffixes=suffixes))
        axis.set_major_formatter(fmt)

        # Minor labels are silenced only when the majors actually label the
        # range. A log axis spanning 2.8 to 4.9 has no decade boundary inside it,
        # so matplotlib labels the MINOR ticks and nothing else; silencing them
        # left an axis carrying no numbers at all, which no geometric check can
        # notice because an absence collides with nothing.
        low, high = sorted(axis.get_view_interval())
        majors = [t for t in axis.get_majorticklocs() if low <= t <= high]
        axis.set_minor_formatter(NullFormatter() if len(majors) >= 2 else fmt)

    for getter, setter in ((ax.get_xlabel, ax.set_xlabel),
                           (ax.get_ylabel, ax.set_ylabel)):
        label = getter()
        if len(label) > AXIS_LABEL_WRAP and "\n" not in label and "$" not in label:
            setter("\n".join(textwrap.wrap(label, AXIS_LABEL_WRAP)))


def _strip_for_room(fig, figure_stem):
    """Take the title and any figure-level caption off, and record them."""
    carried = []
    for ax in fig.axes:
        _room_axis_text(ax)
    for ax in fig.axes:
        title = ax.get_title()
        if title.strip():
            carried.append(title.strip())
            ax.set_title("")
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text().strip():
        carried.append(sup.get_text().strip())
        sup.set_text("")
    for text in list(fig.texts):
        body = text.get_text().strip()
        if len(body) >= CAPTION_MIN_CHARS:
            carried.append(body)
            text.set_text("")
    # Axes-level text at the higher threshold. These are the provenance captions
    # this course requires on the surface a student reads, "computed on the toy
    # corpus: 342 tokens, 50 sentences" and "stylised conditionals, shown for
    # shape not for value", and after the titles came off they were the widest
    # run in eleven of the nineteen charts still missing the floor. They move to
    # the slide rather than disappearing, and check_room_carried_text.py is what
    # makes that a fact rather than an intention.
    for ax in fig.axes:
        for text in list(ax.texts):
            body = text.get_text().strip()
            if len(body) >= AXES_CAPTION_MIN_CHARS:
                carried.append(body)
                # An annotation is a text AND an arrow. Blanking the text left
                # the arrow behind, pointing from nothing at a bubble, on
                # cost_quality_latency. Removing the artist takes both, which is
                # what "this caption moves to the slide" has to mean: an arrow
                # with no label is not a smaller caption, it is a defect that
                # looks deliberate.
                if getattr(text, "arrow_patch", None) is not None:
                    text.remove()
                else:
                    text.set_text("")
            else:
                plain = _room_demath(body)
                if plain != body:
                    text.set_text(plain)

    # Fit the decorations INSIDE the figure before saving.
    #
    # Without this the room charts came out wider than the slide even though
    # every glyph in them was legal: at 19pt in a 4.9 inch figure the tick
    # labels and the axis label do not fit the default margins, so they are
    # drawn past the figure edge, and bbox_inches='tight' then EXPANDS the saved
    # area to include them. token_distribution reached 8.02 inches from a 4.9
    # inch figsize that way, was scaled to 0.72 by beamer to fit the slide, and
    # arrived at 12.27pt: under the floor, with nothing in it under the floor.
    # A width failure and a size failure look identical in the output and have
    # opposite remedies, which is why the report carries both columns.
    try:
        fig.tight_layout()
    except Exception:
        # Some figures here place axes by hand, where tight_layout has nothing
        # to solve and says so. Not fatal, and not silent either: such a figure
        # stays too wide and the floor check names it.
        pass

    # There WAS a `_fit_to_slot` here, shrinking the figsize until the tight box
    # fit the slide. It is gone, and the reason is worth more than the code was.
    #
    # Shrinking the canvas does not shrink the type, which is fixed in points. So
    # the axes give up the whole difference: iterate that four times and the axes
    # are a smear a centimetre wide with 18pt labels lying across each other.
    # inference_compute_scaling and token_distribution came out unreadable, and
    # the floor check called them legible, because every glyph in the wreckage
    # was still 17 or 18pt. Its own docstring said the relationship was not
    # proportional and then treated that as a reason to iterate rather than as a
    # reason to stop.
    #
    # Beamer's scaling is the better tool and was available the whole time: it
    # scales the type WITH the figure, so the layout survives and only the size
    # falls. A chart is therefore saved at whatever width its content needs,
    # `\roomfig` shrinks it to the text width if it must, and the floor is
    # applied to what comes out. When that lands under 16pt the remedy is less
    # text, which is an editorial decision and not one a layout pass may take.
    _record_carried(figure_stem, carried)
    return carried


def _install_room_hook():
    """Wrap Figure.savefig so the room pass strips before it writes.

    A wrapper rather than an edit at 45 call sites, because the transform is the
    same everywhere and a per-call edit would be 45 chances to miss one. It is
    installed only when ROOM is set, so a standard run reaches matplotlib's own
    savefig untouched and the committed figures cannot move: that property is
    what tools/add_room_mode.py --verify-bytes measures.
    """
    from matplotlib.figure import Figure

    if getattr(Figure.savefig, "_room_wrapped", False):
        return
    original = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        try:
            stem = Path(str(fname)).stem
        except Exception:
            stem = "unknown"
        _strip_for_room(self, stem)
        return original(self, fname, *args, **kwargs)

    savefig._room_wrapped = True
    Figure.savefig = savefig


if ROOM:
    _install_room_hook()


def room_pt(default):
    """A literal point size, room-aware, for the handful that are not FONT_SIZE.

    Doubled and then floored, so hierarchy survives: a 6pt annotation and a 9pt
    label do not both flatten to 16.
    """
    if not ROOM:
        return default
    return max(ROOM_FLOOR_PT, round(default * 2))


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
