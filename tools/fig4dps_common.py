"""Shared style and layout for the 4DPS paper figures.

Every
colour, radius, border weight and font size a figure uses is defined here and
nowhere else.

Figures are drawn at 1:1 print size. The ICLR 2027 text block is a single
5.5 in column, so a figure included at width=\\linewidth prints at the size it
is drawn and the point sizes below are the printed sizes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np

# ---------------------------------------------------------------------------
# Page geometry (iclr2027_conference.sty: \textwidth = 5.5in, one column)
# ---------------------------------------------------------------------------

TEXT_WIDTH_INCHES = 5.5
FIG_WIDTH_INCHES = TEXT_WIDTH_INCHES

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

GRN = "#1E7A46"   # ours, rendered, forward pass
RED = "#B31B1B"   # existing methods, gradient (backward) pass
VIO = "#6A3FB5"   # the loss and its bracket
BLU = "#1565C0"   # inputs; transmitters
AMB = "#B45F17"   # receivers
GRY = "#6A6A6A"   # real / measured, secondary text

INK = "#1B1B1B"           # body text in figures
MUTED = GRY               # captions inside a figure, axis labels

EXISTING_FILL = "#FBE7E6"
OURS_FILL = "#E5F4E5"
HIGHLIGHT_FILL = "#F6D9C0"

# Role aliases: figures use these, not the hues above
OURS, EXISTING, LOSS = GRN, RED, VIO
REAL, RENDERED = GRY, GRN
FORWARD, GRADIENT = GRN, RED
INPUT, TX, RX = BLU, BLU, AMB

BAND_COLOR = "#F5F5F5"    # outer band behind a row of tiles
TILE_COLOR = "#EBEBEB"    # a tile on a band
BLOCK_COLOR = "#E0E0E0"   # a block inside a tile
IMAGE_BAND_COLOR = "#ECEBEB"
NOT_PRODUCED_COLOR = "#DADADA"

BACKGROUND_COLOR = IMAGE_BAND_COLOR
TEST_COL_COLOR = "#F5C6C0"

# Heatmaps: linear magnitude only, never dB (the paper states log compression
# inflates agreement). One colormap for every fidelity panel.
HEATMAP_CMAP = "inferno"
# Signed radial velocity of points (toward / away), a continuous diverging map; its blue and red
# ends are not the TX / existing roles. Its neutral is the body grey (not near-white, as coolwarm's),
# so a moving body reads as a body and fades toward white the same way a grey one does.
from matplotlib.colors import LinearSegmentedColormap as _LSC
VELOCITY_CMAP = _LSC.from_list("velocity", ["#2F4FB0", "#8A8A8A", "#B8322A"])
MATERIAL_CMAP = "plasma"

# ---------------------------------------------------------------------------
# Shapes. Radii are physical (inches), circular in both axes.
# ---------------------------------------------------------------------------

BAND_RADIUS_INCHES = 0.06
TILE_RADIUS_FRACTION = 0.028
TILE_RADIUS_MIN_INCHES = 0.027
TILE_RADIUS_MAX_INCHES = BAND_RADIUS_INCHES
CORNER_RADIUS_INCHES = BAND_RADIUS_INCHES

TILE_BORDER_PT = 0.0
ACCENT_BORDER_PT = 1.1
HEADER_BAND_FRACTION = 0.066
HEADER_BAND_MIN_INCHES = 0.15    # never shorter than one header line plus padding

TILE_GAP_INCHES = 0.09
BAND_PAD_INCHES = 0.05

ARROW_LW_PT = 0.7
ARROW_DASH = (0, (4, 3))
ARROW_PAIR_OFFSET_INCHES = 0.03  # spacing of a forward / gradient pair
BRACKET_LW_PT = 1.0


def tile_radius(width_in):
    """Corner radius for a tile of the given width: 2.8% of the width, clamped."""
    return float(np.clip(TILE_RADIUS_FRACTION * width_in,
                         TILE_RADIUS_MIN_INCHES, TILE_RADIUS_MAX_INCHES))


def header_band_height(width_in):
    """Height of a tile's header band for a tile of the given width."""
    return max(HEADER_BAND_FRACTION * width_in, HEADER_BAND_MIN_INCHES)

# ---------------------------------------------------------------------------
# Typography. Times, to match the ICLR body (Times 10pt). Printed sizes: the
# ---------------------------------------------------------------------------

PAPER_BODY_PT = 10.0
FIGURE_TITLE_PT = 8.0     # tile titles, header bands
FIGURE_HEADER_PT = 7.5    # column headers, row labels
FIGURE_BASE_PT = 7.0      # labels, ticks, legends
FIGURE_SMALL_PT = 6.5     # per-cell overlays (correlation values)
FIGURE_FLOOR_PT = 6.0     # nothing in a figure is smaller


def apply_paper_font():
    """Configure matplotlib to match the ICLR paper typography.

    Times-compatible serif with Computer Modern math, TrueType embedding, no
    usetex. Safe to call repeatedly; call once before creating a Figure."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Nimbus Roman", "Times New Roman",
                                "Liberation Serif", "DejaVu Serif"],
        "font.size":          FIGURE_BASE_PT,
        "axes.labelsize":     FIGURE_BASE_PT,
        "axes.titlesize":     FIGURE_HEADER_PT,
        "xtick.labelsize":    FIGURE_FLOOR_PT,
        "ytick.labelsize":    FIGURE_FLOOR_PT,
        "legend.fontsize":    FIGURE_BASE_PT,
        "figure.titlesize":   FIGURE_TITLE_PT,
        "axes.linewidth":     0.5,
        "lines.linewidth":    0.8,
        "mathtext.fontset":   "cm",
        "mathtext.rm":        "serif",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "savefig.dpi":        300,
    })


def add_rounded_bg(fig, radius_inches=BAND_RADIUS_INCHES, color=BACKGROUND_COLOR):
    """Add a rounded-rectangle background with physically circular corners."""
    fig_w, fig_h = fig.get_size_inches()
    rx = radius_inches / fig_w
    ry = radius_inches / fig_h

    k = 0.5523  # Bezier approximation of a quarter-circle
    verts = [
        (0, ry),
        (0, ry * (1 - k)), (rx * (1 - k), 0), (rx, 0),
        (1 - rx, 0),
        (1 - rx * (1 - k), 0), (1, ry * (1 - k)), (1, ry),
        (1, 1 - ry),
        (1, 1 - ry * (1 - k)), (1 - rx * (1 - k), 1), (1 - rx, 1),
        (rx, 1),
        (rx * (1 - k), 1), (0, 1 - ry * (1 - k)), (0, 1 - ry),
        (0, ry),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    path = Path(verts, codes)
    patch = mpatches.PathPatch(
        path, facecolor=color, edgecolor="none",
        transform=fig.transFigure, zorder=-1,
    )
    fig.patches.append(patch)
    fig.patch.set_alpha(0.0)


class GridLayout:
    """Compute a grid layout in inches, converting to figure fractions.

    This ensures images maintain their aspect ratio regardless of the
    overall figure shape (avoiding stretching from fraction-based math).
    """

    def __init__(self, n_rows, n_cols, cell_w_in, cell_h_in, *,
                 margin_in=0.06, col_gap_in=0.03, row_gap_in=0.03,
                 header_in=0.16, label_w_in=0.34):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.cell_w_in = cell_w_in
        self.cell_h_in = cell_h_in
        self.margin_in = margin_in
        self.col_gap_in = col_gap_in
        self.row_gap_in = row_gap_in
        self.header_in = header_in
        self.label_w_in = label_w_in

        self.fig_w = FIG_WIDTH_INCHES
        self.fig_h = (
            2 * margin_in + header_in
            + n_rows * cell_h_in
            + (n_rows - 1) * row_gap_in
        )

    def cell_pos(self, row, col):
        """Return (left, bottom, width, height) in figure-fraction coords."""
        fw, fh = self.fig_w, self.fig_h
        left = (self.margin_in + self.label_w_in
                + col * (self.cell_w_in + self.col_gap_in)) / fw
        top = 1.0 - (self.margin_in + self.header_in
                      + row * (self.cell_h_in + self.row_gap_in)) / fh
        bottom = top - self.cell_h_in / fh
        w = self.cell_w_in / fw
        h = self.cell_h_in / fh
        return left, bottom, w, h

    def header_y(self):
        """Y-position (figure frac) for column header text."""
        return 1.0 - (self.margin_in + self.header_in * 0.15) / self.fig_h

    def row_label_x(self):
        """X-position (figure frac) for row labels."""
        return (self.margin_in + self.label_w_in * 0.5) / self.fig_w

    def row_label_y(self, row):
        """Y-center (figure frac) for a given row label."""
        _, bottom, _, h = self.cell_pos(row, 0)
        return bottom + h / 2

    @classmethod
    def from_image_aspect(cls, n_rows, n_cols, img_aspect=1.0, **kwargs):
        """Create layout computing cell width from available space."""
        return cls.from_fig_width(n_rows, n_cols, FIG_WIDTH_INCHES,
                                  img_aspect=img_aspect, **kwargs)

    @classmethod
    def from_fig_width(cls, n_rows, n_cols, fig_width_in, img_aspect=1.0,
                        **kwargs):
        """Like ``from_image_aspect`` but with a custom total figure width."""
        margin_in = kwargs.pop("margin_in", 0.06)
        col_gap_in = kwargs.pop("col_gap_in", 0.03)
        label_w_in = kwargs.pop("label_w_in", 0.34)
        usable_w = (fig_width_in - label_w_in - 2 * margin_in
                     - (n_cols - 1) * col_gap_in)
        cell_w = usable_w / n_cols
        cell_h = cell_w * img_aspect
        obj = cls(n_rows, n_cols, cell_w, cell_h,
                   margin_in=margin_in, col_gap_in=col_gap_in,
                   label_w_in=label_w_in, **kwargs)
        obj.fig_w = fig_width_in
        obj.fig_h = (
            2 * obj.margin_in + obj.header_in
            + n_rows * cell_h
            + (n_rows - 1) * obj.row_gap_in
        )
        return obj


def _rounded_rect_path(x0, y0, w, h, fig_w, fig_h, radius_in,
                       corners=(True, True, True, True)):
    """Build a matplotlib ``Path`` for a rounded rectangle with physically
    circular corners (i.e., radius in inches is the same in x and y, even
    though the figure width and height differ in fraction-of-figure units).

    Coordinates are in figure-fraction (matches ``transform=fig.transFigure``).
    ``corners`` = (TL, TR, BR, BL); False leaves that corner square, which is
    how a header band is clipped to its tile's top corners.
    """
    rx = radius_in / fig_w   # x-radius in figure fraction (corner is circular
    ry = radius_in / fig_h   # in inches, but rx/ry differ in fraction units)
    rx = min(rx, w * 0.5)
    ry = min(ry, h * 0.5)
    k = 0.5523               # cubic-Bezier quarter-circle coefficient
    x1, y1 = x0 + w, y0 + h
    tl, tr, br, bl = corners

    verts, codes = [], []
    if bl:
        verts += [(x0, y0 + ry), (x0, y0 + ry * (1 - k)),
                  (x0 + rx * (1 - k), y0), (x0 + rx, y0)]
        codes += [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        verts.append((x0, y0)); codes.append(Path.MOVETO)
    if br:
        verts += [(x1 - rx, y0), (x1 - rx * (1 - k), y0),
                  (x1, y0 + ry * (1 - k)), (x1, y0 + ry)]
        codes += [Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        verts.append((x1, y0)); codes.append(Path.LINETO)
    if tr:
        verts += [(x1, y1 - ry), (x1, y1 - ry * (1 - k)),
                  (x1 - rx * (1 - k), y1), (x1 - rx, y1)]
        codes += [Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        verts.append((x1, y1)); codes.append(Path.LINETO)
    if tl:
        verts += [(x0 + rx, y1), (x0 + rx * (1 - k), y1),
                  (x0, y1 - ry * (1 - k)), (x0, y1 - ry)]
        codes += [Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        verts.append((x0, y1)); codes.append(Path.LINETO)
    verts.append(verts[0]); codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


def add_column_highlight(fig, layout, col, color=TEST_COL_COLOR,
                          radius_inches=TILE_RADIUS_MIN_INCHES,
                          inset_in=0.02, include_header=True):
    """Highlight one full column (across all rows) with a rounded tile.

    Drawn between the background band (``add_rounded_bg``, zorder=-1) and the
    row cells (default zorder 0). ``include_header=True`` extends the tile
    over the column-header strip.
    """
    left, top_bottom, w, _ = layout.cell_pos(0, col)
    _, bot_bottom, _, _ = layout.cell_pos(layout.n_rows - 1, col)
    top_of_first = top_bottom + layout.cell_h_in / layout.fig_h
    full_h = top_of_first - bot_bottom
    inset_x = inset_in / layout.fig_w
    inset_y = inset_in / layout.fig_h
    x0 = left - inset_x
    y0 = bot_bottom - inset_y
    rect_w = w + 2 * inset_x
    rect_h = full_h + 2 * inset_y
    if include_header:
        header_top_y = 1.0 - layout.margin_in / layout.fig_h
        rect_h = header_top_y - y0

    path = _rounded_rect_path(x0, y0, rect_w, rect_h,
                                layout.fig_w, layout.fig_h, radius_inches)
    patch = mpatches.PathPatch(
        path, transform=fig.transFigure, facecolor=color, edgecolor="none",
        zorder=-0.5, linewidth=0,
    )
    fig.patches.append(patch)

# ---------------------------------------------------------------------------
# figure draws tiles, headers, arrows and brackets the same way)
# ---------------------------------------------------------------------------


def _frac(fig, x_in, y_in):
    fw, fh = fig.get_size_inches()
    return x_in / fw, y_in / fh


def draw_tile(fig, x_in, y_in, w_in, h_in, color=TILE_COLOR, *,
              radius_in=None, edgecolor=None, border_pt=TILE_BORDER_PT,
              corners=(True, True, True, True), zorder=0):
    """A rounded tile at inch coordinates. Radius defaults to tile_radius(w)."""
    fw, fh = fig.get_size_inches()
    r = tile_radius(w_in) if radius_in is None else radius_in
    path = _rounded_rect_path(x_in / fw, y_in / fh, w_in / fw, h_in / fh,
                              fw, fh, r, corners=corners)
    patch = mpatches.PathPatch(
        path, transform=fig.transFigure, facecolor=color,
        edgecolor=edgecolor if (edgecolor and border_pt) else "none",
        linewidth=border_pt, zorder=zorder)
    fig.patches.append(patch)
    return patch


def draw_header_tile(fig, x_in, y_in, w_in, h_in, title, *,
                     header_color=INK, fill=TILE_COLOR, text_color="white",
                     header_h_in=None, zorder=0):
    """A tile whose header is its own top section, flush and clipped by the
    tile's corner radius. Returns the content box
    (x, y, w, h) below the header, in inches."""
    hh = header_band_height(w_in) if header_h_in is None else header_h_in
    r = tile_radius(w_in)
    draw_tile(fig, x_in, y_in, w_in, h_in, fill, radius_in=r, zorder=zorder)
    draw_tile(fig, x_in, y_in + h_in - hh, w_in, hh, header_color, radius_in=r,
              corners=(True, True, False, False), zorder=zorder + 0.1)
    fx, fy = _frac(fig, x_in + w_in / 2, y_in + h_in - hh / 2)
    fig.text(fx, fy, title, ha="center", va="center", color=text_color,
             fontsize=FIGURE_TITLE_PT, fontweight="bold", zorder=zorder + 0.2)
    return x_in, y_in, w_in, h_in - hh


def draw_method_tile(fig, x_in, y_in, w_in, h_in, title, ours, *, zorder=0):
    """The red / green method tile of the teaser (variant b): coloured
    title above, a coloured border, content inside. Returns the content box."""
    accent = OURS if ours else EXISTING
    draw_tile(fig, x_in, y_in, w_in, h_in, "white", edgecolor=accent,
              border_pt=ACCENT_BORDER_PT, zorder=zorder)
    fx, fy = _frac(fig, x_in + w_in / 2, y_in + h_in + 0.02)
    fig.text(fx, fy, title, ha="center", va="bottom", color=accent,
             fontsize=FIGURE_TITLE_PT, fontweight="bold", zorder=zorder + 0.2)
    return x_in, y_in, w_in, h_in


def draw_arrow(fig, p0_in, p1_in, color=GRY, *, lw=ARROW_LW_PT, dashed=False,
               zorder=5):
    """A single arrow between two inch-coordinate points."""
    a = mpatches.FancyArrowPatch(
        _frac(fig, *p0_in), _frac(fig, *p1_in), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=6, linewidth=lw, color=color,
        linestyle=ARROW_DASH if dashed else "-", shrinkA=0, shrinkB=0,
        zorder=zorder)
    fig.patches.append(a)
    return a


def draw_forward_gradient(fig, p0_in, p1_in, *, gradient=True, zorder=5):
    """The paired arrows: dashed green forward arrow p0 -> p1 and, offset beside
    it, a dashed red gradient arrow p1 -> p0."""
    (x0, y0), (x1, y1) = p0_in, p1_in
    d = np.array([x1 - x0, y1 - y0], float)
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)
    o = ARROW_PAIR_OFFSET_INCHES * n
    draw_arrow(fig, (x0 + o[0], y0 + o[1]), (x1 + o[0], y1 + o[1]),
               FORWARD, dashed=True, zorder=zorder)
    if gradient:
        draw_arrow(fig, (x1 - o[0], y1 - o[1]), (x0 - o[0], y0 - o[1]),
                   GRADIENT, dashed=True, zorder=zorder)


def draw_loss_bracket(fig, x_left_in, x_right_in, y_top_in, depth_in=0.08,
                      label="Loss", color=LOSS, zorder=5):
    """The loss bracket: two upward ticks joined below, label under it."""
    fw, fh = fig.get_size_inches()
    xs = [x_left_in, x_left_in, x_right_in, x_right_in]
    ys = [y_top_in, y_top_in - depth_in, y_top_in - depth_in, y_top_in]
    fig.add_artist(plt.Line2D([x / fw for x in xs], [y / fh for y in ys],
                              color=color, lw=BRACKET_LW_PT, zorder=zorder))
    for x in (x_left_in, x_right_in):
        draw_arrow(fig, (x, y_top_in - 0.02), (x, y_top_in + 0.01), color,
                   lw=BRACKET_LW_PT, zorder=zorder)
    fx, fy = _frac(fig, (x_left_in + x_right_in) / 2, y_top_in - depth_in - 0.02)
    fig.text(fx, fy, label, ha="center", va="top", color=color,
             fontsize=FIGURE_HEADER_PT, fontweight="bold", zorder=zorder)


def figtext(fig, x_in, y_in, text, **kwargs):
    """fig.text at inch coordinates; size defaults to FIGURE_BASE_PT."""
    kwargs.setdefault("fontsize", FIGURE_BASE_PT)
    kwargs.setdefault("color", INK)
    fx, fy = _frac(fig, x_in, y_in)
    return fig.text(fx, fy, text, transform=fig.transFigure, **kwargs)


def save(fig, out_stem):
    """Write <stem>.pdf (transparent) and <stem>.png (white) at 300 dpi."""
    fig.savefig(out_stem + ".pdf", dpi=300, facecolor="none")
    fig.savefig(out_stem + ".png", dpi=300, facecolor="white")


def draw_block_arrows(fig, x_in, y_in, w_in, h_in, direction="right", gradient=True, color=None, zorder=5):
    """The solid block arrows: a green forward arrow and,
    beside it, a red gradient arrow pointing back; ``gradient=False`` or ``color`` draws one arrow (the grey
    real-data arrow). ``direction`` is the forward direction: "right" or "down"."""
    from matplotlib.patches import Polygon
    fw, fh = fig.get_size_inches()

    def arrow(x0, y0, w, h, d, col):
        # a shaft plus a head, in a (w x h) box; d: +1 right/down, -1 left/up
        if direction == "right":
            t = h * 0.36
            pts = [(0, h / 2 - t / 2), (w * 0.62, h / 2 - t / 2), (w * 0.62, 0), (w, h / 2),
                   (w * 0.62, h), (w * 0.62, h / 2 + t / 2), (0, h / 2 + t / 2)]
            if d < 0:
                pts = [(w - px, py) for px, py in pts]
        else:
            t = w * 0.36
            pts = [(w / 2 - t / 2, h), (w / 2 - t / 2, h * 0.38), (0, h * 0.38), (w / 2, 0),
                   (w, h * 0.38), (w / 2 + t / 2, h * 0.38), (w / 2 + t / 2, h)]
            if d < 0:
                pts = [(px, h - py) for px, py in pts]
        fig.patches.append(Polygon([((x0 + px) / fw, (y0 + py) / fh) for px, py in pts], closed=True,
                                   transform=fig.transFigure, facecolor=col, edgecolor="none", zorder=zorder))

    if color is not None or not gradient:
        arrow(x_in, y_in, w_in, h_in, +1, color or FORWARD)
        return
    if direction == "right":
        hh = h_in * 0.46
        arrow(x_in, y_in + h_in - hh, w_in, hh, +1, FORWARD)
        arrow(x_in, y_in, w_in, hh, -1, GRADIENT)
    else:
        ww = w_in * 0.46
        arrow(x_in, y_in, ww, h_in, +1, FORWARD)
        arrow(x_in + w_in - ww, y_in, ww, h_in, -1, GRADIENT)
