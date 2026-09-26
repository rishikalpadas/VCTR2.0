"""Centreline tracing for thin line work.

Tracing a thin stroke by its *outline* - which is all Potrace can do - fits
the two edges of the stroke independently. At tracing resolution the line work
on typical sticker art is two or three pixels wide, so the half-pixel error in
each edge fit is a 20-25% swing in stroke width, and it swings differently
along every stroke: the line work comes back visibly heavier in some places
than others. Where two strokes meet, the outline also has to wrap around the
little spur the pixel skeleton of a blob always has, which is the nub that
shows at every junction.

Neither can be tuned away while the stroke is an outline. So thin line work is
traced by its *centre* instead and emitted as an SVG stroke of one width:

  1. split the ink into thin strokes and thick shapes (lettering, solid bars)
     with a morphological opening - only the thin part is re-traced here;
  2. skeletonize the thin part and walk the skeleton as a graph, chains of
     pixels between junctions and ends, plus closed loops;
  3. drop short spurs that end in nothing - they are the nubs;
  4. smooth each chain and fit it with Catmull-Rom cubics;
  5. stroke every chain of one connected network at that network's mean
     width, measured as area over centreline length.

Round caps and joins make chains that share a junction meet seamlessly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# Mean length per pixel of an 8-connected digital line over all orientations
# (between 1 for axis-aligned and sqrt(2) for diagonal runs).
_PIXEL_LENGTH = 1.11

# How a chain end terminates.
_END = "end"  # dangles in nothing
_JUNCTION = "junction"  # meets other chains
_ANCHOR = "anchor"  # runs into a thick shape


@dataclass(frozen=True)
class Stroke:
    points: np.ndarray  # (n, 2) float x, y
    closed: bool
    width: float
    ends: tuple[str, str] = (_END, _END)


def split_linework(ink: np.ndarray, max_width: float) -> tuple[np.ndarray, np.ndarray]:
    """(thick, thin) partition of an ink mask.

    An opening by a disc wider than ``max_width`` keeps exactly the shapes too
    wide to be a stroke. They are kept as filled shapes; everything else is
    line work.
    """
    radius = max(1, round(max_width / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = ink.astype(np.uint8)
    core = cv2.erode(mask, kernel)
    # A core only a few stroke widths across is where two strokes run
    # together, not a shape; left in, it is traced as a filled blob sitting
    # in the line work. Real shapes - letters, bars, a full stop - are far
    # larger than that.
    min_area = (3 * max_width) ** 2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    small = stats[1:, cv2.CC_STAT_AREA] < min_area
    if small.any():
        core[np.isin(labels, np.nonzero(small)[0] + 1)] = 0
    # Growing the cores back by the full erosion radius (an opening) rounds
    # every convex corner of a shape by that radius - letter corners and the
    # tips of a bar come back soft. One more radius restores them, and the
    # stroke roots it also takes in are hidden under the stroke anyway.
    reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * radius + 1, 4 * radius + 1))
    thick = (cv2.dilate(core, reach) > 0) & ink
    return thick, ink & ~thick


def trace_strokes(
    thin: np.ndarray,
    *,
    anchor: np.ndarray | None = None,
    smooth_sigma: float = 2.0,
    spur_factor: float = 2.0,
    weight: float = 1.0,
) -> list[Stroke]:
    """Centreline strokes for every thin network in ``thin``.

    ``anchor`` marks pixels (typically the thick shapes) that a chain ending
    next to is attached to rather than dangling, so it is never pruned as a
    spur and is extended under the shape so the round cap cannot leave a gap.
    """
    skeleton = skeletonize(thin)
    if not skeleton.any():
        return []

    count, components = cv2.connectedComponents(thin.astype(np.uint8), connectivity=8)
    area = np.bincount(components[thin], minlength=count).astype(np.float64)
    length = np.bincount(components[skeleton], minlength=count) * _PIXEL_LENGTH
    widths = np.divide(area, length, out=np.zeros(count), where=length > 0)

    near_anchor = None
    if anchor is not None and anchor.any():
        near_anchor = cv2.dilate(anchor.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    strokes: list[Stroke] = []
    for chain, closed, ends in _walk(skeleton, near_anchor):
        ys, xs = chain[:, 1].astype(int), chain[:, 0].astype(int)
        width = float(widths[components[ys[len(ys) // 2], xs[len(xs) // 2]]]) * weight
        if width <= 0:
            continue
        dangling = ends.count(_END)
        if not closed and dangling == 1 and _length(chain) < spur_factor * width:
            continue  # a spur off a junction: the nub, not artwork
        # +0.5: pixel (x, y) covers [x, x+1] in the SVG's coordinates, the
        # convention the Potrace-traced fills use, so its centre is half a
        # pixel in from the index. Without it every stroke sits up and left
        # of the ink it came from.
        points = _smooth(chain, smooth_sigma, closed) + 0.5
        if not closed:
            # A full width past the shape's edge, not half: the round cap has
            # to cover the shape's corner, which now comes back sharp.
            points = _extend_ends(points, ends, width)
        strokes.append(Stroke(points=points, closed=closed, width=width, ends=ends))
    return strokes


def stroke_to_path_d(
    stroke: Stroke, tolerance: float = 0.8, corner_angle: float = 50.0
) -> str:
    """Least-squares Bezier path along the centreline.

    A curve that interpolates the centreline points reproduces every pixel of
    skeleton wobble - a straight rule comes back gently wavy and a circle
    lumpy. Fitting instead (Schneider's algorithm) places as few cubics as
    stay within ``tolerance`` pixels of the points, so the wobble averages
    out. Runs are split at genuine corners first so those stay sharp, a run
    that is straight within tolerance becomes an exact line, and a closed
    loop that is round within tolerance becomes an exact circle.
    """
    # The skeleton's wobble comes from the raster stroke's own edges, so it
    # grows with the stroke's traced width - not with any fixed pixel budget.
    # Without this floor a lower supersample factor shrinks the tolerance
    # below the wobble and the rules go wavy again.
    tolerance = max(tolerance, _WOBBLE_PER_WIDTH * stroke.width)
    pts = _dedupe(stroke.points.astype(np.float64))
    if len(pts) < 2:
        pts = stroke.points[[0, -1]].astype(np.float64)
    if stroke.closed and len(pts) >= 8:
        circle = _fit_circle(pts, max(tolerance, stroke.width / 2))
        if circle is not None:
            return _circle_d(*circle)

    window = max(3, round(stroke.width * 1.5))
    sigma = _RUN_SMOOTH_PER_WIDTH * stroke.width
    parts = [f"M{pts[0][0]:.2f},{pts[0][1]:.2f}"]
    if stroke.closed:
        corners = _corners(pts, window, corner_angle, closed=True)
        if corners:
            pts = np.roll(pts, -corners[0], axis=0)
            corners = [(c - corners[0]) % len(pts) for c in corners]
            parts = [f"M{pts[0][0]:.2f},{pts[0][1]:.2f}"]
            ring = np.vstack([pts, pts[:1]])
            bounds = corners + [len(pts)]
            for a, b in zip(bounds, bounds[1:]):
                parts.extend(_fit_run(_smooth_run(ring[a : b + 1], sigma), tolerance))
        else:
            # One run around the loop, with the same tangent at both ends so
            # the seam where it closes is invisible.
            pts = _smooth(pts, min(sigma, len(pts) / 8), closed=True)
            ring = np.vstack([pts, pts[:1]])
            k = min(window, len(pts) // 4) or 1
            tangent = _unit(pts[k % len(pts)] - pts[-k])
            parts.extend(_fit_run(ring, tolerance, tangent, -tangent))
        parts.append("Z")
        return " ".join(parts)

    corners = _corners(pts, window, corner_angle, closed=False)
    bounds = [0] + corners + [len(pts) - 1]
    last = len(bounds) - 2
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        run = _smooth_run(pts[a : b + 1], sigma)
        # A skeleton bends in its last pixel or two, so an end that meets
        # nothing may slide onto the run's fitted line. Junction ends and
        # corners are shared with another run and must stay put.
        free_start = i == 0 and stroke.ends[0] != _JUNCTION
        free_end = i == last and stroke.ends[1] != _JUNCTION
        line = _straight(run, tolerance, free_start, free_end, trim=round(stroke.width))
        if line is not None:
            start, end = line
            if i == 0:
                parts = [f"M{start[0]:.2f},{start[1]:.2f}"]
            parts.append(f"L{end[0]:.2f},{end[1]:.2f}")
        else:
            parts.extend(_fit_run(run, tolerance))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

_KAPPA = 0.5522847498  # cubic handle length for a quarter circle
_WOBBLE_PER_WIDTH = 0.3
# Smoothing along a corner-free run, as a multiple of the stroke width. The
# raster edge of a thin stroke jogs by a pixel every few dozen pixels, which
# no fit tolerance tight enough to keep real curves can ignore: the fit splits
# at every jog and the rule comes back rippled. A low-pass this wide flattens
# the jogs, and on a gentle curve moves the line by only sigma^2 / 2R - a
# tenth of a pixel on a 200px-radius arc. Applied after corners are found, so
# an apex is never rounded by it.
_RUN_SMOOTH_PER_WIDTH = 2.5


def _smooth_run(points: np.ndarray, sigma: float) -> np.ndarray:
    """Low-pass one run with its ends pinned; short runs get less.

    Padded by point reflection through each end, which a straight or evenly
    curving run continues exactly - padding by repeating the end point would
    drag the last few points towards it and bend every rule near its ends.
    """
    sigma = min(sigma, len(points) / 6)
    if sigma <= 0.5 or len(points) < 4:
        return points
    radius = min(int(3 * sigma), len(points) - 1)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    head = 2 * points[0] - points[radius:0:-1]
    tail = 2 * points[-1] - points[-2 : -radius - 2 : -1]
    padded = np.vstack([head, points, tail])
    smoothed = np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in (0, 1)], axis=1
    )
    smoothed[0], smoothed[-1] = points[0], points[-1]
    return smoothed


def _dedupe(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
    return points[keep]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(2)


def _corners(points: np.ndarray, window: int, angle: float, closed: bool) -> list[int]:
    """Indices where the chain turns by more than ``angle`` degrees.

    The turn is measured between chords ``window`` points either side, so a
    one-pixel wiggle cannot register, and only the sharpest point of each
    turning region is kept.
    """
    n = len(points)
    if n < 2 * window + 1:
        return []
    idx = np.arange(n)
    if closed:
        before = points[(idx - window) % n]
        after = points[(idx + window) % n]
        valid = np.ones(n, bool)
    else:
        before = points[np.clip(idx - window, 0, n - 1)]
        after = points[np.clip(idx + window, 0, n - 1)]
        valid = (idx >= window) & (idx <= n - 1 - window)
    a = points - before
    b = after - points
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.einsum("ij,ij->i", a, b) / np.where(norms > 0, norms, 1)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    turn[~valid] = 0
    corners = []
    for i in np.argsort(-turn):
        if turn[i] < angle:
            break
        near = [c for c in corners if min(abs(c - i), n - abs(c - i) if closed else n) <= window]
        if not near:
            corners.append(int(i))
    return sorted(corners)


def _fit_run(
    points: np.ndarray,
    tolerance: float,
    t_start: np.ndarray | None = None,
    t_end: np.ndarray | None = None,
) -> list[str]:
    """Path commands (after the moveto) for one corner-free run."""
    end = points[-1]
    if len(points) <= 2 or _max_chord_offset(points) <= tolerance:
        return [f"L{end[0]:.2f},{end[1]:.2f}"]
    k = min(4, len(points) - 1)
    if t_start is None:
        t_start = _unit(points[k] - points[0])
    if t_end is None:
        t_end = _unit(points[-1 - k] - points[-1])
    return [
        f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}"
        for _, c1, c2, p3 in _fit_cubic(points, t_start, t_end, tolerance, depth=0)
    ]


def _straight(
    points: np.ndarray,
    tolerance: float,
    free_start: bool,
    free_end: bool,
    trim: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """(start, end) of the run as one line, if it is straight within tolerance.

    Judged against the least-squares line through the points, not the chord
    between the two ends, so an end that hooks by a pixel neither tilts nor
    offsets the whole rule. A free end's last ``trim`` points - where a
    skeleton bends - are left out of the judgement, and the end is projected
    onto the line; fixed ends must already lie within tolerance of it.
    """
    if len(points) < 2:
        return None
    lo = trim if free_start else 0
    hi = len(points) - (trim if free_end else 0)
    core = points[lo:hi] if hi - lo >= max(2, len(points) // 2) else points
    centre = core.mean(axis=0)
    _, _, vt = np.linalg.svd(core - centre, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    offsets = np.abs((points - centre) @ normal)
    checked = np.ones(len(points), bool)
    if core is not points:
        checked[:lo] = False
        checked[hi:] = False
    if offsets[checked].max() > tolerance:
        return None

    def project(p: np.ndarray) -> np.ndarray:
        return centre + direction * float((p - centre) @ direction)

    start = project(points[0]) if free_start else points[0]
    end = project(points[-1]) if free_end else points[-1]
    return start, end


def _max_chord_offset(points: np.ndarray) -> float:
    start, end = points[0], points[-1]
    chord = end - start
    length = np.linalg.norm(chord)
    rel = points - start
    if length < 1e-9:
        return float(np.linalg.norm(rel, axis=1).max())
    return float(np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]).max() / length)


def _fit_cubic(points, t_start, t_end, tolerance, depth):
    """Schneider's recursive least-squares cubic fit (Graphics Gems, 1990)."""
    n = len(points)
    if n == 2 or depth > 12:
        dist = np.linalg.norm(points[-1] - points[0]) / 3
        return [(points[0], points[0] + t_start * dist, points[-1] + t_end * dist, points[-1])]

    u = _chord_params(points)
    bez = _generate_bezier(points, u, t_start, t_end)
    error, split = _max_error(points, bez, u)
    if error <= tolerance:
        return [bez]
    if error <= tolerance * 4:
        for _ in range(4):
            u = _reparameterize(points, bez, u)
            bez = _generate_bezier(points, u, t_start, t_end)
            error, split = _max_error(points, bez, u)
            if error <= tolerance:
                return [bez]

    split = min(max(split, 1), n - 2)
    k = min(3, split, n - 1 - split)
    t_mid = _unit(points[split - k] - points[split + k])
    if not t_mid.any():
        t_mid = _unit(points[split - 1] - points[split + 1])
    return _fit_cubic(points[: split + 1], t_start, t_mid, tolerance, depth + 1) + _fit_cubic(
        points[split:], -t_mid, t_end, tolerance, depth + 1
    )


def _chord_params(points: np.ndarray) -> np.ndarray:
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(points))


def _bezier(bez, u: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = bez
    u = u[:, None]
    m = 1 - u
    return m**3 * p0 + 3 * m * m * u * p1 + 3 * m * u * u * p2 + u**3 * p3


def _generate_bezier(points, u, t_start, t_end):
    p0, p3 = points[0], points[-1]
    b1 = 3 * u * (1 - u) ** 2
    b2 = 3 * u * u * (1 - u)
    a1 = t_start[None, :] * b1[:, None]
    a2 = t_end[None, :] * b2[:, None]
    c00 = float(np.einsum("ij,ij->", a1, a1))
    c01 = float(np.einsum("ij,ij->", a1, a2))
    c11 = float(np.einsum("ij,ij->", a2, a2))
    base = _bezier((p0, p0, p3, p3), u)
    rest = points - base
    x0 = float(np.einsum("ij,ij->", a1, rest))
    x1 = float(np.einsum("ij,ij->", a2, rest))
    det = c00 * c11 - c01 * c01
    seg = np.linalg.norm(p3 - p0)
    alpha1 = alpha2 = 0.0
    if abs(det) > 1e-12:
        alpha1 = (x0 * c11 - x1 * c01) / det
        alpha2 = (c00 * x1 - c01 * x0) / det
    # Degenerate or backwards handles loop the curve; fall back to the
    # Wu/Barsky heuristic of a third of the chord.
    eps = 1e-6 * seg
    if alpha1 < eps or alpha2 < eps or alpha1 > seg * 2 or alpha2 > seg * 2:
        alpha1 = alpha2 = seg / 3
    return (p0, p0 + t_start * alpha1, p3 + t_end * alpha2, p3)


def _max_error(points, bez, u) -> tuple[float, int]:
    dist = np.linalg.norm(_bezier(bez, u) - points, axis=1)
    dist[0] = dist[-1] = 0
    i = int(np.argmax(dist))
    return float(dist[i]), i


def _reparameterize(points, bez, u):
    p0, p1, p2, p3 = bez
    uu = u[:, None]
    m = 1 - uu
    q = _bezier(bez, u)
    q1 = 3 * (m * m * (p1 - p0) + 2 * m * uu * (p2 - p1) + uu * uu * (p3 - p2))
    q2 = 6 * (m * (p2 - 2 * p1 + p0) + uu * (p3 - 2 * p2 + p1))
    diff = q - points
    num = np.einsum("ij,ij->i", diff, q1)
    den = np.einsum("ij,ij->i", q1, q1) + np.einsum("ij,ij->i", diff, q2)
    step = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12)
    new = np.clip(u - step, 0, 1)
    new[0], new[-1] = 0.0, 1.0
    return np.maximum.accumulate(new)


def _fit_circle(points: np.ndarray, max_offset: float):
    """(cx, cy, r) if the closed chain is a circle within ``max_offset``."""
    x, y = points[:, 0], points[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    offsets = np.abs(np.hypot(x - cx, y - cy) - r)
    if r < 4 or offsets.max() > max_offset:
        return None
    # The chain must go all the way round, not just trace an arc of one.
    angles = np.sort(np.arctan2(y - cy, x - cx))
    if np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() > np.pi / 6:
        return None
    return float(cx), float(cy), r


def _circle_d(cx: float, cy: float, r: float) -> str:
    h = r * _KAPPA
    pts = [(cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r)]
    handles = [
        ((cx + r, cy + h), (cx + h, cy + r)),
        ((cx - h, cy + r), (cx - r, cy + h)),
        ((cx - r, cy - h), (cx - h, cy - r)),
        ((cx + h, cy - r), (cx + r, cy - h)),
    ]
    parts = [f"M{pts[0][0]:.2f},{pts[0][1]:.2f}"]
    for i, (c1, c2) in enumerate(handles):
        e = pts[(i + 1) % 4]
        parts.append(f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {e[0]:.2f},{e[1]:.2f}")
    parts.append("Z")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Skeleton graph walk
# ---------------------------------------------------------------------------


def _walk(skeleton: np.ndarray, near_anchor: np.ndarray | None):
    """Yield (points, closed, (start_kind, end_kind)) per chain.

    Junction pixels (3+ neighbours) are grouped into clusters, each standing
    for one junction at the cluster's centroid; chains run between clusters
    and endpoints. What is left once every chain is walked is closed loops.
    """
    height, width = skeleton.shape
    padded = np.pad(skeleton, 1).astype(np.uint8)
    neighbours = cv2.filter2D(padded, -1, np.ones((3, 3), np.float32))[1:-1, 1:-1] - skeleton
    junction = skeleton & (neighbours >= 3)
    endpoint = skeleton & (neighbours <= 1)
    node = junction | endpoint

    _, clusters = cv2.connectedComponents(junction.astype(np.uint8), connectivity=8)
    centroids: dict[int, tuple[float, float]] = {}
    for label in np.unique(clusters[junction]):
        ys, xs = np.nonzero(clusters == label)
        centroids[int(label)] = (float(xs.mean()), float(ys.mean()))

    def node_point(y: int, x: int) -> tuple[float, float]:
        label = int(clusters[y, x])
        return centroids[label] if label else (float(x), float(y))

    def kind(y: int, x: int) -> str:
        if junction[y, x]:
            return _JUNCTION
        if near_anchor is not None and near_anchor[y, x]:
            return _ANCHOR
        return _END

    def around(y: int, x: int):
        for dy, dx in _NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and skeleton[ny, nx]:
                yield ny, nx

    visited = np.zeros_like(skeleton)

    for y, x in zip(*np.nonzero(node)):
        y, x = int(y), int(x)
        for ny, nx in around(y, x):
            if node[ny, nx] or visited[ny, nx]:
                continue
            chain = [node_point(y, x), (float(nx), float(ny))]
            visited[ny, nx] = True
            prev, cur = (y, x), (ny, nx)
            end = None
            while end is None:
                step = None
                for cy, cx in around(*cur):
                    if (cy, cx) == prev or (not node[cy, cx] and visited[cy, cx]):
                        continue
                    if node[cy, cx]:
                        # Never walk back into the cluster we started from.
                        if clusters[cy, cx] and clusters[cy, cx] == clusters[y, x] and len(chain) < 3:
                            continue
                        end = (cy, cx)
                        break
                    step = (cy, cx)
                if end is None:
                    if step is None:
                        end = cur  # dead end inside a pixel knot
                        break
                    visited[step] = True
                    chain.append((float(step[1]), float(step[0])))
                    prev, cur = cur, step
            if end != cur:
                chain.append(node_point(*end))
            yield np.array(chain), False, (kind(y, x), kind(*end))
        if endpoint[y, x] and not any(True for _ in around(y, x)):
            continue  # isolated pixel

    # Closed loops: rings with no junction or end at all.
    remaining = skeleton & ~node & ~visited
    for y, x in zip(*np.nonzero(remaining)):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        chain = [(float(x), float(y))]
        visited[y, x] = True
        cur = (y, x)
        while True:
            step = next(
                ((cy, cx) for cy, cx in around(*cur) if not visited[cy, cx] and not node[cy, cx]),
                None,
            )
            if step is None:
                break
            visited[step] = True
            chain.append((float(step[1]), float(step[0])))
            cur = step
        if len(chain) >= 3:
            yield np.array(chain), True, (_JUNCTION, _JUNCTION)


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _smooth(points: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    """Gaussian-smooth a pixel chain along its length, ends pinned."""
    if sigma <= 0 or len(points) < 3:
        return points.astype(np.float64)
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    mode = "wrap" if closed else "edge"
    padded = np.pad(points.astype(np.float64), ((radius, radius), (0, 0)), mode=mode)
    smoothed = np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in (0, 1)], axis=1
    )
    if not closed:
        # Pinned ends keep chains meeting at a junction on the same point.
        smoothed[0], smoothed[-1] = points[0], points[-1]
    return smoothed


def _extend_ends(points: np.ndarray, ends: tuple[str, str], distance: float) -> np.ndarray:
    """Push chain ends that run into a thick shape under it by ``distance``.

    Only those: a junction end must stay on the junction point or it pokes
    out past the join as a stub.

    The last stretch of skeleton before a thick shape bends towards the
    shape's nearest corner - the skeleton of a stroke fanning into a blob -
    and a cap that follows it pokes out past the shape as an ear. So that
    stretch (about a stroke width and a half) is dropped, and the end is
    pushed out along the direction of the stroke before it instead.
    """
    if len(points) < 2:
        return points
    trim = max(1, round(1.5 * distance))
    result = points
    for at_start, flag in ((True, ends[0]), (False, ends[1])):
        if flag != _ANCHOR or len(result) < 2:
            continue
        cut = min(trim, (len(result) - 2) // 2)
        pts = result if at_start else result[::-1]
        kept = pts[cut:]
        span = min(len(kept) - 1, max(4, trim))
        direction = kept[0] - kept[span]
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        reach = float(np.linalg.norm(pts[0] - kept[0])) + distance
        extended = np.vstack([kept[0] + direction / norm * reach, kept])
        result = extended if at_start else extended[::-1]
    return result
