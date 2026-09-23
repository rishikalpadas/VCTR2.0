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


def split_linework(ink: np.ndarray, max_width: float) -> tuple[np.ndarray, np.ndarray]:
    """(thick, thin) partition of an ink mask.

    An opening by a disc wider than ``max_width`` keeps exactly the shapes too
    wide to be a stroke. They are kept as filled shapes; everything else is
    line work.
    """
    radius = max(1, round(max_width / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = ink.astype(np.uint8)
    core = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Grow the cores back by one pixel so a stroke meeting a thick shape
    # hands over at the shape's edge rather than leaving its fringe as a
    # sliver of "line work".
    thick = (cv2.dilate(core, np.ones((3, 3), np.uint8)) > 0) & ink
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
        points = _smooth(chain, smooth_sigma, closed)
        if not closed:
            points = _extend_ends(points, ends, width / 2)
        strokes.append(Stroke(points=points, closed=closed, width=width))
    return strokes


def stroke_to_path_d(stroke: Stroke, epsilon: float = 0.4) -> str:
    """Catmull-Rom cubic path through the simplified centreline."""
    pts = stroke.points.astype(np.float32).reshape(-1, 1, 2)
    simple = cv2.approxPolyDP(pts, epsilon, stroke.closed).reshape(-1, 2).astype(np.float64)
    if len(simple) < 2:
        simple = stroke.points[[0, -1]]
    parts = [f"M{simple[0][0]:.2f},{simple[0][1]:.2f}"]
    n = len(simple)
    segments = n if stroke.closed else n - 1
    for i in range(segments):
        p0 = simple[(i - 1) % n] if (stroke.closed or i > 0) else simple[i]
        p1 = simple[i]
        p2 = simple[(i + 1) % n]
        p3 = simple[(i + 2) % n] if (stroke.closed or i + 2 < n) else p2
        # Chord-length tangents: a uniform Catmull-Rom handle is sized by the
        # neighbouring points' spacing, so a short segment next to a long one
        # (a straight rule simplified to three points) gets a handle many
        # times its own length and loops out past its end.
        c1 = p1 + _tangent(p0, p1, p2) * np.linalg.norm(p2 - p1) / 3
        c2 = p2 - _tangent(p1, p2, p3) * np.linalg.norm(p2 - p1) / 3
        parts.append(
            f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}"
        )
    if stroke.closed:
        parts.append("Z")
    return " ".join(parts)


def _tangent(before: np.ndarray, at: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Unit-speed tangent at ``at`` under chord-length parametrization."""
    span = np.linalg.norm(at - before) + np.linalg.norm(after - at)
    return (after - before) / span if span > 0 else np.zeros(2)


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
    """
    if len(points) < 2:
        return points
    result = points.copy()
    span = min(len(points) - 1, 4)
    for end, ref, flag in ((0, span, ends[0]), (-1, -1 - span, ends[1])):
        if flag != _ANCHOR:
            continue
        direction = points[end] - points[ref]
        norm = np.linalg.norm(direction)
        if norm > 0:
            result[end] = points[end] + direction / norm * distance
    return result
