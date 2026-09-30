"""Polygon obstacles: JSON loading, containment tests, rasterization.

Single responsibility: everything about obstacle *geometry* lives here --
parsing the case file, validating polygons, and rasterizing them to a
grid mask. No phiFlow, no physics.
"""

import json

import numpy as np


class Polygon:
    """A single named polygonal obstacle."""

    def __init__(self, name: str, points):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError(f"polygon '{name}' needs >= 3 (x, y) points")
        self.name = name
        self.points = pts

    def contains(self, pts: np.ndarray) -> np.ndarray:
        """Vectorized ray-casting point-in-polygon test.

        pts:  (..., 2) array of query points (any winding of the polygon)
        returns: (...,) boolean array
        """
        x = pts[..., 0]
        y = pts[..., 1]
        inside = np.zeros(x.shape, dtype=bool)
        poly = self.points
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # edge straddles the horizontal ray through the point?
            cond = ((y1 > y) != (y2 > y)) & (
                x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
            )
            inside ^= cond
        return inside

    def closed_points(self) -> np.ndarray:
        """Vertices with the first point appended (for outline plotting)."""
        return np.vstack([self.points, self.points[:1]])


def read_domain(path: str):
    """Return the optional ``{"x": .., "y": ..}`` domain dict from a case file."""
    with open(path) as f:
        return json.load(f).get("domain")


class ObstacleSet:
    """The collection of obstacles for one simulation case."""

    def __init__(self, polygons=()):
        self.polygons = list(polygons)

    @property
    def names(self):
        return [p.name for p in self.polygons]

    @classmethod
    def from_json(cls, path: str) -> "ObstacleSet":
        """Load polygons from a case JSON file.

        Each entry is either ``{"name": .., "points": [[x, y], ...]}``
        or a bare ``[[x, y], ...]`` list.
        """
        with open(path) as f:
            case = json.load(f)
        polygons = []
        for entry in case.get("polygons", []):
            if isinstance(entry, dict):
                name = entry.get("name", f"poly{len(polygons)}")
                pts = entry["points"]
            else:
                name = f"poly{len(polygons)}"
                pts = entry
            polygons.append(Polygon(name, pts))
        return cls(polygons)

    def rasterize(self, nx: int, ny: int, lx: float, ly: float) -> np.ndarray:
        """Boolean ``(nx, ny)`` solid mask at cell centers (x-first ordering)."""
        yy, xx = np.mgrid[0:ny, 0:nx]
        pts = np.stack([((xx + 0.5) / nx * lx).T,
                        ((yy + 0.5) / ny * ly).T], axis=-1)
        solid = np.zeros((nx, ny), dtype=bool)
        for poly in self.polygons:
            solid |= poly.contains(pts)
        return solid
