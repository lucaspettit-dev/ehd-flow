"""CoulombField: electrostatic field lines for image-defined electrodes.

Single responsibility: the physics and rendering of the Coulomb field.
Electrode polygons come from :class:`ehd_flow.polygons.PolygonExtractor`
(red channel -> positive, blue channel -> negative); this class owns the
charge discretization, field summation, line tracing, and rendering.

Each electrode polygon is modeled as a uniformly charged conducting ring:
its total charge is spread over point charges along its boundary, and the
field is the Coulomb sum over all of them (k = 1, arbitrary units).

Field lines are seeded around the electrodes and integrated along E, drawn
in the style of magnetic field-line diagrams. Arrowheads always point along
E, i.e. the direction a positive ion drifts (positive -> negative).

Charge ratio R sets |Q_negative| / |Q_positive| = R:
  R = 1   -> equal magnitude (default)
  R = 0.5 -> positive is 2x more charged than negative
  R = 2   -> negative is 2x more charged than positive
"""

import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .polygons import PolygonExtractor, show_image


def efield(pts, cpos, cq, soft, chunk=20000):
    """Coulomb field at pts (G, 2) from charges at cpos (M, 2). k = 1."""
    pts = np.asarray(pts, dtype=float)
    E = np.zeros_like(pts)
    if len(cq) == 0:
        return E
    for i in range(0, len(pts), chunk):
        c = pts[i:i + chunk]
        d = c[:, None, :] - cpos[None, :, :]          # (C, M, 2)
        r2 = np.maximum((d ** 2).sum(-1), soft ** 2)  # (C, M)
        E[i:i + chunk] = (d / r2[..., None] ** 1.5 * cq[None, :, None]).sum(1)
    return E


class CoulombField:
    """Field lines + arrows for image-defined electrodes.

    Parameters:
      charge_ratio   -- |Q_negative| / |Q_positive|.
      epsilon        -- polygon simplification as fraction of perimeter.
      min_area       -- ignore blobs smaller than this many px^2.
      charge_spacing -- spacing (px) of discrete charges along edges.
      seeds          -- field-line seeds per electrode.
      line_step      -- field-line integration step (px).
    """

    def __init__(self, charge_ratio=1.0, epsilon=0.002, min_area=10.0,
                 charge_spacing=3.0, seeds=48, line_step=2.5):
        if cv2 is None:
            raise ImportError(
                "opencv-python is required (pip install opencv-python)")
        if charge_ratio < 0:
            raise ValueError("charge_ratio must be >= 0")
        self.charge_ratio = charge_ratio
        self.charge_spacing = charge_spacing
        self.seeds = seeds
        self.line_step = line_step
        self.extractor = PolygonExtractor(epsilon, min_area)

    # -- pipeline stages -------------------------------------------------

    def electrodes_from_image(self, path):
        """Return (pos_polys, neg_polys, color_bgr) from the image channels.

        Red channel (BGR index 2) -> positive electrodes, blue channel
        (index 0) -> negative electrodes.
        """
        color = cv2.imread(path, cv2.IMREAD_COLOR)
        if color is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        pos = self.extractor.extract_polygons(
            self.extractor.binarize(color[:, :, 2]))
        neg = self.extractor.extract_polygons(
            self.extractor.binarize(color[:, :, 0]))
        print(f"positive electrodes (red): {len(pos)}, "
              f"negative electrodes (blue): {len(neg)}")
        if not pos and not neg:
            raise ValueError("no electrodes found in red or blue channels")
        return pos, neg, color

    def discretize_boundary(self, polygons, total_charge):
        """Spread total_charge uniformly over point charges along edges.

        Returns (positions (M, 2) float array, charges (M,) float array).
        """
        positions, counts = [], []
        for _, pts in polygons:
            p = np.asarray(pts, dtype=float)
            edge_pts = []
            for a, b in zip(p, np.roll(p, -1, axis=0)):
                length = float(np.linalg.norm(b - a))
                n = max(1, int(round(length / self.charge_spacing)))
                for t in np.linspace(0.0, 1.0, n, endpoint=False):
                    edge_pts.append(a + t * (b - a))
            positions.extend(edge_pts)
            counts.append(len(edge_pts))
        if not positions:
            return np.zeros((0, 2)), np.zeros(0)
        positions = np.asarray(positions)
        charges = np.concatenate([
            np.full(c, total_charge / c) for c in counts
        ])
        return positions, charges

    def outward_seeds(self, polygons, gap):
        """Seed points just outside each polygon boundary.

        Uses cv2.pointPolygonTest so concave shapes get true outward
        normals. Returns (S, 2) array of seed points.
        """
        seeds = []
        contours = [np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
                    for _, pts in polygons]
        for contour in contours:
            n = len(contour)
            stride = max(1, n // self.seeds)
            for j in range(0, n, stride):
                p0 = contour[j, 0]
                p1 = contour[(j + 1) % n, 0]
                edge = p1 - p0
                L = float(np.linalg.norm(edge)) + 1e-12
                n1 = np.array([-edge[1], edge[0]]) / L
                mid = (p0 + p1) / 2
                # pointPolygonTest: +ve inside, -ve outside
                s1 = cv2.pointPolygonTest(
                    contour,
                    (float(mid[0] + n1[0] * gap),
                     float(mid[1] + n1[1] * gap)), True)
                n_out = n1 if s1 < 0 else -n1
                seeds.append(mid + n_out * gap)
        return np.asarray(seeds, dtype=float).reshape(-1, 2)

    def trace_field_lines(self, seeds, cpos_all, q_all, cpos_stop, bounds,
                          capture, max_steps):
        """Integrate dr/ds = E/|E| from each seed; stop near cpos_stop or
        outside bounds. Returns a list of (K, 2) polylines."""
        h, w = bounds
        step = self.line_step
        lines = []
        for s in seeds:
            p = s.copy()
            line = [p.copy()]
            for _ in range(max_steps):
                E = efield(p[None, :], cpos_all, q_all,
                           soft=step * 0.75)[0]
                norm = float(np.linalg.norm(E))
                if norm < 1e-12:
                    break
                p = p + E / norm * step
                if not (-step <= p[0] <= w + step and
                        -step <= p[1] <= h + step):
                    break
                if len(cpos_stop) and np.min(
                        np.linalg.norm(cpos_stop - p, axis=1)) < capture:
                    line.append(p.copy())
                    break
                line.append(p.copy())
            if len(line) > 4:
                lines.append(np.asarray(line))
        return lines

    @staticmethod
    def draw_arrow(img, pt, direction, color, size=9):
        """Draw a small arrowhead at pt pointing along direction (unit)."""
        d = np.asarray(direction, dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return
        d = d / n
        start = (pt - d * size * 1.4).astype(int)
        end = (pt + d * size * 0.5).astype(int)
        cv2.arrowedLine(img, tuple(start), tuple(end), color, 2,
                        tipLength=0.45, line_type=cv2.LINE_AA)

    # -- full pipeline ---------------------------------------------------

    def render(self, input_image, output=None):
        """Run the full pipeline: image -> field-line PNG.

        Returns the output path.
        """
        pos_polys, neg_polys, color = self.electrodes_from_image(input_image)
        h, w = color.shape[:2]

        pos = self.discretize_boundary(pos_polys, +1.0)
        neg = self.discretize_boundary(neg_polys, -self.charge_ratio)
        cpos_all = np.vstack([p for p, _ in (pos, neg) if len(p)])
        q_all = np.concatenate([q for _, q in (pos, neg) if len(q)])

        gap = self.charge_spacing
        max_steps = int(2.5 * np.hypot(w, h) / self.line_step)
        capture = self.charge_spacing * 1.5

        lines = []
        if len(pos[0]):
            seeds = self.outward_seeds(pos_polys, gap)
            # follow +E (positive-ion drift); stop at negative electrodes
            lines += self.trace_field_lines(seeds, cpos_all, q_all, neg[0],
                                            (h, w), capture, max_steps)
        if len(neg[0]):
            seeds = self.outward_seeds(neg_polys, gap)
            # trace backwards along -E from negatives; arrows still follow +E
            back = self.trace_field_lines(seeds, cpos_all, -q_all, pos[0],
                                          (h, w), capture, max_steps)
            lines += [line[::-1] for line in back]
        print(f"traced {len(lines)} field lines")

        # render on a dimmed copy of the input for contrast
        canvas = (color.astype(float) * 0.72).astype(np.uint8)
        for pts in pos_polys:
            cv2.polylines(canvas, [np.asarray(pts[1], dtype=np.float32)
                                   .astype(np.int32)], True, (0, 0, 255), 2)
        for pts in neg_polys:
            cv2.polylines(canvas, [np.asarray(pts[1], dtype=np.float32)
                                   .astype(np.int32)], True, (255, 0, 0), 2)
        line_color = (255, 255, 0)  # cyan in BGR
        for line in lines:
            cv2.polylines(canvas, [line.astype(np.int32)], False, line_color,
                          1, lineType=cv2.LINE_AA)
            E = efield(line, cpos_all, q_all, soft=self.line_step * 0.75)
            for frac in (0.3, 0.55, 0.8):
                k = int(len(line) * frac)
                if k < 1 or k >= len(line) - 1:
                    continue
                self.draw_arrow(canvas, line[k], E[k], line_color)

        legend = (f"red +1.0 / blue -{self.charge_ratio:g}   "
                  f"arrows: +ion drift")
        cv2.rectangle(canvas, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.putText(canvas, legend, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

        out = output or os.path.splitext(input_image)[0] + "_field.png"
        cv2.imwrite(out, canvas)
        return out

    def show(self, path):
        """Open the rendered image in a preview window (headless-safe)."""
        show_image(path, "coulomb field (cyan), + red / - blue")
