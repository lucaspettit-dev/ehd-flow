#!/usr/bin/env python3
"""coulomb_field.py -- Coulomb field lines for image-defined electrodes.

Takes a color image, extracts electrode polygons from two channels, and
renders the electrostatic field as field lines with arrows:

  - red channel   -> positive electrodes (charge +1 each)
  - blue channel  -> negative electrodes (charge -R each, R = --charge-ratio)

Each electrode polygon is modeled as a uniformly charged conducting ring:
its total charge is spread over point charges along its boundary, and the
field is the Coulomb sum over all of them (k = 1, arbitrary units).

Field lines are seeded around the electrodes and integrated along E, drawn
in the style of magnetic field-line diagrams. Arrowheads always point along
E, i.e. the direction a positive ion drifts (positive -> negative).

Charge ratio: --charge-ratio R sets |Q_negative| / |Q_positive| = R.
  R = 1   -> equal magnitude (default)
  R = 0.5 -> positive is 2x more charged than negative
  R = 2   -> negative is 2x more charged than positive

Usage:
    python coulomb_field.py electrodes.png
    python coulomb_field.py electrodes.png --output field.png \\
        --charge-ratio 0.5 --no-window

Notes:
  - Polygon extraction is shared with image_to_polygons.py (grayscale,
    binarize at 255/2, external contours only, Douglas-Peucker simplify).
  - The preview window needs a display; on headless machines it is skipped
    and only the output PNG is written.
"""

import argparse
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("error: opencv-python is required (pip install opencv-python)")

from image_to_polygons import binarize, extract_polygons


def discretize_boundary(polygons, total_charge, spacing):
    """Spread total_charge uniformly over point charges along polygon edges.

    Returns (positions (M, 2) float array, charges (M,) float array).
    """
    positions, counts = [], []
    for _, pts in polygons:
        p = np.asarray(pts, dtype=float)
        edge_pts = []
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            length = float(np.linalg.norm(b - a))
            n = max(1, int(round(length / spacing)))
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


def outward_seeds(polygons, per_electrode, gap):
    """Seed points just outside each polygon boundary.

    Uses cv2.pointPolygonTest so concave shapes get true outward normals.
    Returns (S, 2) array of seed points.
    """
    seeds = []
    contours = [np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
                for _, pts in polygons]
    for contour in contours:
        n = len(contour)
        stride = max(1, n // per_electrode)
        for j in range(0, n, stride):
            p0 = contour[j, 0]
            p1 = contour[(j + 1) % n, 0]
            edge = p1 - p0
            L = float(np.linalg.norm(edge)) + 1e-12
            n1 = np.array([-edge[1], edge[0]]) / L
            mid = (p0 + p1) / 2
            # pointPolygonTest: +ve inside, -ve outside
            s1 = cv2.pointPolygonTest(contour, (float(mid[0] + n1[0] * gap),
                                               float(mid[1] + n1[1] * gap)), True)
            n_out = n1 if s1 < 0 else -n1
            seeds.append(mid + n_out * gap)
    return np.asarray(seeds, dtype=float).reshape(-1, 2)


def trace_field_lines(seeds, cpos_all, q_all, cpos_stop, step, max_steps,
                      bounds, capture):
    """Integrate dr/ds = E/|E| from each seed; stop near cpos_stop or bounds.

    Returns a list of (K, 2) polylines.
    """
    h, w = bounds
    lines = []
    for s in seeds:
        p = s.copy()
        line = [p.copy()]
        for _ in range(max_steps):
            E = efield(p[None, :], cpos_all, q_all, soft=step * 0.75)[0]
            norm = float(np.linalg.norm(E))
            if norm < 1e-12:
                break
            p = p + E / norm * step
            if not (-step <= p[0] <= w + step and -step <= p[1] <= h + step):
                break
            if len(cpos_stop) and np.min(
                    np.linalg.norm(cpos_stop - p, axis=1)) < capture:
                line.append(p.copy())
                break
            line.append(p.copy())
        if len(line) > 4:
            lines.append(np.asarray(line))
    return lines


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_image", help="color image file (any format)")
    ap.add_argument("--output", default=None,
                    help="output PNG (default: <input>_field.png)")
    ap.add_argument("--charge-ratio", type=float, default=1.0,
                    help="|Q_negative| / |Q_positive| (default 1.0: equal; "
                         "0.5 -> positive 2x stronger; 2 -> negative 2x)")
    ap.add_argument("--epsilon", type=float, default=0.002,
                    help="polygon simplification as fraction of perimeter")
    ap.add_argument("--min-area", type=float, default=10.0,
                    help="ignore blobs smaller than this many px^2")
    ap.add_argument("--charge-spacing", type=float, default=3.0,
                    help="spacing (px) of discrete charges along edges")
    ap.add_argument("--seeds", type=int, default=48,
                    help="field-line seeds per electrode")
    ap.add_argument("--line-step", type=float, default=2.5,
                    help="field-line integration step (px)")
    ap.add_argument("--no-window", action="store_true",
                    help="do not try to open a preview window")
    args = ap.parse_args()

    if args.charge_ratio < 0:
        sys.exit("error: --charge-ratio must be >= 0")

    color = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    if color is None:
        sys.exit(f"error: cannot read image: {args.input_image}")
    h, w = color.shape[:2]

    # red channel (BGR index 2) -> positive, blue channel (index 0) -> negative
    pos_polys = extract_polygons(binarize(color[:, :, 2]),
                                 args.epsilon, args.min_area)
    neg_polys = extract_polygons(binarize(color[:, :, 0]),
                                 args.epsilon, args.min_area)
    print(f"positive electrodes (red): {len(pos_polys)}, "
          f"negative electrodes (blue): {len(neg_polys)}")
    if not pos_polys and not neg_polys:
        sys.exit("error: no electrodes found in red or blue channels")

    pos = discretize_boundary(pos_polys, +1.0, args.charge_spacing)
    neg = discretize_boundary(neg_polys, -args.charge_ratio,
                              args.charge_spacing)
    cpos_all = np.vstack([p for p, _ in (pos, neg) if len(p)])
    q_all = np.concatenate([q for _, q in (pos, neg) if len(q)])

    gap = args.charge_spacing
    max_steps = int(2.5 * np.hypot(w, h) / args.line_step)
    capture = args.charge_spacing * 1.5

    lines = []
    if len(pos[0]):
        seeds = outward_seeds(pos_polys, args.seeds, gap)
        # follow +E (positive-ion drift); stop at negative electrodes
        lines += trace_field_lines(seeds, cpos_all, q_all, neg[0],
                                   args.line_step, max_steps, (h, w), capture)
    if len(neg[0]):
        seeds = outward_seeds(neg_polys, args.seeds, gap)
        # trace backwards along -E from negatives; arrows still follow +E
        back = trace_field_lines(seeds, cpos_all, -q_all, pos[0],
                                 args.line_step, max_steps, (h, w), capture)
        lines += [l[::-1] for l in back]
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
        cv2.polylines(canvas, [line.astype(np.int32)], False, line_color, 1,
                      lineType=cv2.LINE_AA)
        E = efield(line, cpos_all, q_all, soft=args.line_step * 0.75)
        for frac in (0.3, 0.55, 0.8):
            k = int(len(line) * frac)
            if k < 1 or k >= len(line) - 1:
                continue
            draw_arrow(canvas, line[k], E[k], line_color)

    legend = (f"red +1.0 / blue -{args.charge_ratio:g}   "
              f"arrows: +ion drift")
    cv2.rectangle(canvas, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(canvas, legend, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    out = args.output or os.path.splitext(args.input_image)[0] + "_field.png"
    cv2.imwrite(out, canvas)
    print(f"wrote {out}")

    if not args.no_window:
        try:
            cv2.imshow("coulomb field (cyan), + red / - blue", canvas)
            print("close the preview window to finish (or press any key)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print(f"no display available; image saved to {out} instead")


if __name__ == "__main__":
    main()
