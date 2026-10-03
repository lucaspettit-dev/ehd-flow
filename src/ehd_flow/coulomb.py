"""CoulombField: electrostatic field lines for image-defined electrodes.

Single responsibility: the physics and rendering of the Coulomb field.
Electrode polygons come from :class:`ehd_flow.polygons.PolygonExtractor`
(red channel -> positive, blue channel -> negative); this class owns the
charge discretization, field summation, line tracing, and rendering.

Each electrode polygon is modeled as a uniformly charged conducting ring:
its total charge is spread over point charges along its boundary, and the
field is the Coulomb sum over all of them (k = 1, arbitrary units).

Field lines are seeded around the electrodes and integrated along E, drawn
thin and dashed. Separately, positive-ion trajectories are integrated with
full Newtonian dynamics (m*dv/dt = q*E, velocity Verlet), so their inertia
makes them deviate from the massless field lines -- the deviation is the
point of the overlay.

Charge ratio R sets |Q_negative| / |Q_positive| = R:
  R = 1   -> equal magnitude (default)
  R = 0.5 -> positive is 2x more charged than negative
  R = 2   -> negative is 2x more charged than positive

Ion dynamics note: the field uses k = 1 arbitrary units, so absolute SI
units cannot be carried through. The ion is taken as singly charged
(q = +e) with the mass of an average dry-air molecule
(m = 28.97 g/mol / N_A ~= 4.81e-26 kg); in sim units only the
charge-to-mass ratio ``ion_qm`` matters. Raise it for a lighter/faster
ion (tracks field lines more closely), lower it for more inertia and
larger deviation from the field lines.

Space charge: optionally, the ions are not just test particles. Each
iteration traces ion paths in the current field, deposits their
time-averaged charge onto a fixed grid over the domain (the traced
cohort stands in for a steady emission stream; ``space_charge`` is its
total charge in the same units where each electrode totals +/-1), and
re-traces both field lines and ion paths in the updated field. The
positive cloud shields the emitter and reshapes the gap field, as in a
real corona. Iterating stops when the field change on the deposition
grid falls below ``sc_tol`` or after ``sc_iters`` rounds; ``sc_relax``
under-relaxes the deposited charge for stability.
"""

import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .polygons import PolygonExtractor, show_image

# Ion properties (dry-air average molecule, singly charged positive ion)
ION_MASS_GMOL = 28.97
AVOGADRO = 6.02214076e23
ELEMENTARY_CHARGE = 1.602176634e-19
ION_MASS_KG = (ION_MASS_GMOL * 1e-3) / AVOGADRO  # ~= 4.81e-26 kg
ION_Q_OVER_M_SI = ELEMENTARY_CHARGE / ION_MASS_KG  # ~= 3.33e6 C/kg


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
    """Field lines (thin, dashed) + ion trajectories (solid) for electrodes.

    Parameters:
      charge_ratio   -- |Q_negative| / |Q_positive|.
      epsilon        -- polygon simplification as fraction of perimeter.
      min_area       -- ignore blobs smaller than this many px^2.
      charge_spacing -- spacing (px) of discrete charges along edges.
      seeds          -- field-line seeds per electrode.
      line_step      -- field-line integration step (px).
      ions_per_positive -- ion trajectories seeded per positive electrode
                        (0 disables ion tracing).
      ion_qm         -- ion charge-to-mass ratio in sim units (higher =
                        lighter ion, follows field lines more closely).
      ion_dt         -- ion integration timestep (sim time units).
      ion_max_steps  -- cap on ion integration steps.
      ion_v0         -- ion initial speed along local E (0 = start at rest).
      space_charge   -- total + charge of the ion cloud deposited back
                        into the field (electrodes total +/-1; 0 = ions
                        stay ghost test particles).
      sc_iters       -- max space-charge self-consistency iterations.
      sc_relax       -- under-relaxation factor for deposited charge.
      sc_tol         -- stop iterating when grid field change < this.
      sc_grid        -- space-charge deposition grid cells across width.
    """

    def __init__(self, charge_ratio=1.0, epsilon=0.002, min_area=10.0,
                 charge_spacing=3.0, seeds=48, line_step=2.5,
                 ions_per_positive=10, ion_qm=2.0, ion_dt=0.05,
                 ion_max_steps=4000, ion_v0=0.0,
                 space_charge=0.5, sc_iters=4, sc_relax=0.7,
                 sc_tol=1e-3, sc_grid=32):
        if cv2 is None:
            raise ImportError(
                "opencv-python is required (pip install opencv-python)")
        if charge_ratio < 0:
            raise ValueError("charge_ratio must be >= 0")
        self.charge_ratio = charge_ratio
        self.charge_spacing = charge_spacing
        self.seeds = seeds
        self.line_step = line_step
        self.ions_per_positive = ions_per_positive
        self.ion_qm = ion_qm
        self.ion_dt = ion_dt
        self.ion_max_steps = ion_max_steps
        self.ion_v0 = ion_v0
        self.space_charge = space_charge
        self.sc_iters = sc_iters
        self.sc_relax = sc_relax
        self.sc_tol = sc_tol
        self.sc_grid = sc_grid
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

    def outward_seeds(self, polygons, gap, count=None):
        """Seed points just outside each polygon boundary.

        Uses cv2.pointPolygonTest so concave shapes get true outward
        normals. Returns (S, 2) array of seed points.
        """
        count = self.seeds if count is None else count
        seeds = []
        contours = [np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
                    for _, pts in polygons]
        for contour in contours:
            n = len(contour)
            stride = max(1, n // count)
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

    def trace_ion_paths(self, seeds, cpos_all, q_all, cpos_stop, bounds,
                        capture, solid_mask=None):
        """Integrate m*dv/dt = q*E (velocity Verlet) for positive ions.

        Unlike field lines (massless drift along E/|E|), ions carry
        momentum, so their paths curve and overshoot where the field
        bends -- the deviation from the field lines is the point.

        All ions advance in lockstep so the field is evaluated in one
        batched call per step. The step is adapted every iteration so
        each step moves the fastest ion by at most ~half the charge
        spacing (and bounds the acceleration displacement the same
        way): ions crawl through weak-field regions in few large
        steps and resolve the strong field near electrodes finely.
        Because dt varies, every saved point carries the dt it
        represents, so charge deposition can weight by residence time.

        Electrodes are solid: when ``solid_mask`` (an image-sized 0/1
        array with every electrode filled) is given, each step's
        segment is checked against it and an ion that reaches any
        electrode stops there, its path clipped at the surface. This
        matters because the field inside a charged ring is weak, so
        an unchecked fast ion would coast straight through a positive
        electrode. An ion also stops when captured near a negative
        electrode's charges, when it leaves the domain, or at
        ion_max_steps. Returns (paths, weights): paths are (K, 2)
        polylines, weights the (K,) dt each point represents.
        """
        h, w = bounds
        qm = self.ion_qm
        soft = self.line_step * 0.75
        cfl = 0.5 * self.charge_spacing          # max move per step (px)
        dt_max = 40.0 * self.ion_dt
        P = np.asarray(seeds, dtype=float).reshape(-1, 2).copy()
        if len(P) == 0:
            return [], []
        E = efield(P, cpos_all, q_all, soft=soft)
        norms = np.linalg.norm(E, axis=1)
        V = np.where(norms[:, None] > 1e-12,
                     E / np.maximum(norms, 1e-300)[:, None] * self.ion_v0,
                     0.0)
        A = qm * E
        alive = np.ones(len(P), dtype=bool)
        paths = [[p.copy()] for p in P]
        weights = [[0.0] for _ in P]
        dt = self.ion_dt
        for _ in range(self.ion_max_steps):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            # adapt dt: fastest displacement / acceleration limits it
            vmax = float(np.linalg.norm(V[idx], axis=1).max(initial=0.0))
            amax = float(np.linalg.norm(A[idx], axis=1).max(initial=0.0))
            dt_step = dt_max
            if vmax > 1e-12:
                dt_step = min(dt_step, cfl / vmax)
            if amax > 1e-12:
                dt_step = min(dt_step, np.sqrt(2.0 * cfl / amax))
            dt = max(min(dt_step, 1.5 * dt), 1e-6 * dt_max)
            V[idx] += 0.5 * A[idx] * dt       # half kick
            P[idx] += V[idx] * dt             # drift
            for i in idx:
                if solid_mask is not None:
                    hit = self._segment_solid_hit(paths[i][-1], P[i],
                                                  solid_mask)
                    if hit is not None:
                        point, frac = hit
                        paths[i].append(point)
                        weights[i].append(dt * frac)
                        P[i] = point
                        alive[i] = False
                        continue
                paths[i].append(P[i].copy())
                weights[i].append(dt)
            alive &= ((P[:, 0] >= 0) & (P[:, 0] <= w) &
                      (P[:, 1] >= 0) & (P[:, 1] <= h))
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            if len(cpos_stop):
                d = np.linalg.norm(
                    P[idx][:, None, :] - cpos_stop[None, :, :], axis=2)
                alive[idx[d.min(axis=1) < capture]] = False
                idx = np.flatnonzero(alive)
                if idx.size == 0:
                    break
            A[idx] = qm * efield(P[idx], cpos_all, q_all, soft=soft)
            V[idx] += 0.5 * A[idx] * dt       # second half kick
        keep = [i for i, p in enumerate(paths) if len(p) > 4]
        return ([np.asarray(paths[i]) for i in keep],
                [np.asarray(weights[i]) for i in keep])

    @staticmethod
    def _segment_solid_hit(a, b, mask):
        """First entry point of segment a->b into a solid pixel.

        Samples the segment at sub-pixel spacing (so no single step
        can hop over a thin electrode) and bisects the entry bracket
        to refine the surface crossing. Returns (point, frac) with
        frac the fraction of the segment travelled, or None.
        """
        mh, mw = mask.shape

        def solid(x, y):
            xi = int(round(x))
            yi = int(round(y))
            return 0 <= xi < mw and 0 <= yi < mh and mask[yi, xi] > 0

        seg = b - a
        length = float(np.linalg.norm(seg))
        n = max(1, int(np.ceil(length / 0.75)))
        prev = a
        for k in range(1, n + 1):
            p = a + seg * (k / n)
            if solid(p[0], p[1]):
                lo, hi = prev, p
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    if solid(mid[0], mid[1]):
                        hi = mid
                    else:
                        lo = mid
                frac = float(np.linalg.norm(hi - a) / max(length, 1e-300))
                return hi, min(max(frac, 0.0), 1.0)
            prev = p
        return None

    def _deposit_grid(self, bounds):
        """Fixed space-charge deposition grid: (centers (K,2), nx, ny).

        Cell centers span the image; the same grid is reused every
        iteration so charge vectors can be relaxed and compared.
        """
        h, w = bounds
        nx = max(4, int(self.sc_grid))
        ny = max(4, int(round(self.sc_grid * h / w)))
        xs = (np.arange(nx) + 0.5) * w / nx
        ys = (np.arange(ny) + 0.5) * h / ny
        cx, cy = np.meshgrid(xs, ys)
        return np.column_stack([cx.ravel(), cy.ravel()]), nx, ny

    def _deposit_space_charge(self, ion_paths, path_weights, bounds, nx, ny):
        """Time-averaged ion charge per grid cell, normalized so the
        total equals self.space_charge.

        Each path point is weighted by the dt it represents, so a cell
        accumulates charge in proportion to residence time: slow
        regions (where ions linger) get more charge.
        Returns (K,) charge per cell center.
        """
        h, w = bounds
        counts = np.zeros(nx * ny)
        for path, wts in zip(ion_paths, path_weights):
            ix = np.clip((path[:, 0] / w * nx).astype(int), 0, nx - 1)
            iy = np.clip((path[:, 1] / h * ny).astype(int), 0, ny - 1)
            np.add.at(counts, iy * nx + ix, wts)
        total = counts.sum()
        if total <= 0:
            return counts
        return counts * (self.space_charge / total)

    @staticmethod
    def draw_dashed_polyline(img, pts, color, thickness=1, dash=(7, 5)):
        """Draw a dashed polyline (OpenCV has no native dashed lines)."""
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return
        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1]
        dash_len, gap_len = dash
        s = 0.0
        while s < total:
            e = min(s + dash_len, total)
            # interpolate sub-segment endpoints along the polyline
            sub = []
            for target in (s, e):
                k = int(np.searchsorted(cum, target, side='right')) - 1
                k = min(max(k, 0), len(pts) - 2)
                t = (target - cum[k]) / max(seg_len[k], 1e-12)
                sub.append(pts[k] + t * seg[k])
            cv2.line(img, tuple(np.round(sub[0]).astype(int)),
                     tuple(np.round(sub[1]).astype(int)),
                     color, thickness, lineType=cv2.LINE_AA)
            s = e + gap_len

    # -- full pipeline ---------------------------------------------------

    def render(self, input_image, output=None):
        """Run the full pipeline: image -> field-line + ion-path PNG.

        Returns the output path.
        """
        pos_polys, neg_polys, color = self.electrodes_from_image(input_image)
        h, w = color.shape[:2]

        pos = self.discretize_boundary(pos_polys, +1.0)
        neg = self.discretize_boundary(neg_polys, -self.charge_ratio)
        elec_pos = np.vstack([p for p, _ in (pos, neg) if len(p)])
        elec_q = np.concatenate([q for _, q in (pos, neg) if len(q)])

        gap = self.charge_spacing
        max_steps = int(2.5 * np.hypot(w, h) / self.line_step)
        capture = self.charge_spacing * 1.5

        # filled-electrode mask: ions stop at any electrode surface
        solid = np.zeros((h, w), dtype=np.uint8)
        for _, pts in list(pos_polys) + list(neg_polys):
            cv2.fillPoly(solid, [np.asarray(pts, dtype=np.int32)], 1)

        # ---- self-consistent space-charge iteration ----
        # Start from the bare electrode field; each round re-traces the
        # ion paths, deposits their charge on a fixed grid, relaxes it
        # into the active charge set, and stops when the field on the
        # grid stops changing. Final lines/paths are re-traced in the
        # converged field so the render is self-consistent.
        sc_enabled = self.space_charge > 0 and self.ions_per_positive > 0
        centers, nx, ny = self._deposit_grid((h, w))
        sc_qv = np.zeros(len(centers))
        n_rounds = max(1, self.sc_iters) if sc_enabled else 1
        soft = self.line_step * 0.75

        def combined(sc):
            keep = sc > 1e-12
            if not keep.any():
                return elec_pos, elec_q
            return (np.vstack([elec_pos, centers[keep]]),
                    np.concatenate([elec_q, sc[keep]]))

        def trace_all(cpos_all, q_all, want_lines=True):
            lines, ion_paths, path_weights = [], [], []
            if len(pos[0]):
                if want_lines:
                    seeds = self.outward_seeds(pos_polys, gap)
                    lines += self.trace_field_lines(
                        seeds, cpos_all, q_all, neg[0], (h, w), capture,
                        max_steps)
                if self.ions_per_positive > 0:
                    ion_seeds = self.outward_seeds(
                        pos_polys, gap, count=self.ions_per_positive)
                    ion_paths, path_weights = self.trace_ion_paths(
                        ion_seeds, cpos_all, q_all, neg[0], (h, w), capture,
                        solid_mask=solid)
            if want_lines and len(neg[0]):
                seeds = self.outward_seeds(neg_polys, gap)
                back = self.trace_field_lines(
                    seeds, cpos_all, -q_all, pos[0], (h, w), capture,
                    max_steps)
                lines += [line[::-1] for line in back]
            return lines, ion_paths, path_weights

        cpos_all, q_all = elec_pos, elec_q
        probe_E_prev = None
        if sc_enabled:
            for it in range(n_rounds):
                cpos_all, q_all = combined(sc_qv)
                probe_E = efield(centers, cpos_all, q_all, soft=soft)
                if probe_E_prev is not None:
                    # rel. field change (RMS over grid) vs previous round
                    num = np.sqrt(((probe_E - probe_E_prev) ** 2)
                                  .sum(axis=1).mean())
                    den = np.sqrt((probe_E_prev ** 2)
                                  .sum(axis=1).mean()) + 1e-300
                    print(f"space-charge round {it}: |dE|/|E| = "
                          f"{num / den:.2e} (cloud charge "
                          f"{sc_qv.sum():.3g})")
                    if num / den < self.sc_tol:
                        break
                probe_E_prev = probe_E
                _, tmp_paths, tmp_weights = trace_all(cpos_all, q_all,
                                                      want_lines=False)
                dep = self._deposit_space_charge(tmp_paths, tmp_weights,
                                                 (h, w), nx, ny)
                sc_qv = (1.0 - self.sc_relax) * sc_qv + self.sc_relax * dep
            cpos_all, q_all = combined(sc_qv)

        lines, ion_paths, _ = trace_all(cpos_all, q_all)
        print(f"traced {len(lines)} field lines, {len(ion_paths)} ion paths")

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
            self.draw_dashed_polyline(canvas, line, line_color, thickness=1,
                                      dash=(7, 5))
        ion_color = (255, 0, 255)  # magenta in BGR
        for path in ion_paths:
            cv2.polylines(canvas, [path.astype(np.int32)], False, ion_color,
                          2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, tuple(path[0].astype(int)), 3, ion_color, -1,
                       lineType=cv2.LINE_AA)

        legend = (f"+ red / - blue (R={self.charge_ratio:g})   "
                  f"cyan dashed: field lines   magenta: ion paths")
        cv2.rectangle(canvas, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(canvas, legend, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        out = output or os.path.splitext(input_image)[0] + "_field.png"
        cv2.imwrite(out, canvas)
        return out

    def show(self, path):
        """Open the rendered image in a preview window (headless-safe)."""
        show_image(path, "coulomb field: cyan dashed field lines, "
                         "magenta ion paths (+ red / - blue)")
