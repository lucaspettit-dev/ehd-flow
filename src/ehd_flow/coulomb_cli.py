"""CLI front end for :class:`ehd_flow.coulomb.CoulombField`.

Renders Coulomb field lines for image-defined electrodes (red channel ->
positive, blue channel -> negative). All physics lives in
:mod:`ehd_flow.coulomb`; this module only parses arguments and drives it.
"""

import argparse
import sys

from .coulomb import CoulombField


def build_parser():
    ap = argparse.ArgumentParser(
        description="Coulomb field lines for image-defined electrodes.")
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
    ap.add_argument("--ions-per-positive", type=int, default=10,
                    help="ion trajectories seeded per positive electrode "
                         "(0 disables ion tracing)")
    ap.add_argument("--ion-qm", type=float, default=2.0,
                    help="ion charge-to-mass ratio in sim units (higher = "
                         "lighter ion, follows field lines more closely)")
    ap.add_argument("--ion-dt", type=float, default=0.05,
                    help="ion integration timestep (sim time units)")
    ap.add_argument("--ion-max-steps", type=int, default=4000,
                    help="cap on ion integration steps")
    ap.add_argument("--ion-v0", type=float, default=0.0,
                    help="ion initial speed along local E (0 = start at rest)")
    ap.add_argument("--space-charge", type=float, default=0.5,
                    help="total + charge of the ion cloud fed back into "
                         "the field (electrodes total +/-1; 0 = ghost "
                         "test-particle ions)")
    ap.add_argument("--sc-iters", type=int, default=4,
                    help="max space-charge self-consistency rounds")
    ap.add_argument("--sc-relax", type=float, default=0.7,
                    help="under-relaxation factor for deposited charge")
    ap.add_argument("--sc-tol", type=float, default=1e-3,
                    help="stop iterating when grid |dE|/|E| < this")
    ap.add_argument("--sc-grid", type=int, default=32,
                    help="space-charge deposition grid cells across width")
    ap.add_argument("--hide-ion-paths", action="store_true",
                    help="keep ions in the physics (space charge) but "
                         "do not draw their magenta paths")
    ap.add_argument("--no-window", action="store_true",
                    help="do not try to open a preview window")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        field = CoulombField(charge_ratio=args.charge_ratio,
                             epsilon=args.epsilon,
                             min_area=args.min_area,
                             charge_spacing=args.charge_spacing,
                             seeds=args.seeds,
                             line_step=args.line_step,
                             ions_per_positive=args.ions_per_positive,
                             ion_qm=args.ion_qm,
                             ion_dt=args.ion_dt,
                             ion_max_steps=args.ion_max_steps,
                             ion_v0=args.ion_v0,
                             space_charge=args.space_charge,
                             sc_iters=args.sc_iters,
                             sc_relax=args.sc_relax,
                             sc_tol=args.sc_tol,
                             sc_grid=args.sc_grid,
                             draw_ion_paths=not args.hide_ion_paths)
    except (ImportError, ValueError) as e:
        sys.exit(f"error: {e}")

    try:
        out = field.render(args.input_image, args.output)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(f"wrote {out}")

    if not args.no_window:
        field.show(out)
