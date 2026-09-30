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
                             line_step=args.line_step)
    except (ImportError, ValueError) as e:
        sys.exit(f"error: {e}")

    try:
        out = field.render(args.input_image, args.output)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(f"wrote {out}")

    if not args.no_window:
        field.show(out)
