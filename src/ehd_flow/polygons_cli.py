"""CLI front end for :class:`ehd_flow.polygons.PolygonExtractor`.

Turns blobs in an image into polygon obstacles and writes them as JSON in
the solver's obstacle format. All image/polygon work lives in
:mod:`ehd_flow.polygons`; this module only parses arguments and drives it.
"""

import argparse
import json
import os
import sys

from .polygons import PolygonExtractor, show_image


def build_parser():
    ap = argparse.ArgumentParser(
        description="Turn blobs in an image into polygon obstacles.")
    ap.add_argument("input_image", help="input image file (any format)")
    ap.add_argument("output_json", help="output JSON file for the polygons")
    ap.add_argument("--epsilon", type=float, default=0.002,
                    help="Douglas-Peucker simplification as a fraction of "
                         "contour perimeter (default 0.002)")
    ap.add_argument("--min-area", type=float, default=10.0,
                    help="ignore blobs smaller than this many px^2 "
                         "(default 10)")
    ap.add_argument("--domain-x", type=float, default=None,
                    help="rescale x into this domain width "
                         "(default: image width in px)")
    ap.add_argument("--domain-y", type=float, default=None,
                    help="rescale y into this domain height "
                         "(default: image height in px)")
    ap.add_argument("--no-window", action="store_true",
                    help="do not try to open a preview window")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        extractor = PolygonExtractor(args.epsilon, args.min_area)
    except ImportError as e:
        sys.exit(f"error: {e}")

    try:
        binary = extractor.load_binary(args.input_image)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    h, w = binary.shape

    dom_x = args.domain_x if args.domain_x else float(w)
    dom_y = args.domain_y if args.domain_y else float(h)
    sx, sy = dom_x / w, dom_y / h

    polygons = extractor.extract_polygons(binary, sx, sy)
    print(f"found {len(polygons)} polygon(s) in {args.input_image} "
          f"({w}x{h}px)")

    case = {
        "domain": {"x": dom_x, "y": dom_y},
        "polygons": [{"name": name, "points": pts}
                     for name, pts in polygons],
    }
    with open(args.output_json, "w") as f:
        json.dump(case, f, indent=2)
    print(f"wrote {args.output_json}")

    preview_path = os.path.splitext(args.output_json)[0] + "_preview.png"
    extractor.save_preview(preview_path,
                           extractor.draw_preview(binary, polygons, sx, sy))
    print(f"wrote {preview_path}")

    if not args.no_window:
        show_image(preview_path,
                   "polygons (green borders on binary image)")
