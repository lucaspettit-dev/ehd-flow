#!/usr/bin/env python3
"""image_to_polygons.py -- turn blobs in an image into polygon obstacles.

Pipeline:
  1. Load the input image and convert to grayscale.
  2. Binarize: pixel values < 255/2 become 0, values > 255/2 become 1
     (a pixel exactly at 127.5 maps to 0).
  3. Find the external contour of every connected blob of 1-valued pixels.
     Holes are ignored, so a doughnut-shaped blob becomes a plain circle.
  4. Simplify each contour into a polygon (Douglas-Peucker).
  5. Write the polygons as JSON in the ehd_flow.py obstacle format and
     show the black-and-white image with the polygon borders drawn on it.

Usage:
    python image_to_polygons.py input.png obstacles.json
    python image_to_polygons.py input.png obstacles.json --epsilon 0.005 \\
        --domain-x 2.0 --domain-y 1.0 --min-area 25

Notes:
  - JSON coordinates are in pixel units with the origin at the top-left
    (x right, y down), matching how ehd_flow.py rasterizes the grid, unless
    --domain-x/--domain-y rescale them into physical units.
  - The preview window needs a display; on headless machines it is skipped
    and only the preview PNG is written.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("error: opencv-python is required (pip install opencv-python)")

THRESHOLD = 255 / 2  # 127.5


def load_binary(path):
    """Load image, grayscale it, binarize at 255/2 -> {0, 1} uint8."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if img.ndim == 3:
        # drop alpha if present, then BGR -> gray
        img = img[:, :, :3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # THRESH_BINARY: dst = 1 where src > 127.5, else 0
    _, binary = cv2.threshold(gray, THRESHOLD, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def extract_polygons(binary, epsilon_ratio=0.002, min_area=10.0,
                     scale_x=1.0, scale_y=1.0):
    """Return a list of (name, points) polygons from the binary image."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # deterministic order: biggest blob first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygons = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
        pts = pts.astype(float)
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        polygons.append((f"blob_{i}", pts.tolist()))
    return polygons


def draw_preview(binary, polygons, scale_x=1.0, scale_y=1.0):
    """Black-and-white image with each polygon drawn as a border."""
    h, w = binary.shape
    canvas = (binary * 255).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for _, pts in polygons:
        arr = np.asarray(pts, dtype=float)
        arr[:, 0] /= scale_x
        arr[:, 1] /= scale_y
        cv2.polylines(canvas, [arr.astype(np.int32)], True, (0, 255, 0), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    args = ap.parse_args()

    binary = load_binary(args.input_image)
    h, w = binary.shape

    dom_x = args.domain_x if args.domain_x else float(w)
    dom_y = args.domain_y if args.domain_y else float(h)
    sx, sy = dom_x / w, dom_y / h

    polygons = extract_polygons(binary, args.epsilon, args.min_area, sx, sy)
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

    preview = draw_preview(binary, polygons, sx, sy)
    preview_path = os.path.splitext(args.output_json)[0] + "_preview.png"
    cv2.imwrite(preview_path, preview)
    print(f"wrote {preview_path}")

    if not args.no_window:
        try:
            cv2.imshow("polygons (green borders on binary image)", preview)
            print("close the preview window to finish (or press any key)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("no display available; preview saved to "
                  f"{preview_path} instead")


if __name__ == "__main__":
    main()
